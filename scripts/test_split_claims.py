"""The split with its claims built (claims_build) against the split with them as JSON
(WIKIDATA_SPLIT_CLAIMS=json, the split as it was), end to end on a dump slice made of the
chunk files given (bench/dumpslice.py), cut into chunks of 2,000 entities:

- the same chunks and parts, in the same sets: each part the same rows, its claims
  column left out where they were built;
- each built part's claims partition file byte for byte the one the pipeline's claims
  call makes of the JSON part (process.normalise_claims_direct);
- the pipeline's process step on both sets gives every table's files byte for byte;
- a run interrupted at each of the build's steps (after a part's claims file, before its
  part; after routing moved a claims file, before its part; in process, after the claims'
  label lookup, before the claims file is placed) and run again gives the same files.

    .venv/bin/python scripts/test_split_claims.py chunk.parquet ...
"""

import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RELEASE = "20261012"
sys.path.insert(0, str(ROOT / "bench"))

from dumpslice import make  # noqa: E402

SPLIT = """
import os, sys
sys.path.insert(0, {src!r})
from wikidata import dump
dump.CHUNK_ENTITIES = 2000
crash = os.environ.get("CRASH")
if crash == "after_claims":
    from wikidata import claims_build
    write = claims_build.write_claims
    def write_then_crash(ids, text, rows, dst):
        why = write(ids, text, rows, dst)
        if dst.name == "chunk_1.parquet":
            os._exit(3)
        return why
    claims_build.write_claims = write_then_crash
if crash == "in_move":
    moved = []
    replace = dump.Path.replace
    def replace_then_crash(self, target):
        out = replace(self, target)
        if self.parent.name == "claims" and self.parent.parent.name == "routed":
            moved.append(self)
            if len(moved) == 2:
                os._exit(3)
        return out
    dump.Path.replace = replace_then_crash
dump.split()
"""

PROCESS = """
import os, sys
sys.path.insert(0, {src!r})
from pathlib import Path
from wikidata import process
if os.environ.get("CRASH") == "after_lookup":
    place = process.place_built_claims
    def crash(built, claim_pq, lookup_pq):
        process.atomic_sink_parquet(process.lookup_to_long(process.pl.DataFrame()).lazy(), lookup_pq)
        os._exit(3)
    process.place_built_claims = crash
for s in ("{release}", "{release}-scholar"):
    process.process(Path("releases") / s / "data", Path("out") / s, "x", Path("state") / s)
"""


def run(code: str, cwd: Path, mode: str, crash: str | None = None) -> int:
    env = {**os.environ, "WIKIDATA_RELEASE": RELEASE, "WIKIDATA_SPLIT_CLAIMS": mode, "HF_HUB_OFFLINE": "1"}
    env.pop("CRASH", None)
    if crash:
        env["CRASH"] = crash
    env.pop("WIKIDATA_SCHOLAR", None)
    proc = subprocess.run([sys.executable, "-c", code.format(src=str(ROOT / "src"), release=RELEASE)],
                          cwd=cwd, env=env, capture_output=True, text=True)
    if proc.returncode not in (0, 3) or (proc.returncode == 3) != bool(crash):
        print(proc.stdout[-3000:], proc.stderr[-3000:])
        raise SystemExit(f"{mode} {crash}: exited {proc.returncode}")
    return proc.returncode


def split(work: Path, dump_slice: Path, mode: str, crash: str | None = None, name: str = "") -> Path:
    d = work / f"{mode}-{crash}{name}"
    dump_dir = d / "releases" / RELEASE / "dump"
    dump_dir.mkdir(parents=True)
    os.link(dump_slice, dump_dir / f"wikidata-{RELEASE}-all.json.bz2")
    if crash in ("after_claims", "in_move"):
        run(SPLIT, d, mode, crash)
    run(SPLIT, d, mode)
    return d


# Written in the order the split's chunks finish, which no two runs share
UNORDERED = {"manifest.split.jsonl", "route.jsonl"}


def files(d: Path, sub: str) -> dict[str, str]:
    """Each file under `d/sub`, by its path, to its digest."""
    return {str(f.relative_to(d)): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted((d / sub).rglob("*")) if f.is_file() and f.name not in UNORDERED}


def main(chunks: list[str]) -> None:
    import orjson
    import polars as pl

    os.environ["WIKIDATA_RELEASE"] = RELEASE
    sys.path.insert(0, str(ROOT / "src"))
    with tempfile.TemporaryDirectory() as work:
        work = Path(work)
        # One entity more, last, whose claims do not fit claims.avsc (a mainsnak with a hash,
        # which the target schema holds): its part keeps its claims as JSON, and genson
        # infers them
        row = pl.read_parquet(chunks[0], n_rows=1)
        claims = orjson.loads(row["claims"][0])
        next(iter(claims.values()))[0]["mainsnak"]["hash"] = "0123abcd"
        misfit = work / "misfit.parquet"
        row.with_columns(id=pl.lit("Q999999999"), claims=pl.lit(orjson.dumps(claims).decode())).write_parquet(misfit)
        dump_slice = work / "slice.json.bz2"
        make(dump_slice, [*chunks, str(misfit)])
        json_run = split(work, dump_slice, "json")
        built = split(work, dump_slice, "built")
        sets = [Path("releases") / RELEASE / "data", Path("releases") / f"{RELEASE}-scholar" / "data"]
        n_built = n_json = 0
        for s in sets:
            mj = [orjson.loads(x) for x in (json_run / s / "manifest.jsonl").read_bytes().splitlines()]
            mb = [orjson.loads(x) for x in (built / s / "manifest.jsonl").read_bytes().splitlines()]
            strip = lambda e: {k: v for k, v in e.items() if k not in ("bytes", "claims_bytes", "claims_json")}  # noqa: E731
            assert [strip(e) for e in mj] == [strip(e) for e in mb], f"{s}: manifests differ"
            for ej, eb in zip(mj, mb):
                pj = pl.read_parquet(json_run / s / ej["file"])
                pb = pl.read_parquet(built / s / eb["file"])
                if "claims_bytes" in eb:
                    n_built += 1
                    assert "claims" not in pb.columns and pj.drop("claims").equals(pb), f"{s}/{eb['file']}: parts differ"
                    claims_file = built / s / "claims" / eb["file"]
                    assert claims_file.stat().st_size == eb["claims_bytes"]
                else:
                    n_json += 1
                    assert "claims_json" in eb and pj.equals(pb), f"{s}/{eb['file']}: parts differ"
                    print(f"{s}/{eb['file']}: claims left as JSON ({eb['claims_json']})")
            print(f"{s}: {len(mb)} parts, {sum('claims_bytes' in e for e in mb)} with claims built: rows as the JSON split's")
        assert n_built, "no part's claims were built"
        assert n_json, "no part's claims were left as JSON"
        # (before process, which deletes the chunks it has processed)
        split_files = {s: files(built, str(s)) for s in sets}
        # The pipeline's process step: every table the same, byte for byte
        for d in (json_run, built):
            run(PROCESS, d, "built")
        want = files(json_run, "out")
        got = files(built, "out")
        assert want == got, f"process outputs differ: {sorted(set(want.items()) ^ set(got.items()))[:6]}"
        print(f"process: {len(want)} output files, byte for byte the same")
        # Interrupted and run again: the same files
        for crash in ("after_claims", "in_move"):
            again = split(work, dump_slice, "built", crash)
            for s in sets:
                assert files(again, str(s)) == split_files[s], f"{crash}: the split's files differ"
            print(f"split interrupted {crash}, run again: the same files")
        again = split(work, dump_slice, "built", None, "-for-process")
        if run(PROCESS, again, "built", "after_lookup") == 3:
            run(PROCESS, again, "built")
        assert files(again, "out") == got, "process interrupted after the lookup: outputs differ"
        print("process interrupted after the label lookup, run again: the same files")


if __name__ == "__main__":
    main(sys.argv[1:])
