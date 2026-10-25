"""The split with its claims and entities built (claims_build, entities_build) against the
split with them as JSON (WIKIDATA_SPLIT_CLAIMS=json, WIKIDATA_SPLIT_ENTITIES=json: the split
as it was), end to end on a dump slice made of the chunk files given (bench/dumpslice.py),
cut into chunks of 2,000 entities:

- the same chunks and parts, in the same sets: each part the same rows, its claims and
  `entity` columns left out where they were built;
- the main set's partition step: the entities placed as their one partition
  (place_unsplit) the same, in rows, order, schema and sidecar (but its file_size), as
  partition_parquet's re-encode of the processed file, the pipeline's partition before;
- the pipeline's process step on both sets gives every table's files byte for byte, but
  the claims', the same in their rows, order, schema and row groups (their encoding is
  each row group's own, as the split writes them a row group at a time);
- a run interrupted at each of the build's steps (after a part's claims file, before its
  part; after routing moved a claims file, before its part; in process, after the claims'
  label lookup, before the claims file is placed) and run again gives the same files;
- the parts numbered as the split goes: interrupted once the first two chunks are
  written, their parts are in their sets' manifests and at their numbers, with their
  claims files, and nothing of theirs is left in `routed/`; run again, the same files.

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
sys.path.insert(0, str(ROOT / "scripts"))

from dumpslice import make  # noqa: E402

SPLIT = """
import os, sys
sys.path.insert(0, {src!r})
from wikidata import dump
dump.CHUNK_ENTITIES = 2000
crash = os.environ.get("CRASH")
if crash == "after_claims":
    from wikidata import claims_build
    # (a part's claims written from their text, or built from its lines: write_built)
    for name in ("write_claims", "write_built"):
        def write_then_crash(*args, write=getattr(claims_build, name)):
            why = write(*args)
            if args[-1].name == "chunk_1.parquet":
                os._exit(3)
            return why
        setattr(claims_build, name, write_then_crash)
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
if crash == "after_numbered":
    advance = dump._Numbering.advance
    def advance_then_crash(self, done):
        advance(self, done)
        if self.next >= 2:  # the first two chunks' parts numbered, the split not done
            os._exit(3)
    dump._Numbering.advance = advance_then_crash
dump.split()
"""

PROCESS = """
import os, sys
sys.path.insert(0, {src!r})
from pathlib import Path
import orjson
from wikidata import main, process
from wikidata.config import PARTITION_COLS, Table
from wikidata.partitioning import partition_parquet, prepare_for_partition
if os.environ.get("CRASH") == "after_lookup":
    place = process.place_built_claims
    def crash(built, claim_pq, lookup_pq):
        process.atomic_sink_parquet(process.lookup_to_long(process.pl.DataFrame()).lazy(), lookup_pq)
        os._exit(3)
    process.place_built_claims = crash
for s in ("{release}", "{release}-scholar"):
    process.process(Path("releases") / s / "data", Path("out") / s, "x", Path("state") / s)
if os.environ.get("PARTITION"):
    # The main set's chunks partitioned (partition_chunk), and its entities as the
    # pipeline partitioned them before (partition_parquet), into ref/, to compare
    s = "{release}"
    for f in sorted((Path("out") / s / "entities").glob("chunk_*.parquet")):
        partition_parquet(by=PARTITION_COLS[Table.ENTITIES], lf=prepare_for_partition(f, Table.ENTITIES),
                          source_name=f.name, dst_dir=Path("ref") / "entities", log_dir=Path("ref") / "audit")
    manifest = (Path("releases") / s / "data" / "manifest.jsonl").read_bytes().splitlines()
    for e in map(orjson.loads, manifest):
        main.partition_chunk(e["chunk"], Path("state") / s, Path("out") / s)
"""


def run(code: str, cwd: Path, mode: str, crash: str | None = None) -> int:
    env = {**os.environ, "WIKIDATA_RELEASE": RELEASE, "WIKIDATA_SPLIT_CLAIMS": mode, "WIKIDATA_SPLIT_ENTITIES": mode,
           "HF_HUB_OFFLINE": "1"}
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
    if crash in ("after_claims", "in_move", "after_numbered"):
        run(SPLIT, d, mode, crash)
        if crash == "after_numbered":
            numbered_midway(d)
    run(SPLIT, d, mode)
    return d


def numbered_midway(d: Path) -> None:
    """A split interrupted once its first two chunks were written: their parts numbered."""
    import orjson

    seen = set()
    for set_dir in (d / "releases" / RELEASE / "data", d / "releases" / f"{RELEASE}-scholar" / "data"):
        lines = [orjson.loads(x) for x in (set_dir / "manifest.jsonl").read_bytes().splitlines()]
        assert [e["chunk"] for e in lines] == list(range(len(lines))), "numbers not 0, 1, ..."
        for e in lines:
            seen.add(e["split_chunk"])
            assert (set_dir / e["file"]).exists(), f"{e['file']} not at its number"
            assert not (set_dir / "routed" / f"chunk_{e['split_chunk']}.parquet").exists()
            if "claims_bytes" in e:
                assert (set_dir / "claims" / e["file"]).stat().st_size == e["claims_bytes"]
    assert {0, 1} <= seen, f"the first two chunks' parts not numbered: {sorted(seen)}"
    assert not (d / "releases" / RELEASE / "data" / "split.done").exists()
    print(f"split interrupted after its first two chunks: {len(seen)} chunks' parts numbered, in place")


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
        # One entity more, last, whose claims do not fit claims.avsc (a mainsnak with a null
        # snaktype, which the target schema holds: claims.avsc's is never null): its part
        # keeps its claims as JSON, and genson infers them
        row = pl.read_parquet(chunks[0], n_rows=1)
        claims = orjson.loads(row["claims"][0])
        next(iter(claims.values()))[0]["mainsnak"]["snaktype"] = None
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
            built_keys = ("bytes", "claims_bytes", "claims_json", "entities_bytes", "entities_json")
            strip = lambda e: {k: v for k, v in e.items() if k not in built_keys}  # noqa: E731
            assert [strip(e) for e in mj] == [strip(e) for e in mb], f"{s}: manifests differ"
            for ej, eb in zip(mj, mb):
                pj = pl.read_parquet(json_run / s / ej["file"])
                pb = pl.read_parquet(built / s / eb["file"])
                dropped = [c for c, k in (("claims", "claims_bytes"), ("entity", "entities_bytes")) if k in eb]
                assert not set(dropped) & set(pb.columns) and pj.drop(dropped).equals(pb), f"{s}/{eb['file']}: parts differ"
                for table in ("claims", "entities"):
                    if f"{table}_bytes" in eb:
                        assert (built / s / table / eb["file"]).stat().st_size == eb[f"{table}_bytes"]
                assert "entities_bytes" in eb, f"{s}/{eb['file']}: its entities were not built"
                if "claims_bytes" in eb:
                    n_built += 1
                else:
                    n_json += 1
                    assert "claims_json" in eb
                    print(f"{s}/{eb['file']}: claims left as JSON ({eb['claims_json']})")
            print(f"{s}: {len(mb)} parts, {sum('claims_bytes' in e for e in mb)} with claims built, all with entities "
                  "built: rows as the JSON split's")
        assert n_built, "no part's claims were built"
        assert n_json, "no part's claims were left as JSON"
        # (before process, which deletes the chunks it has processed)
        split_files = {s: files(built, str(s)) for s in sets}
        # The pipeline's process step: every table the same, byte for byte
        for d in (json_run, built):
            run(PROCESS, d, "built")
        want = files(json_run, "out")
        got = files(built, "out")
        assert want.keys() == got.keys(), "process outputs differ in their files"
        claims = [f for f in want if Path(f).parent.name == "claims" and want[f] != got[f]]
        from test_claims_build import same_partition
        for f in claims:
            assert same_partition(json_run / f, built / f), f"{f} differs"
        others = [f for f in want if f not in claims]
        assert all(want[f] == got[f] for f in others), "process outputs differ"
        print(f"process: {len(others)} output files byte for byte the same, {len(claims)} claims "
              f"files the same in their rows, order, schema and row groups")
        # The main set's partition step: the entities placed as their one partition, as
        # partition_parquet partitioned them before
        os.environ["PARTITION"] = "1"
        run(PROCESS, built, "built")
        del os.environ["PARTITION"]
        from test_claims_build import same_partition
        placed = sorted((built / "out" / RELEASE / "entities" / "all").glob("chunk_*.parquet"))
        assert placed
        for f in placed:
            ref = built / "ref" / "entities" / "all" / f.name
            assert same_partition(f, ref), f"{f.name}: the entities partition differs from partition_parquet's"
            side = lambda d: pl.read_parquet(d / f.name).drop("file_size", "path", strict=False)  # noqa: E731
            assert side(built / "releases" / RELEASE / "audit" / "entities").equals(side(built / "ref" / "audit")), \
                f"{f.name}: the entities sidecar differs"
        print(f"partition: {len(placed)} entities partition files placed, each as partition_parquet's in rows, "
              "order, schema, row groups and sidecar (but its size)")
        # Interrupted and run again: the same files
        for crash in ("after_claims", "in_move", "after_numbered"):
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
