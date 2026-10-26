"""The split's census of the dump's order (dump.chunk_census, dump._Census, dump.split_census;
docs/journal/2026-10-26-progressive-finalise.md), against a brute-force pass over the ids in
dump order:

- **the shard rule:** dump.shard_of is Wikibase's `hexdec(substr(sha1(id), 0, 8)) % 8`, and
  gives the shard every id of the journal's survey of wikidata-20261019 was found in (its
  `sample.csv.gz`, 30,059 ids);
- **a synthetic dump of 8 shards** (each a sweep over the ids sha1 puts in it, its head
  below Q10k shuffled, its body monotone to within 2, with stragglers planted on both sides
  of the margin and properties among them), cut into chunks that straddle the shards:
  every chunk's census and the summary are exact, whatever order the chunks finish in;
- **a restart:** the census stopped partway (some chunks finished but not yet written),
  resumed from the lines written, gives the same lines.

Given chunk files (bench/data's c0, r50, s30, s90), it also splits a dump slice made of
them (bench/dumpslice.py) in an order that plants a straggler, in chunks of 500 entities
with 3 workers, as the CLI runs it:

- the manifest's lines are in chunk order, and each one's census is the brute force's;
- a split killed (SIGKILL) partway and run again writes the same manifest and files;
- with --against SRC (another checkout's src, e.g. master's), every file the split writes
  is byte-identical to that checkout's split of the same slice, and the split's manifest
  is the same less its census.

    .venv/bin/python scripts/test_split_census.py [--against SRC] [c0.parquet r50.parquet s30.parquet s90.parquet]
"""

import argparse
import csv
import gzip
import hashlib
import json
import os
import random
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import orjson
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "bench"))

from wikidata import dump  # noqa: E402

SAMPLE = ROOT / "docs/journal/2026-10-26-progressive-finalise/layout-20261019/sample.csv.gz"
RELEASE = "20261012"
FIELDS = ("n", "first", "max", "before", "after", "lagged", "lag", "late", "head_lagged", "head_lag", "head_late",
          "other")


def reference(chunks: list[list[str]]) -> list[dict]:
    """Each chunk's census by brute force: id by id in dump order, each shard's running maximum."""
    running: dict[int, int] = {}
    out = []
    for ids in chunks:
        shards: dict[int, dict] = {}
        for i in ids:
            s = int(hashlib.sha1(i.encode()).hexdigest()[:8], 16) % 8  # (Wikibase's own form)
            c = shards.setdefault(s, {k: 0 for k in FIELDS} | {"first": None, "max": None,
                                                                 "before": running.get(s), "after": running.get(s)})
            c["n"] += 1
            if not i.startswith("Q"):
                c["other"] += 1
                continue
            x = int(i[1:])
            lag = max(0, running.get(s, -1) - x)
            p = "head_" if x < dump.CENSUS_HEAD else ""
            c[p + "lagged"] += lag > 0
            c[p + "lag"] = max(c[p + "lag"], lag)
            c[p + "late"] += lag > dump.CENSUS_MARGIN
            c["first"] = x if c["first"] is None else c["first"]
            c["max"] = x if c["max"] is None else max(c["max"], x)
            running[s] = max(running.get(s, -1), x)
            c["after"] = running[s]
        out.append({str(s): shards[s] for s in sorted(shards)})
    return out


def run_census(chunks: list[list[str]], order: list[int], stop: int | None = None,
               done: dict[int, dict] | None = None) -> tuple[list[dict], dict[int, dict]]:
    """The split's census (as dump.split runs it) of the chunks, finishing in `order`; with
    `stop`, only the first `stop` of them finish. Returns the lines written, in the order
    written, and the manifest they make (by chunk), resuming from `done`'s."""
    done = dict(done or {})
    census = dump._Census(done)
    written = []
    for n, i in enumerate(order):
        if stop is not None and n == stop:
            break
        entry = {"chunk": i, "rows": len(chunks[i]), dump.CENSUS: dump.chunk_census(pl.Series(chunks[i]))}
        census.add(entry, entry.pop(dump.CENSUS))
        for e in census.lines():
            e = orjson.loads(orjson.dumps(e))  # (as the manifest holds it)
            written.append(e)
            done[e["chunk"]] = e
    return written, done


def synthetic(seed: int = 0) -> list[str]:
    """A dump of 8 shards, one after another, each a sweep over the ids up to Q100,000 that
    sha1 puts in it: its head (below Q10k) shuffled in windows of 2,000 ids, its body
    monotone to within 2, two stragglers planted 10,001 below the running maximum and one
    exactly 10,000 below, a straggler in the head, and properties at their places."""
    rng = random.Random(seed)
    out = []
    for s in range(8):
        qs = [x for x in range(1, 100_000) if dump.shard_of(f"Q{x}") == s]
        head, body = [x for x in qs if x < 10_000], [x for x in qs if x >= 10_000]
        head = [x for w in range(0, 10_000, 2_000) for x in rng.sample(
            [y for y in head if w <= y < w + 2_000], len([y for y in head if w <= y < w + 2_000]))]
        for j in range(0, len(body) - 1, 7):  # within 2
            if body[j + 1] - body[j] <= 2:
                body[j], body[j + 1] = body[j + 1], body[j]
        seq = head + body
        # stragglers: an id moved to just after the id 10,001 (late) or 10,000 (not) above it
        for lag, at in ((10_001, 0.2), (10_001, 0.4), (10_000, 0.6)):
            records, top = set(), -1  # (the running maxima)
            for y in seq:
                if y > top:
                    records.add(y)
                    top = y
            x = next(x for x in body[int(at * len(body)):] if x + lag in records)
            seq.remove(x)
            seq.insert(seq.index(x + lag) + 1, x)
        # one straggler in the head: an early item long after its place (past Q15k)
        x = head[5]
        seq.remove(x)
        seq.insert(next(k for k, y in enumerate(seq) if y > 15_000) + 1, x)
        ids = [f"Q{x}" for x in seq]
        props = [p for p in range(1, 400) if dump.shard_of(f"P{p}") == s]
        for p in props:
            ids.insert(rng.randrange(len(ids)), f"P{p}")
        out += ids
    return out


def chunked(ids: list[str], size: int) -> list[list[str]]:
    return [ids[i:i + size] for i in range(0, len(ids), size)]


def check_rule() -> None:
    for x in range(1, 20_000):
        for i in (f"Q{x}", f"P{x}", f"L{x}"):
            assert dump.shard_of(i) == int(hashlib.sha1(i.encode()).hexdigest()[:8], 16) % 8, i
    with gzip.open(SAMPLE, "rt") as f:
        rows = list(csv.DictReader(f))
    bad = [r["id"] for r in rows if dump.shard_of(r["id"]) != int(r["shard"])]
    assert not bad, bad[:10]
    print(f"shard rule: Wikibase's on 60,000 ids, and the shard of each of the survey's {len(rows):,} ids")


def check_synthetic() -> None:
    ids = synthetic()
    chunks = chunked(ids, 997)  # (straddling the shards' ends)
    want = reference(chunks)
    lines, _ = run_census(chunks, list(range(len(chunks))))
    assert [e["shards"] for e in lines] == want
    for seed in range(5):
        order = random.Random(seed).sample(range(len(chunks)), len(chunks))
        got, done = run_census(chunks, order)
        assert [e["chunk"] for e in got] == list(range(len(chunks))), "lines not in chunk order"
        assert got == lines, f"seed {seed}: the census depends on the order the chunks finish in"
        # a restart: stopped with some chunks finished but not written, resumed from the lines
        stop = len(chunks) // 2
        first, done = run_census(chunks, order, stop=stop)
        assert len(first) < stop, "no chunk was finished and not yet written when it stopped"
        rest = [i for i in order if i not in done]
        second, done = run_census(chunks, random.Random(seed + 100).sample(rest, len(rest)), done=done)
        assert first + second == lines, f"seed {seed}: a restart changed the census"
    summary = dump.split_census(lines)
    a = summary["all"]
    assert summary["complete"] and summary["shards_seen"] == 8
    assert a["late"] == 16 and a["head_late"] == 8, a  # (2 a shard past the head, 1 in it)
    assert a["lag"] == 10_001 and a["head_lag"] > 5_000, a
    assert a["ids"] == len(ids) and a["other"] == sum(1 for i in ids if i[0] == "P")
    for s, t in summary["by_shard"].items():
        assert t["late"] == 2 and t["head_late"] == 1 and len(t["late_chunks"]) == 2, (s, t)
        assert t["max"] == max(int(i[1:]) for i in ids if i[0] == "Q" and dump.shard_of(i) == int(s))
    # a manifest begun before the census: incomplete
    assert not dump.split_census([{k: v for k, v in e.items() if k != "shards"} if e["chunk"] == 3 else e
                                  for e in lines])["complete"]
    print(f"synthetic dump: {len(ids):,} ids in {len(chunks)} chunks over 8 shards, exact in every order "
          f"and across a restart; {a['late']} stragglers past the heads (lag {a['lag']:,}), "
          f"{a['head_late']} in them")


def slice_files(files: dict[str, Path], work: Path) -> tuple[list[Path], list[str]]:
    """The chunk files cut and reordered as the slice's pieces: shard 0's head and body with
    one of its ids (Q12890) moved after Q29k, a straggler, and other shards' runs between."""
    c0 = pl.read_parquet(files["c0"])
    at = c0["id"].to_list().index("Q12890")
    pieces = [c0[:2000].filter(pl.col("id") != "Q12890"), pl.read_parquet(files["r50"])[:800], c0[2000:3500],
              c0[at:at + 1], pl.read_parquet(files["s30"])[:800], pl.read_parquet(files["s90"])[:800], c0[3500:3600]]
    out = []
    for k, p in enumerate(pieces):
        out.append(work / f"piece{k}.parquet")
        p.write_parquet(out[-1])
    return out, [i for p in pieces for i in p["id"].to_list()]


CODE = """
import sys
sys.path.insert(0, {src!r})
from wikidata import dump
dump.CHUNK_ENTITIES = 500
dump.split(workers=3)
"""


def split(src: Path, d: Path, dump_slice: Path, kill_after: int | None = None) -> None:
    (d / "releases" / RELEASE / "dump").mkdir(parents=True, exist_ok=True)
    target = d / "releases" / RELEASE / "dump" / f"wikidata-{RELEASE}-all.json.bz2"
    if not target.exists():
        os.link(dump_slice, target)
    env = {**os.environ, "WIKIDATA_RELEASE": RELEASE, "HF_HUB_OFFLINE": "1"}
    proc = subprocess.Popen([sys.executable, "-c", CODE.format(src=str(src))], cwd=d, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
    if kill_after is not None:
        manifest = d / "releases" / RELEASE / "data" / dump.SPLIT_MANIFEST.name
        while proc.poll() is None:
            if manifest.exists() and manifest.read_bytes().count(b"\n") >= kill_after:
                os.killpg(proc.pid, signal.SIGKILL)  # (its own session: the split and its decoder)
                break
            time.sleep(0.05)
        proc.wait()
        assert proc.returncode != 0, "the split ended before it was killed"
        return
    out = proc.communicate()[0].decode()
    assert proc.returncode == 0, out[-3000:]
    return out


def files_of(d: Path) -> dict[str, str]:
    return {str(p.relative_to(d)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((d / "releases").rglob("*")) if p.is_file() and "dump" not in p.relative_to(d).parts}


def lines_of(d: Path) -> list[dict]:
    path = d / "releases" / RELEASE / "data" / dump.SPLIT_MANIFEST.name
    return [orjson.loads(line) for line in path.read_bytes().splitlines() if line]


def check_split(files: dict[str, Path], against: Path | None) -> None:
    from dumpslice import make

    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as work:
        work = Path(work)
        pieces, ids = slice_files(files, work)
        dump_slice = work / "slice.json.bz2"
        make(dump_slice, [str(p) for p in pieces])
        out = split(ROOT / "src", work / "census", dump_slice)
        lines = lines_of(work / "census")
        assert [e["chunk"] for e in lines] == list(range(len(lines))), "the manifest is not in chunk order"
        want = reference(chunked(ids, 500))
        assert [e["shards"] for e in lines] == want, "the census is not the brute force's"
        summary = dump.split_census(lines)
        a = summary["all"]
        assert summary["complete"] and a["late"] == 1 and a["ids"] == len(ids), summary
        census_line = next(line for line in out.splitlines() if line.startswith("Census"))
        print(f"split of a slice ({len(ids):,} entities, {len(lines)} chunks, shards "
              f"{sorted(summary['by_shard'])}): every line in chunk order and exact; {census_line}")
        # killed partway, then run again
        split(ROOT / "src", work / "killed", dump_slice, kill_after=3)
        killed_at = len(lines_of(work / "killed"))
        split(ROOT / "src", work / "killed", dump_slice)
        assert lines_of(work / "killed") == lines, "a restart changed the split's manifest"
        assert files_of(work / "killed") == files_of(work / "census"), "a restart changed the split's files"
        print(f"killed after {killed_at} of {len(lines)} lines and run again: the same manifest and files")
        if against is not None:
            split(against, work / "against", dump_slice)
            mine, theirs = files_of(work / "census"), files_of(work / "against")
            manifest = next(k for k in mine if k.endswith(dump.SPLIT_MANIFEST.name))
            assert mine.keys() == theirs.keys(), sorted(mine.keys() ^ theirs.keys())
            differ = [k for k in mine if mine[k] != theirs[k] and k != manifest]
            assert not differ, f"files differ from {against}'s split: {differ}"
            bare = sorted(({k: v for k, v in e.items() if k != "shards"} for e in lines), key=lambda e: e["chunk"])
            assert bare == sorted(lines_of(work / "against"), key=lambda e: e["chunk"]), "the manifest differs"
            print(f"parity with {against}: {len(mine) - 1} files byte-identical, the split's manifest the same "
                  "less its census")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--against", type=Path, help="another checkout's src, for parity")
    ap.add_argument("files", nargs="*", type=Path, help="c0, r50, s30 and s90.parquet")
    args = ap.parse_args()
    check_rule()
    check_synthetic()
    if args.files:
        check_split({p.stem: p for p in args.files}, args.against and args.against.resolve())


if __name__ == "__main__":
    main()
