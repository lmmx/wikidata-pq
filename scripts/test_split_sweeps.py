"""The split's sweep reader (src/wikidata/sweeps.py, WIKIDATA_SPLIT_SWEEPS;
docs/journal/2026-10-26-progressive-finalise.md, PR 2), against brute force:

- **the rewrap:** every block of a bzip2-written stream (its blocks at every bit
  alignment) decoded alone and in runs by `lbzip2 -n1` from the file's bytes as they are,
  behind a dummy block; the stream CRC from the block CRCs; a wrong stream CRC and a
  flipped bit refused;
- **synthetic dumps** of 8 shards one after another, each a sweep over the ids sha1 puts
  in it, with lines longer than a block and a shard's first line straddling blocks, one
  bz2 stream written by lbzip2 and by bzip2:
  - the cuts are each shard's first line, exactly;
  - the lines come out as the merge rule gives them by brute force (the next from the
    sweep whose running maximum is least), every line exactly once;
  - in segments of 64 kB, each its own decoder, the same;
  - resumed from the sweeps' positions after any line, the same lines follow;
  - one sweep (WIKIDATA_SPLIT_SWEEPS=1) gives the dump's own order;
  - a corrupt byte, a wrong stream CRC and a dropped block halt it;

Given chunk files (bench/data's c0, r50, s30 and s90) it also splits dumps made of them
(bench/dumpslice.py) as the CLI runs it, in chunks of 500 entities with 3 workers:

- **flag off,** with --against SRC (another checkout's src, e.g. master's): every file
  byte-identical to that checkout's split, the manifest the same;
- **one sweep:** every file byte-identical to the split without the flag, the manifest
  the same less the sweeps' positions;
- **8 sweeps** of the slice's entities in shard order (dumpslice.sharded: given new ids
  round-robin over 8 shards, as #181 proposed, each file's rows in their order; and
  without new ids, the slice's own 4 shards): every entity in exactly one chunk, its row
  the one the split without the flag makes of it; the census's frontier rising with the
  ids read, against the byte order's;
- **a restart:** the sweeps' split killed (SIGKILL, its process group) partway and run
  again writes the same manifest and files (its chunks cut at 500 entities or 4 MB of
  lines, so that both rules cut some);
- with --published, the release's published files from both splits (both sets' runs and
  finalise's, as bench/sweeps.py runs them) are the same.

    .venv/bin/python scripts/test_split_sweeps.py [--against SRC] [--published] [c0.parquet r50.parquet ...]
"""

import argparse
import bz2
import hashlib
import heapq
import os
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import orjson

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "bench"))

from wikidata import sweeps  # noqa: E402
from wikidata_reshape import bz2_blocks  # noqa: E402

RELEASE = "20261012"


def check_rewrap(work: Path) -> None:
    rng = random.Random(1)
    text = b"".join(b'{"type":"item","id":"Q%d","x":"%s"},\n' % (
        i, bytes(rng.choice(b"abcdefghij ") for _ in range(rng.randrange(50, 3000)))) for i in range(24000))
    src = work / "rewrap.json"
    src.write_bytes(text)
    subprocess.run(["bzip2", "-9", "-k", "-f", str(src)], check=True)
    path = src.with_suffix(".json.bz2")
    data = path.read_bytes()
    f = sweeps._File(path)
    hits = [(b, c) for b, c in bz2_blocks(data) if b < f.eos]
    bounds = [b for b, _ in hits] + [f.eos]
    alone = []
    for (b, c), e in zip(hits, bounds[1:]):
        s, drop = sweeps.stream(f.level, data, 0, b, e, [c])
        alone.append(bz2.decompress(s)[drop:])
    assert b"".join(alone) == text, "the blocks decoded alone are not the text"
    c = 0
    for _, crc in hits:
        c = sweeps.fold(c, crc)
    assert c == f.crc, "the stream CRC from the block CRCs is not the stream's"
    residues = set()
    for i in range(len(hits) - 2):
        b, e = hits[i][0], hits[i + 2][0]
        s, drop = sweeps.stream(f.level, data, 0, b, e, [h[1] for h in hits[i:i + 2]])
        r = subprocess.run(["lbzip2", "-dc", "-n1"], input=s, capture_output=True)
        assert r.returncode == 0 and r.stdout[drop:] == b"".join(alone[i:i + 2]), r.stderr
        residues.add(b % 8)
    s, _ = sweeps.stream(f.level, data, 0, hits[0][0], hits[2][0], [hits[0][1], hits[1][1] ^ 1])
    assert subprocess.run(["lbzip2", "-dc", "-n1"], input=s, capture_output=True).returncode, "a wrong CRC passed"
    s = bytearray(sweeps.stream(f.level, data, 0, hits[0][0], hits[2][0], [hits[0][1], hits[1][1]])[0])
    s[len(s) // 2] ^= 0x10
    assert subprocess.run(["lbzip2", "-dc", "-n1"], input=bytes(s), capture_output=True).returncode, "a flip passed"
    f.close()
    assert len(residues) == 8, residues
    print(f"rewrap: {len(hits)} blocks of a bzip2 stream at all 8 bit alignments, alone (libbz2) and in runs "
          "(lbzip2 -n1) from the file's own bytes; the stream CRC from theirs; a wrong CRC and a flip refused")


def synthetic(seed: int, n: int = 40_000) -> tuple[list[list[bytes]], list[bytes]]:
    """8 shards' lines (each a sweep over the ids up to Q`n` sha1 puts in it, monotone to
    within 2, with a few lines of 1-2 MB), and the dump's text: `[`, the lines
    comma-separated, `]`."""
    rng = random.Random(seed)
    shards = []
    for s in range(8):
        qs = [x for x in range(1, n) if sweeps.shard_of(f"Q{x}", 8) == s]
        for j in range(0, len(qs) - 1, 5):
            if qs[j + 1] - qs[j] <= 2:
                qs[j], qs[j + 1] = qs[j + 1], qs[j]
        lines = []
        for x in qs:
            size = rng.choice((40, 200, 1000, 5000)) if rng.random() > 0.0008 else rng.randrange(1 << 20, 2 << 20)
            pad = bytes(rng.choice(b"abcdefghijklmnopqrstuvwxyz0123456789 ") for _ in range(min(size, 4000)))
            pad = (pad * (size // len(pad) + 1))[:size]
            lines.append(b'{"type":"item","id":"Q%d","labels":{"en":{"language":"en","value":"%s"}}}' % (x, pad))
        props = [p for p in range(1, 300) if sweeps.shard_of(f"P{p}", 8) == s]
        for p in props:
            lines.insert(rng.randrange(len(lines)), b'{"type":"property","datatype":"string","id":"P%d"}' % p)
        shards.append(lines)
    return shards, [b"[\n" + b",\n".join(line for sh in shards for line in sh) + b"\n]\n"]


def merged(shards: list[list[bytes]]) -> list[bytes]:
    """The merge rule by brute force: the next line from the sweep whose running maximum
    (of Q numbers) is least, the lowest-numbered of those equal."""
    pos, rmax, out = [0] * len(shards), [-1] * len(shards), []
    while True:
        live = [k for k in range(len(shards)) if pos[k] < len(shards[k])]
        if not live:
            return out
        k = min(live, key=lambda k: (rmax[k], k))
        line = shards[k][pos[k]]
        pos[k] += 1
        rmax[k] = max(rmax[k], sweeps.line_q(line))
        out.append(line)


def read_all(path: Path, factor: int, state: Path, done: dict | None = None, upto: int | None = None):
    reader = sweeps.Reader(path, factor, state, done or {})
    out, states = [], []
    gen = reader.lines()
    for line in gen:
        out.append(line)
        states.append(reader.state())
        if upto is not None and len(out) == upto:
            gen.close()
            break
    return out, states, reader


def check_synthetic(work: Path) -> None:
    shards, text = synthetic(0)
    want = merged(shards)
    serial = [line for sh in shards for line in sh]
    raw = work / "syn.json"
    raw.write_bytes(text[0])
    for tool in ("lbzip2", "bzip2"):
        path = work / f"syn-{tool}.json.bz2"
        with raw.open("rb") as i, path.open("wb") as o:
            subprocess.run([tool, "-9", "-c"], stdin=i, stdout=o, check=True)
        state = work / f"state-{tool}"
        state.mkdir()
        lines, states, reader = read_all(path, 8, state)
        plan = reader.plan
        firsts = [sh[0] for sh in shards]
        assert [c["id"] for c in plan["sweeps"]] == [sweeps.line_id(f) for f in firsts], plan["sweeps"]
        assert [c["shard"] for c in plan["sweeps"]] == list(range(8))
        assert lines == want, f"{tool}: the lines are not the merge rule's"
        assert sorted(lines) == sorted(serial)
        # segments of 64 kB: the same lines, from many decoders a sweep
        keep = sweeps.SEGMENT_BYTES
        sweeps.SEGMENT_BYTES = 64 << 10
        try:
            again, states, _ = read_all(path, 8, state)
            assert again == want, f"{tool}: in segments, the lines differ"
            for i in sorted(random.Random(2).sample(range(len(want)), 12)) + [len(want) - 1]:
                done = {0: {"chunk": 0, "rows": i + 1, "sweeps": orjson.loads(orjson.dumps(states[i]))}}
                rest, _, _ = read_all(path, 8, state, done)
                assert rest == want[i + 1:], f"{tool}: resumed after line {i}, the lines differ"
        finally:
            sweeps.SEGMENT_BYTES = keep
        # one sweep: the dump's own order
        one = work / f"state1-{tool}"
        one.mkdir()
        assert read_all(path, 1, one)[0] == serial, f"{tool}: one sweep is not the dump's order"
        print(f"synthetic dump ({tool}, {path.stat().st_size / 1e6:.1f} MB, {len(want):,} lines): the 8 cuts at each "
              f"shard's first line ({plan['probes']} probes, {plan['seconds']} s); the merge rule's lines, each once; "
              "the same in 64 kB segments and resumed after any of 13 lines; one sweep the dump's order")
    # tiny shards: shard 3 of 3 lines, shard 5 of 40 (a few blocks' worth), shard 6 empty:
    # each line once, each sweep starting at its shard's first line
    tiny = [sh[:3] if s == 3 else sh[:40] if s == 5 else [] if s == 6 else sh for s, sh in enumerate(shards)]
    path = work / "tiny.json.bz2"
    raw.write_bytes(b"[\n" + b",\n".join(line for sh in tiny for line in sh) + b"\n]\n")
    with raw.open("rb") as i, path.open("wb") as o:
        subprocess.run(["lbzip2", "-9", "-c"], stdin=i, stdout=o, check=True)
    state = work / "state-tiny"
    state.mkdir()
    lines, _, reader = read_all(path, 8, state)
    assert sorted(lines) == sorted(line for sh in tiny for line in sh), "tiny shards: a line lost or repeated"
    firsts = {sweeps.line_id(sh[0]): s for s, sh in enumerate(tiny) if sh}
    cuts = reader.plan["sweeps"]
    assert all(firsts.get(c["id"]) == c["shard"] for c in cuts), cuts
    if len(cuts) == 7:
        assert lines == merged([sh for sh in tiny if sh]), "tiny shards: not the merge rule's lines"
    print(f"tiny shards (3 lines, 40 lines, none): {len(cuts)} sweeps (shards {[c['shard'] for c in cuts]}), each at "
          "its shard's first line; every line once")
    # corruption: a flipped byte in a block, a wrong stream CRC, a block dropped
    path = work / "syn-lbzip2.json.bz2"
    data = path.read_bytes()
    hits = [b for b, _ in bz2_blocks(data)]
    flip = len(data) * 3 // 10  # (inside a shard, not in a block the cuts decode)
    for name, broken in (("a flipped byte", data[:flip] + bytes([data[flip] ^ 0x40]) + data[flip + 1:]),
                         ("a wrong stream CRC", data[:-2] + bytes([data[-2] ^ 1]) + data[-1:]),
                         ("a block dropped", data[:hits[5] // 8] + data[hits[6] // 8:])):
        bad = work / "bad.json.bz2"
        bad.write_bytes(broken)
        state = work / f"state-bad-{len(name)}"
        state.mkdir()
        if name == "a flipped byte":  # (the good dump's cuts, so that the decoders meet it, not the probes)
            (state / sweeps.PLAN).write_bytes((work / "state-lbzip2" / sweeps.PLAN).read_bytes())
        try:
            read_all(bad, 8, state)
        except (RuntimeError, SystemExit) as e:
            print(f"  {name}: halted ({str(e).replace(str(work) + chr(47), '')[:150]})")
        else:
            raise AssertionError(f"{name} was not caught")
    print("corruption: a flipped byte, a wrong stream CRC and a dropped block each halt the sweeps")


SPLIT = """
import sys
sys.path.insert(0, {src!r})
from wikidata import dump
try:
    from wikidata import sweeps
    sweeps.SEGMENT_BYTES = {segment}
except ImportError:
    pass  # (a checkout without the sweep reader)
dump.CHUNK_ENTITIES = 500
dump.SWEEPS_CHUNK_BYTES = 4 << 20  # (with sweeps, chunks of the slice's head cut by bytes too; unused by a checkout without)
dump.split(workers=3)
"""


def run_split(src: Path, d: Path, dump_file: Path, factor: int | None = None, kill_after: int | None = None,
              segment: int = 256 << 10) -> str | None:
    """The CLI's split of `dump_file` in `d` (in byte order, or as `factor` sweeps of
    `segment` bytes), in a session of its own; with `kill_after`, killed (SIGKILL, its
    whole process group: the split, its threads and its decoders) once its manifest has
    that many lines."""
    path = d / "releases" / RELEASE / "dump" / f"wikidata-{RELEASE}-all.json.bz2"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        os.link(dump_file, path)
    env = {k: v for k, v in os.environ.items() if k != "WIKIDATA_SPLIT_SWEEPS"}
    env |= {"WIKIDATA_RELEASE": RELEASE, "HF_HUB_OFFLINE": "1"} | ({"WIKIDATA_SPLIT_SWEEPS": str(factor)} if factor else {})
    proc = subprocess.Popen([sys.executable, "-c", SPLIT.format(src=str(src), segment=segment)], cwd=d, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
    if kill_after is None:
        out = proc.communicate()[0].decode()
        assert proc.returncode == 0, out[-3000:]
        return out
    manifest = d / "releases" / RELEASE / "data" / "manifest.split.jsonl"
    while proc.poll() is None:
        if manifest.exists() and manifest.read_bytes().count(b"\n") >= kill_after:
            os.killpg(proc.pid, signal.SIGKILL)
            break
        time.sleep(0.02)
    proc.wait()
    assert proc.returncode != 0, "the split ended before it was killed"
    for _ in range(100):  # (none of its group is left: its decoders went with it)
        try:
            os.killpg(proc.pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        raise AssertionError("a process of the killed split's group is still running")
    return None


def files_of(d: Path) -> dict[str, str]:
    return {str(p.relative_to(d)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((d / "releases").rglob("*")) if p.is_file() and "dump" not in p.relative_to(d).parts}


def lines_of(d: Path) -> list[dict]:
    path = d / "releases" / RELEASE / "data" / "manifest.split.jsonl"
    return [orjson.loads(line) for line in path.read_bytes().splitlines() if line]


def ids_of(d: Path) -> dict[str, list[str]]:
    import polars as pl

    out = {}
    for name, data in (("main", d / "releases" / RELEASE / "data"), ("scholar", d / "releases" / f"{RELEASE}-scholar" / "data")):
        out[name] = sorted(i for f in sorted(data.glob("chunk_*.parquet")) for i in pl.read_parquet(f, columns=["id"])["id"])
    return out


def census_of(d: Path) -> dict:
    from wikidata import dump

    return dump.split_census(lines_of(d))


def slice_of(files: dict[str, Path], work: Path) -> Path:
    """A dump slice of the chunk files' first rows (dumpslice.make)."""
    import polars as pl
    from dumpslice import make

    pieces = []
    for name, n in (("c0", 1500), ("r50", 600), ("s30", 600), ("s90", 600)):
        pieces.append(work / f"{name}.parquet")
        pl.read_parquet(files[name])[:n].write_parquet(pieces[-1])
    out = work / "slice.json.bz2"
    make(out, [str(p) for p in pieces])
    return out


def check_split(files: dict[str, Path], against: Path | None, published: bool, work: Path) -> None:
    from dumpslice import sharded

    src = ROOT / "src"
    dump_slice = slice_of(files, work)
    # flag off, and one sweep: the dump's own order
    run_split(src, work / "serial", dump_slice)
    serial = files_of(work / "serial")
    if against is not None:
        run_split(against, work / "against", dump_slice)
        theirs = files_of(work / "against")
        assert serial == theirs, sorted(k for k in serial.keys() | theirs.keys() if serial.get(k) != theirs.get(k))
        print(f"flag off: all {len(serial)} files the split writes byte-identical to {against}'s, its manifest too")
    run_split(src, work / "one", dump_slice, factor=1)
    one = files_of(work / "one")
    manifest = f"releases/{RELEASE}/data/manifest.split.jsonl"
    plan = f"releases/{RELEASE}/data/sweeps.json"
    differ = sorted(k for k in serial.keys() | one.keys() if serial.get(k) != one.get(k) and k not in (manifest, plan))
    assert not differ, f"one sweep: files differ from the split's in byte order: {differ}"
    assert [{k: v for k, v in e.items() if k != "sweeps"} for e in lines_of(work / "one")] == lines_of(work / "serial")
    print(f"one sweep: all {len(serial) - 1} chunk files byte-identical to the split's in byte order, the manifest "
          "the same less the sweeps' positions")
    # 8 sweeps, of the slice's entities given new ids over 8 shards; and of its own 4 shards
    for renumber in (True, False):
        name = "renumbered" if renumber else "own"
        dump8 = work / f"sharded-{name}.json.bz2"
        counts = sharded(dump_slice, dump8, renumber=renumber)["lines"]
        a, b = work / f"{name}-serial", work / f"{name}-sweeps"
        run_split(src, a, dump8)
        out = run_split(src, b, dump8, factor=8)
        cuts = orjson.loads((b / plan).read_bytes())["sweeps"]
        shards = [c["shard"] for c in cuts]
        assert shards == [s for s in range(8) if counts[s]], (shards, counts)
        ids_a, ids_b = ids_of(a), ids_of(b)
        for k in ids_a:
            assert len(set(ids_b[k])) == len(ids_b[k]), f"{name}: an entity in two parts"
            assert ids_a[k] == ids_b[k], f"{name}: the {k} set's entities differ"
        ca, cb = census_of(a), census_of(b)
        fa, fb = ca["frontier"], cb["frontier"]
        line = next(x for x in out.splitlines() if x.startswith("Read as"))
        sizes = [e["rows"] for e in lines_of(b)]
        assert max(sizes) == 500 and min(sizes[:-1]) < 500, sizes  # (both rules cut chunks)
        print(f"8 sweeps, {name} ids ({sum(counts):,} entities, shards {shards}): every entity in one part, each set's "
              f"the same as in byte order; the frontier at most {fb['lag']:,} below the largest id read, rising after "
              f"{fb['rises']} of {cb['chunks']} chunks (in byte order {fa['lag']:,}, after {fa['rises']}); the census: "
              f"{cb['all']['lag']:,} the most a Q id came below its shard's running maximum, {cb['all']['late']} more "
              f"than 10,000 below")
        print(f"  {line}")
        if renumber:
            assert fb["lag"] < fa["lag"] // 10 and fb["rises"] > fa["rises"], (fa, fb)
            # killed partway (twice) and run again: the same manifest and files
            whole = files_of(b)
            for kills in ((2,), (1, 4)):
                k = work / f"killed-{'-'.join(map(str, kills))}"
                for n in kills:
                    run_split(src, k, dump8, factor=8, kill_after=n)
                at = len(lines_of(k))
                run_split(src, k, dump8, factor=8)
                assert lines_of(k) == lines_of(b), "a restart changed the sweeps' manifest"
                assert {**files_of(k), plan: None} == {**whole, plan: None}, "a restart changed the sweeps' files"
                cut = lambda d: {x: v for x, v in orjson.loads((d / plan).read_bytes()).items() if x != "seconds"}  # noqa: E731
                assert cut(k) == cut(b), "a restart changed the sweeps' cuts"
                print(f"  killed (SIGKILL, its process group) after {', then '.join(map(str, kills))} lines "
                      f"({at} written) and run again: the same {len(lines_of(b))} lines and {len(whole)} files")
            if published:
                import sweeps as stage

                os.environ["WIKIDATA_RELEASE"] = RELEASE
                with open(work / "published.log", "w") as log:
                    pa, pb = stage.published(src, a, 3, log), stage.published(src, b, 3, log)
                mb = sum(sum(t.values()) for t in pb["published"].values()) / 1e6
                assert pa["digest"] == pb["digest"], "the published files differ"
                assert pa["published"] == pb["published"]
                print(f"  published (both sets' runs, then finalise's three, through the stand-in Hub): "
                      f"{mb:.1f} MB, byte for byte the same from both splits")
        for d in (a, b):
            shutil.rmtree(d)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--against", type=Path, help="another checkout's src, for the flag-off parity")
    ap.add_argument("--published", action="store_true", help="the published files from both splits")
    ap.add_argument("files", nargs="*", type=Path, help="c0, r50, s30 and s90.parquet")
    args = ap.parse_args()
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as work:
        work = Path(work)
        check_rewrap(work)
        check_synthetic(work)
        if args.files:
            check_split({p.stem: p for p in args.files}, args.against and args.against.resolve(), args.published,
                        work)


if __name__ == "__main__":
    main()
