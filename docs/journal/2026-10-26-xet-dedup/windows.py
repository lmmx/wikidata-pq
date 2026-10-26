"""The bench slice's entities in other weekly dumps, by byte-range reads of
dumps.wikimedia.org (stdlib only, but pyarrow for the chunk files' ids).

The quick tier's 4 chunk files (bench-data: c0, r50, s30, s90) are each a 10,000-line
chunk of one shard of the 20261012 dump (or the set's share of one), so their entities
are one contiguous stretch of that shard: shard 0's head (c0, Q8-Q83751), shard 3 at
Q140.07M-Q140.16M (r50), shard 2 at Q42.78M-Q42.86M (s30), shard 7 at Q28.24M-Q28.32M
(s90). The shards are in page-id order and monotone in id to within 2 past their first
~1,000 entities (2026-10-26-progressive-finalise.md, Gate 2). For each file and each dump
asked for, the stretch is found by a bisection of small probes (192 KiB each) between
two points of #180's layout of 20261019 (offsets.csv, scaled by the dump's size), and
read whole, its blocks decoded by bz2probe (#180's: each block rewrapped and decoded by
libbz2, which checks its CRC). Each of the chunk's ids found is kept, its line as the
dump has it; an id not found in the stretch is gone (deleted or redirected).

Also, with --tail, the first N entities created after 20261012 at shard 7's end (from
#188's tails.json: shard 7's largest id in 20261012 and where 20261019's tail starts).

Writes `<out>/<dump>.<file>.lines.gz` (id TAB line, the chunk file's order) to the
scratch directory given, and the reads' record to --log (merged into it if it exists).
Holds each read in memory.

    python3 -I windows.py --dumps 20261019 20261005 --files c0.parquet r50.parquet s30.parquet \\
        s90.parquet --layout ../2026-10-26-progressive-finalise/layout-20261019 --out SCRATCH \\
        --tail 20 --log results/windows.json
"""

import argparse
import bz2
import csv
import gzip
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-10-26-progressive-finalise"))
import bz2probe as bp  # noqa: E402


def shard_of(eid: str, n: int = 8) -> int:
    """The dumps' shard of an id (#188's churn.py)."""
    return int(hashlib.sha1(eid.encode()).hexdigest()[:8], 16) % n


def size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": bp.UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])


def decode_from(url: str, offset: int, nbytes: int) -> list[bytes]:
    """The whole lines decoded from the first block at or after `offset` in `nbytes`
    (from offset 0: the stream itself)."""
    buf = bp.fetch(url, offset, nbytes)
    if offset == 0:
        d = bz2.BZ2Decompressor()
        out = []
        for i in range(0, len(buf), 1 << 20):
            try:
                out.append(d.decompress(buf[i:i + (1 << 20)]))
            except (OSError, EOFError):
                break
        text = b"".join(out)
        lines = text.split(b"\n")[1:-1]  # ("[" first, a cut line last)
    else:
        blocks = bp.bit_positions(buf, bp.BLOCK, limit=2)
        if not blocks:
            return []
        text = bp.decode_blocks(buf, blocks[0], want=1 << 62)
        lines = text.split(b"\n")[1:-1]
    return [x.rstrip(b",") for x in lines if x.startswith(b"{")]


def head_id(line: bytes) -> str | None:
    m = bp.ID.match(line)
    return (m.group(2) + m.group(3)).decode() if m else None


def shard_ids(lines: list[bytes], s: int) -> list[tuple[int, str, bytes]]:
    out = []
    for line in lines:
        e = head_id(line)
        if e and shard_of(e) == s:
            out.append((int(e[1:]), e, line))
    return out


def bracket(layout: Path, s: int, target: int) -> tuple[int, int]:
    """Offsets in 20261019 around shard `s`'s id `target`, from #180's offsets.csv: the
    last point of the shard at or below it, the next above (or the shard's ends)."""
    edges = [0] + [b["before"] for b in json.loads((layout / "boundaries.json").read_text())["boundaries"]]
    edges.append(json.loads((layout / "boundaries.json").read_text())["bytes"])
    rows = [r for r in csv.DictReader(open(layout / "offsets.csv")) if r["q_min"]]
    pts = sorted((int(r["offset"]), int(r["q_min"])) for r in rows
                 if edges[s] <= int(r["offset"]) < edges[s + 1] and int(r["offset"]) > edges[s] + (1 << 20))
    lo = max([o for o, q in pts if q <= target], default=edges[s])
    hi = min([o for o, q in pts if q > target and o > lo], default=edges[s + 1] - (1 << 20))
    return lo, hi


def first_id(url: str, s: int, p: int, log: list) -> int | None:
    q = [n for n, e, _ in shard_ids(decode_from(url, p, 192 << 10), s) if e[0] == "Q"]
    log.append(p)
    return min(q[:20]) if q else None


def locate(url: str, s: int, target: int, lo: int, hi: int, log: list) -> int:
    """An offset from which shard `s`'s ids start at or a little below `target`: a
    bisection between offsets whose ids are below and above it (the bracket widened
    first if the scaled guess missed)."""
    for _ in range(8):
        m_lo, m_hi = first_id(url, s, lo, log), first_id(url, s, hi, log)
        if m_lo is not None and m_lo <= target and (m_hi is None or m_hi > target):
            break
        w = hi - lo
        if m_lo is None or m_lo > target:
            lo = max(0, lo - w)
        if m_hi is not None and m_hi <= target:
            hi = hi + w
    for _ in range(40):
        if hi - lo < (256 << 10):
            return lo
        mid = (lo + hi) // 2
        m = first_id(url, s, mid, log)
        if m is None or m > target:
            hi = mid
        else:
            lo = mid
            if target - m < 3000:
                return lo
    raise SystemExit(f"could not locate Q{target} in shard {s} of {url}")


def window(url: str, s: int, nums: list[int], start: int | None, br: tuple[int, int]) -> tuple[dict, dict]:
    """The chunk's stretch of shard `s` in `url`: its lines by id, and the reads."""
    lo, hi = nums[len(nums) // 200], nums[-1]  # (lo past a straggler or two below the stretch)
    probes: list[int] = []
    t0 = bp.READS[0]
    if start is None:
        p_lo = locate(url, s, lo, br[0], br[1], probes)
        p_hi = locate(url, s, hi, p_lo, br[1] + (64 << 20), probes)
    else:
        p_lo, p_hi = start, start
    found: dict[str, bytes] = {}
    p, end = p_lo, p_hi + (2 << 20)
    while True:
        lines = decode_from(url, p, end - p)
        q = shard_ids(lines, s)
        for _, e, line in q:
            found.setdefault(e, line)
        top = max((n for n, e, _ in q[-50:] if e[0] == "Q"), default=0)
        if top > hi + 2000 or not lines:
            break
        p, end = end - (1 << 20), end + (8 << 20)  # (more, overlapping by a block or so)
    rec = {"shard": s, "lo": lo, "hi": hi, "offset": p_lo, "probes": len(probes), "bytes": bp.READS[0] - t0,
           "lines_in_shard": len(found)}
    return found, rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", nargs="*", default=[])
    ap.add_argument("--files", nargs="*", default=[])
    ap.add_argument("--layout", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--tail", type=int, default=0)
    ap.add_argument("--tails-json", type=Path)
    ap.add_argument("--log", required=True, type=Path)
    a = ap.parse_args()
    import pyarrow.parquet as pq

    chunks = []
    for f in a.files:
        ids = pq.read_table(f, columns=["id"]).column("id").to_pylist()
        s = shard_of(ids[0])
        chunks.append((Path(f).stem, ids, s, sorted(int(i[1:]) for i in ids if shard_of(i) == s and i[0] == "Q")))
    a.out.mkdir(parents=True, exist_ok=True)
    log = json.loads(a.log.read_text()) if a.log.exists() else {"dumps": {}, "files": {}, "bytes_read": 0}
    log["files"] |= {c[0]: {"entities": len(c[1]), "shard": c[2]} for c in chunks}
    size19 = None
    for d in a.dumps:
        url = bp.URL.format(r=d)
        total = size(url)
        size19 = size19 or (total if d == "20261019" else size(bp.URL.format(r="20261019")))
        recs = {}
        for name, ids, s, nums in chunks:
            with gzip.open(a.out / f"{d}.{name}.lines.gz", "wt") as w:
                t = time.time()
                start = 0 if name == "c0" else None
                br = tuple(int(x * total / size19) for x in bracket(a.layout, s, nums[len(nums) // 200]))
                found, rec = window(url, s, nums, start, br)
                kept = 0
                for e in ids:
                    if e in found:
                        w.write(e + "\t" + found[e].decode() + "\n")
                        kept += 1
                rec.update(found=kept, gone=len(ids) - kept, seconds=round(time.time() - t, 1))
                recs[name] = rec
                print(d, name, rec, flush=True)
        log["dumps"].setdefault(d, {"bytes": total, "files": {}})["files"] |= recs
    if a.tail:
        t = json.loads(a.tails_json.read_text())
        pair = next(x for x in t if x["newer"] == "20261019")
        url = bp.URL.format(r="20261019")
        lines = decode_from(url, pair["locate_offset"], 512 << 10)
        new = [(e, line) for n, e, line in shard_ids(lines, 7) if e[0] == "Q" and n > pair["older_max_shard7"]][:a.tail]
        with gzip.open(a.out / "new.lines.gz", "wt") as w:
            for e, line in new:
                w.write(e + "\t" + line.decode() + "\n")
        log["tail"] = {"offset": pair["locate_offset"], "above": pair["older_max_shard7"], "kept": len(new),
                       "ids": [e for e, _ in new]}
        print("tail", log["tail"], flush=True)
    log["bytes_read"] += bp.READS[0]
    log.setdefault("runs", []).append({"files": [c[0] for c in chunks], "dumps": a.dumps, "bytes": bp.READS[0]})
    a.log.parent.mkdir(parents=True, exist_ok=True)
    a.log.write_text(json.dumps(log, indent=1))
    print("read", bp.READS[0] / 1e6, "MB")


if __name__ == "__main__":
    main()
