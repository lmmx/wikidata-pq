"""Survey a dump's bz2 file: ids against byte offset at N even offsets, then the shard
boundaries (where the ids fall back to low Q numbers) found by bisection to within
`--tol` bytes, and what the file holds at each boundary (a new bz2 stream or not).

    python3 -I survey.py 20261019 --n 100 --out layout-20261019

Writes {out}/offsets.csv (one row a probe: offset, first block's bit offset, entities,
first, last, min, max id number, types seen, inversions within the probe), and
{out}/ids.csv (every id decoded, with its probe's offset and its index there), and
{out}/boundaries.json.
"""

import argparse
import csv
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bz2probe as bp  # noqa: E402


def size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": bp.UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])


def summary(p: dict) -> dict:
    ids = p["ids"]
    q = [n for t, letter, n in ids if letter == "Q"]
    inv = sum(1 for a, b in zip(q, q[1:]) if b < a)
    lag = 0
    run = 0
    for n in q:
        run = max(run, n)
        lag = max(lag, run - n)
    return {
        "offset": p["offset"], "block_bit": p.get("block_bit"), "entities": len(ids),
        "first": f"{ids[0][1]}{ids[0][2]}" if ids else "", "last": f"{ids[-1][1]}{ids[-1][2]}" if ids else "",
        "q_min": min(q) if q else "", "q_max": max(q) if q else "",
        "types": "/".join(sorted({t for t, _, _ in ids})),
        "non_q": ";".join(f"{letter}{n}" for t, letter, n in ids if letter != "Q"),
        "q_descents": inv, "q_max_lag": lag, "streams": ";".join(map(str, p["streams"])),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("release")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--tol", type=int, default=4 << 20)
    ap.add_argument("--read", type=int, default=320 << 10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    url = bp.URL.format(r=a.release)
    total = size(url)
    os.makedirs(a.out, exist_ok=True)
    cache: dict[int, dict] = {}

    def at(off: int) -> dict:
        if off not in cache:
            cache[off] = bp.probe(url, off, read=a.read, want=1_800_000)
        return cache[off]

    offsets = [int(total * i / a.n) for i in range(a.n)] + [max(0, total - a.read)]
    for off in offsets:
        p = at(off)
        s = summary(p)
        print(s, flush=True)

    # a boundary: between consecutive probes whose ids fall from high to low
    def lo_q(p):
        q = [n for _, letter, n in p["ids"] if letter == "Q"]
        return (min(q), max(q)) if q else None

    bounds = []
    srt = sorted(cache)
    for x, y in zip(srt, srt[1:]):
        a_, b_ = lo_q(cache[x]), lo_q(cache[y])
        if a_ and b_ and b_[0] < a_[1] / 4:  # ids fell back to a quarter or less
            lo, hi = x, y
            while hi - lo > a.tol:
                mid = (lo + hi) // 2
                m = lo_q(at(mid))
                if m is None:
                    break
                if m[0] < a_[1] / 4 and m[1] < a_[1] / 4:
                    hi = mid
                else:
                    lo = mid
                    a_ = m
            # the boundary's neighbourhood: one read across it, for streams and marks
            near = bp.fetch(url, lo, min(hi - lo + a.read, 8 << 20))
            streams = [lo + m.start() for m in bp.STREAM.finditer(near)]
            eos = [lo * 8 + b for b in bp.bit_positions(near, bp.EOS, limit=4)]
            bounds.append({"after": lo, "before": hi, "last_q_before": cache[lo]["ids"][-1] if cache[lo]["ids"] else None,
                           "first_q_after": cache[hi]["ids"][0] if cache[hi]["ids"] else None,
                           "streams_near": streams, "eos_bits_near": eos})
            print("boundary", bounds[-1], flush=True)

    rows = [summary(cache[o]) for o in sorted(cache)]
    with open(f"{a.out}/offsets.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(f"{a.out}/ids.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["offset", "index", "type", "id"])
        for o in sorted(cache):
            for i, (t, letter, n) in enumerate(cache[o]["ids"]):
                w.writerow([o, i, t, f"{letter}{n}"])
    with open(f"{a.out}/boundaries.json", "w") as f:
        json.dump({"release": a.release, "bytes": total, "probes": len(cache), "bytes_read": bp.READS[0],
                   "boundaries": bounds}, f, indent=1)
    print("read", bp.READS[0], "bytes in", len(cache), "probes")


if __name__ == "__main__":
    main()
