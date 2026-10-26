"""Per-entity records from byte windows of a dump's bz2 file (stdlib only), for the
closure curves (bytes per table and set, by id) and the lag census (how far behind its
shard's running maximum an id comes).

Each window is read once (its bytes counted), decoded from its first block (bz2probe:
the block offsets match bzip2recover's, and libbz2 checks each block's CRC), and every
whole line parsed. Nothing is written but the records.

    python3 -I windows.py 20261019 --boundaries layout-20261019/boundaries.json \
        --uniform 101 --uniform-bytes 327680 --long 9 --long-bytes 12582912 \
        --out layout-20261019/entities.csv

`--uniform N`: N windows at even offsets over the whole file (a byte-weighted sample of
the dump). `--long K`: K long windows, one in each shard at the fraction (k + 0.5) / 8 of
it, and one at a shard's start (the oldest, densest items), for the lag census.
"""

import argparse
import csv
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bz2probe as bp  # noqa: E402

# src/wikidata/scholarly.py SCHOLARLY_CLASSES (copied: that module imports polars)
SCHOLARLY = {
    "Q13442814", "Q871232", "Q2782326", "Q815382", "Q187685", "Q1348305", "Q1907875", "Q18918145",
    "Q45182324", "Q23927052", "Q1402850", "Q15781350", "Q1266946", "Q7316896", "Q580922", "Q5246046",
    "Q30749496", "Q111475835", "Q56478376", "Q798134", "Q114613919", "Q92998777", "Q58901591",
    "Q10885494", "Q132115645", "Q59387148", "Q1385450", "Q130709863", "Q54670950", "Q51282918",
    "Q51282711", "Q15706459", "Q111475860", "Q51283092", "Q58900768", "Q58897583", "Q58898636",
    "Q591041", "Q193842", "Q58632367", "Q3099732", "Q637866", "Q21481766",
}

FIELDS = ["window", "kind", "shard", "offset", "index", "type", "id", "num", "scholarly", "line_bytes",
          "labels_bytes", "descriptions_bytes", "aliases_bytes", "sitelinks_bytes", "claims_bytes",
          "labels_n", "descriptions_n", "aliases_n", "sitelinks_n", "statements_n"]


def sub_len(raw: bytes, e: dict, key: str) -> int:
    v = e.get(key)
    return len(json.dumps(v, separators=(",", ":"))) if v else 0


def record(line: bytes) -> dict | None:
    try:
        e = json.loads(line)
    except ValueError:
        return None
    claims = e.get("claims") or {}
    p31 = {s.get("mainsnak", {}).get("datavalue", {}).get("value", {}).get("id")
           for s in claims.get("P31", []) if isinstance(s, dict)} if isinstance(claims, dict) else set()
    eid = e["id"]
    return {
        "type": e.get("type"), "id": eid, "num": int(eid[1:]) if eid[1:].isdigit() else -1,
        "scholarly": int(bool(p31 & SCHOLARLY)), "line_bytes": len(line),
        "labels_bytes": sub_len(line, e, "labels"), "descriptions_bytes": sub_len(line, e, "descriptions"),
        "aliases_bytes": sub_len(line, e, "aliases"), "sitelinks_bytes": sub_len(line, e, "sitelinks"),
        "claims_bytes": sub_len(line, e, "claims"),
        "labels_n": len(e.get("labels") or {}), "descriptions_n": len(e.get("descriptions") or {}),
        "aliases_n": sum(len(v) for v in (e.get("aliases") or {}).values()),
        "sitelinks_n": len(e.get("sitelinks") or {}),
        "statements_n": sum(len(v) for v in claims.values()) if isinstance(claims, dict) else 0,
    }


def window(url: str, offset: int, nbytes: int) -> tuple[list[dict], int, int]:
    buf = bp.fetch(url, offset, nbytes)
    blocks = bp.bit_positions(buf, bp.BLOCK, limit=2)
    if not blocks:
        return [], len(buf), 0
    text = bp.decode_blocks(buf, blocks[0], want=1 << 62)
    del buf
    lines = text.split(b"\n")[1:-1]
    del text
    out = []
    for line in lines:
        line = line.rstrip(b",")
        if line.startswith(b"{"):
            r = record(line)
            if r:
                out.append(r)
    return out, nbytes, sum(len(x) for x in lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("release")
    ap.add_argument("--boundaries", required=True)
    ap.add_argument("--uniform", type=int, default=0)
    ap.add_argument("--uniform-bytes", type=int, default=320 << 10)
    ap.add_argument("--long", type=int, default=0)
    ap.add_argument("--long-bytes", type=int, default=12 << 20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    url = bp.URL.format(r=a.release)
    b = json.load(open(a.boundaries))
    total = b["bytes"]
    edges = [0] + [x["before"] for x in b["boundaries"]] + [total]
    plan = [("uniform", int(total * i / a.uniform), a.uniform_bytes) for i in range(a.uniform)]
    for k in range(a.long):
        s = k % 8
        frac = 0.0 if k == 8 else (s + 0.5) / 8
        off = edges[s] + int((edges[s + 1] - edges[s]) * frac) + (4096 if k == 8 else 0)
        plan.append(("long", off, a.long_bytes))
    log = []
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for i, (kind, off, n) in enumerate(plan):
            recs, read, text = window(url, off, n)
            shard = max(j for j, e in enumerate(edges[:-1]) if off >= e)
            for j, r in enumerate(recs):
                w.writerow({"window": i, "kind": kind, "shard": shard, "offset": off, "index": j, **r})
            log.append({"window": i, "kind": kind, "offset": off, "shard": shard, "bytes_read": read,
                        "text_bytes": text, "entities": len(recs)})
            print(log[-1], flush=True)
    with open(a.out.replace(".csv", "-windows.json"), "w") as f:
        json.dump({"release": a.release, "bytes_read": bp.READS[0], "windows": log}, f, indent=0)
    print("read", bp.READS[0])


if __name__ == "__main__":
    main()
