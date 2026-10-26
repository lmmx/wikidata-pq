"""Every entity of churn.py's newest-dump windows, with its set and its fields' bytes
(stdlib only): the same windows re-read (the same offsets and lengths, from churn.py's
windows.jsonl), so each compared entity gets the set it is published in and the bytes it
has in each table, for the churn by set and the dedup model (dedup.py).

    python3 -I bside.py --newest 20261019 --windows raw/windows.jsonl --bytes 1310720 \\
        --out raw/bside.csv.gz
"""

import argparse
import csv
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bz2probe as bp  # noqa: E402
import churn  # noqa: E402

COLS = ["point", "shard", "num", "scholarly", "line_bytes", "labels_bytes", "descriptions_bytes", "aliases_bytes",
        "sitelinks_bytes", "claims_bytes", "labels_n", "descriptions_n", "aliases_n", "sitelinks_n", "statements_n"]


def size(v) -> int:
    return len(json.dumps(v, separators=(",", ":"), ensure_ascii=False).encode()) if v else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--newest", required=True)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--bytes", type=int, default=1280 << 10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    url = bp.URL.format(r=a.newest)
    with gzip.open(a.out, "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for rec in map(json.loads, open(a.windows)):
            s = rec["shard"]
            for n, line in churn.shard_q(churn.lines_at(url, rec["offset_b"], a.bytes), s):
                if not rec["lo"] <= n <= rec["hi"]:
                    continue
                e = json.loads(line)
                c = e.get("claims") or {}
                w.writerow({
                    "point": rec["point"], "shard": s, "num": n,
                    "scholarly": int(bool(set(churn.p31(e)) & churn.SCHOLARLY)), "line_bytes": len(line),
                    **{f"{k}_bytes": size(e.get(k)) for k in churn.FIELDS},
                    "labels_n": len(e.get("labels") or {}), "descriptions_n": len(e.get("descriptions") or {}),
                    "aliases_n": sum(len(v) for v in (e.get("aliases") or {}).values()),
                    "sitelinks_n": len(e.get("sitelinks") or {}),
                    "statements_n": sum(len(v) for v in c.values()) if isinstance(c, dict) else 0,
                })
            print(rec["point"], bp.READS[0], flush=True)
    print("read", bp.READS[0])


if __name__ == "__main__":
    main()
