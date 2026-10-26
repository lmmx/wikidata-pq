"""The entities created between two dumps (stdlib only), from the end of the last shard.

Each shard is in page-id order, so the entities created since the older dump are at the
end of every shard of the newer one. Shard 7 is the file's last, so its end is the file's.
For each (older, newer) pair: the older dump's last window gives its largest id in shard
7 (mA); in the newer dump, mA is located by small probes (churn.locate), and the bytes
from there to the end are shard 7's new entities. Those times 8 (the shards are a hash
of the id) are the new entities' bz2 bytes; a window read there gives their density
(entities a bz2 byte, and the share of new ids that are live).

    python3 -I tails.py --pairs 20261005:20261012 20261012:20261019 --out raw/tails.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bz2probe as bp  # noqa: E402
import churn  # noqa: E402

WIN = 1 << 20


def tail(url: str, total: int) -> list[int]:
    return [n for n, _ in churn.shard_q(churn.lines_at(url, total - WIN, WIN), 7)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = []
    for pr in a.pairs:
        ra, rb = pr.split(":")
        ua, ub = bp.URL.format(r=ra), bp.URL.format(r=rb)
        ta, tb = churn.size(ua), churn.size(ub)
        r0 = bp.READS[0]
        qa = tail(ua, ta)
        qb = tail(ub, tb)
        ma, mb = max(qa), max(qb)
        dens_b = WIN / max(1, max(qb) - min(qb))  # bz2 bytes a unit of id, at B's end
        probes: list[int] = []
        pos, first = churn.locate(ub, 7, ma, tb - int((mb - ma) * dens_b), dens_b, tb, probes)
        # the window from there: how many of B's shard-7 entities above mA a bz2 byte
        win = [n for n, _ in churn.shard_q(churn.lines_at(ub, pos, WIN), 7)]
        above = [n for n in win if n > ma]
        span = (max(win) - min(win)) if win else 0
        rec = {
            "older": ra, "newer": rb, "older_bytes": ta, "newer_bytes": tb,
            "older_max_shard7": ma, "newer_max_shard7": mb, "older_tail_n": len(qa), "newer_tail_n": len(qb),
            "locate_offset": pos, "locate_first": first, "probes": len(probes),
            "shard7_new_bz2_bytes": tb - pos, "new_bz2_bytes_all_shards": 8 * (tb - pos),
            "window_n": len(win), "window_above_ma": len(above), "window_span_ids": span,
            # live share of ids in the window's range (a shard holds 1 id in 8)
            "live_share_window": round(8 * len(win) / span, 4) if span else None,
            "bz2_bytes_per_entity_window": round(WIN / len(win), 1) if win else None,
            "new_entities_est": round(8 * (tb - pos) / (WIN / len(win))) if win else None,
            "new_ids_range": mb - ma,
            "bytes_read": bp.READS[0] - r0,
        }
        out.append(rec)
        print(rec, flush=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("read", bp.READS[0])


if __name__ == "__main__":
    main()
