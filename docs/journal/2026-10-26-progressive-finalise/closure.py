"""The frontier of a synchronised split and the share of each published table that is
final as it rises (stdlib only).

Inputs (this folder): layout-20261019/offsets.csv and boundaries.json (survey.py), and
layout-20261019/sample.csv.gz (windows.py's uniform windows: 101 windows of 320 KB at even
offsets, a byte-weighted sample of the dump's entities). Every figure is modelled from
that sample; see the journal entry for the assumptions.

- **The frontier.** Each shard's running maximum id against the fraction of the shard
  read, q_s(f), interpolated between probes. A synchronised split reads every shard to the
  same fraction f, so every id up to F(f) = min_s q_s(f) - margin has been read (the lag
  census: no id comes more than 2 behind its shard's running maximum, past the first
  ~1,000 entities of a shard, which hold ids below Q10k).
- **A part's closure.** A part is a string range [a, b) of a key's ids. It is final once
  the largest number whose string falls in it (at most N_MAX) is under the frontier
  (`max_number`). A part holding any of the last ids (Q141.7M+) closes at the end.
- **Parts.** Each key's rows are cut into n parts of equal bytes in string order (the
  sample's weighted quantiles); n is the key's bytes over a target part size: 500 MiB
  (today's COMPACT_FILE_BYTES) or a smaller regional target (design B, once).
- **Bytes.** Each table's bytes by entity are its JSON field's bytes (claims, labels,
  descriptions, aliases, sitelinks; one unit an entity for `entities`), by set
  (scholarly: src/wikidata/scholarly.py's classes in P31), scaled to the release's
  published bytes (bench 17.4.0's finalise_link_bytes, as the L2 model scales them).
  The keys of the language and site tables take the per-key byte shares of
  docs/dataset_cards_metadata.json (a main-set build), for both sets, each key's bytes
  spread over ids as its table's are.

    python3 -I closure.py [--margin 10000] [--json closure.json]
"""

import argparse
import bisect
import csv
import gzip
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
LAYOUT = os.path.join(HERE, "layout-20261019")
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

N_MAX = 141_800_000  # the largest Q id in the dump (the survey's largest: Q141,794,888)
MIB = 1024**2
# bench 17.4.0 (296adf4) finalise_link_bytes, scaled by RELEASE_SOURCE_BYTES / sample (GB)
PUBLISHED_GB = {
    "main": {"claims": 41.43, "labels": 6.60, "descriptions": 3.82, "aliases": 2.34, "links": 5.64,
             "entities": 0.25, "claims_labels": 2.63},
    "scholar": {"claims": 52.79, "labels": 3.50, "descriptions": 7.54, "aliases": 0.07, "links": 0.03,
                "entities": 0.53, "claims_labels": 1.74},
}
FIELD = {"claims": "claims_bytes", "labels": "labels_bytes", "descriptions": "descriptions_bytes",
         "aliases": "aliases_bytes", "links": "sitelinks_bytes", "entities": None}
KEYED = ("labels", "descriptions", "aliases", "links")
TARGETS = {"today (500 MiB parts)": (500 * MIB, 1), "regional, 128 MiB": (128 * MIB, 1), "regional, 32 MiB": (32 * MIB, 1),
           "regional, 128 MiB, every key in at least 4 files": (128 * MIB, 4)}
GRID = [i / 100 for i in range(101)]


def max_number(a: str | None, b: str | None, n_max: int = N_MAX) -> int:
    """The largest n <= n_max whose decimal string s has a <= s < b (None: unbounded)."""
    for length in range(len(str(n_max)), 0, -1):
        lo = 10 ** (length - 1)
        if a:
            if len(a) <= length:
                lo = max(lo, int(a.ljust(length, "0")))
            else:
                lo = max(lo, int(a[:length]) + 1)
        hi = 10**length - 1
        if b is not None:
            if len(b) > length:
                hi = int(b[:length])
            else:
                d = int(b) - 1
                if d < 0:
                    continue
                s = str(d).zfill(len(b)) + "9" * (length - len(b))
                hi = int(s)
        hi = min(hi, n_max)
        if hi >= max(lo, 10 ** (length - 1)) and len(str(hi)) == length:
            return hi
    return 0


def _check_max_number() -> None:
    import random
    rnd = random.Random(1)
    n_max = 1_418
    nums = sorted((str(n) for n in range(1, n_max + 1)))
    for _ in range(3000):
        a = rnd.choice([None] + nums)
        b = rnd.choice([None] + nums)
        if a is not None and b is not None and a >= b:
            continue
        inside = [int(s) for s in nums if (a is None or s >= a) and (b is None or s < b)]
        assert max_number(a, b, n_max) == (max(inside) if inside else 0), (a, b)


def frontier(margin: int):
    b = json.load(open(os.path.join(LAYOUT, "boundaries.json")))
    edges = [0] + [x["before"] for x in b["boundaries"]] + [b["bytes"]]
    pts = [[(0.0, 0)] for _ in range(8)]
    for r in csv.DictReader(open(os.path.join(LAYOUT, "offsets.csv"))):
        if not r["q_max"]:
            continue
        o = int(r["offset"])
        s = max(i for i in range(8) if o >= edges[i])
        f = (o - edges[s]) / (edges[s + 1] - edges[s])
        pts[s].append((f, int(r["q_max"])))
    curves = []
    for p in pts:
        p.sort()
        run, out = 0, []
        for f, q in p:
            run = max(run, q)
            out.append((f, run))
        out.append((1.0, run))
        curves.append(out)

    def q(s: int, f: float) -> float:
        c = curves[s]
        i = bisect.bisect_left([x for x, _ in c], f)
        if i == 0:
            return c[0][1]
        (f0, q0), (f1, q1) = c[i - 1], c[min(i, len(c) - 1)]
        return q0 if f1 == f0 else q0 + (q1 - q0) * (f - f0) / (f1 - f0)

    def F(f: float) -> float:
        return min(q(s, f) for s in range(8)) - margin

    return F, q, edges


def sample():
    rows = []
    with gzip.open(os.path.join(LAYOUT, "sample.csv.gz"), "rt") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def key_shares() -> dict[str, list[float]]:
    meta = json.load(open(os.path.join(REPO, "docs", "dataset_cards_metadata.json")))
    out = {}
    for t in KEYED:
        b = [v["bytes"] for v in meta[t].values()]
        tot = sum(b)
        out[t] = sorted((x / tot for x in b), reverse=True)
    return out


def curves(margin: int, rows: list | None = None, layouts: dict | None = None) -> dict:
    F, q, edges = frontier(margin)
    grid_f = [i / 1000 for i in range(1001)]
    grid_F = [F(f) for f in grid_f]

    def f_close(c: int) -> float:
        if c + 0 > grid_F[-1]:
            return 1.0
        return grid_f[bisect.bisect_left(grid_F, c)]

    rows = rows if rows is not None else sample()
    shares = key_shares()
    result = {"frontier": [(f, F(f)) for f in GRID], "tables": {}}
    for st in ("main", "scholar"):
        ents = [r for r in rows if (r["scholarly"] == "1") == (st == "scholar")]
        for table, field in FIELD.items():
            w = [(r["id"], float(r[field]) if field else 1.0) for r in ents]
            q_ids = sorted(((i[1:], x) for i, x in w if i[0] == "Q" and x > 0))
            p_bytes = sum(x for i, x in w if i[0] != "Q")
            tot = sum(x for _, x in q_ids) + p_bytes
            cum, acc = [], 0.0
            for s, x in q_ids:
                acc += x
                cum.append(acc)
            pub = PUBLISHED_GB[st][table] * 1e9
            keys = [k * pub for k in shares[table]] if table in KEYED else [pub]
            if not q_ids:  # (a resample with none of this table's rows: all at the end)
                for layout in (layouts or TARGETS):
                    closed = [0.0] * len(GRID)
                    closed[-1] = pub
                    result["tables"].setdefault(st, {}).setdefault(table, {})[layout] = {
                        "bytes": pub, "closed": closed, "parts": len(keys)}
                continue
            for layout, (target, min_n) in (layouts or TARGETS).items():
                closed = [0.0] * len(GRID)
                for kb in keys:
                    n = max(min_n, round(kb / target))
                    # cut points: the ids at the weighted quantiles k/n of the Q rows (a
                    # repeated cut merges two parts: their bytes go to the one left)
                    idx = sorted({min(len(q_ids) - 1, bisect.bisect_left(cum, acc * k / n)) for k in range(1, n)})
                    cuts = [None] + [q_ids[i][0] for i in idx] + [None]
                    w_at = [0.0] + [cum[i - 1] if i else 0.0 for i in idx] + [acc]
                    q_share = (tot - p_bytes) / tot
                    for k in range(len(cuts) - 1):
                        a, b = cuts[k], cuts[k + 1]
                        part = kb * q_share * (w_at[k + 1] - w_at[k]) / acc
                        if part <= 0:
                            continue
                        fc = f_close(max_number(a, b))
                        for g, f in enumerate(GRID):
                            if fc <= f + 1e-9 or g == len(GRID) - 1:
                                closed[g] += part
                    # (the P ids: their own part, closed at the end)
                    closed[-1] += kb * (1 - q_share)
                result["tables"].setdefault(st, {}).setdefault(table, {})[layout] = {
                    "bytes": pub, "closed": closed, "parts": sum(max(min_n, round(kb / target)) for kb in keys)}
    return result


def total(res: dict, layout: str) -> list[float]:
    """The share of both sets' published bytes (claims_labels too) final at each GRID f."""
    tot, b = [0.0] * len(GRID), 0.0
    for st, tables in res["tables"].items():
        for d in tables.values():
            tot = [x + y for x, y in zip(tot, d[layout]["closed"])]
            b += d[layout]["bytes"]
        cl = PUBLISHED_GB[st]["claims_labels"] * 1e9
        tot[-1] += cl
        b += cl
    return [x / b for x in tot]


def bootstrap(margin: int, n: int) -> None:
    """The closure curve's sampling error: the 101 windows resampled with replacement."""
    import random
    rows = sample()
    by: dict[str, list] = {}
    for r in rows:
        by.setdefault(r["window"], []).append(r)
    ws = sorted(by)
    rnd = random.Random(0)
    layout = {"regional, 128 MiB": TARGETS["regional, 128 MiB"]}
    cols = [i / 10 for i in range(1, 10)] + [0.95, 0.99]
    draws = []
    for _ in range(n):
        pick = [r for w in (rnd.choice(ws) for _ in ws) for r in by[w]]
        draws.append(total(curves(margin, pick, layout), "regional, 128 MiB"))
    point = total(curves(margin, rows, layout), "regional, 128 MiB")
    print(f"Both sets' published bytes final (regional, 128 MiB), point and 95% interval over {n} resamples:\n")
    print("| f | " + " | ".join(f"{c:.0%}" for c in cols) + " |\n|---|" + "---|" * len(cols))
    cells = []
    for c in cols:
        g = round(c * 100)
        v = sorted(d[g] for d in draws)
        cells.append(f"{point[g]:.0%} ({v[int(0.025 * n)]:.0%}–{v[int(0.975 * n) - 1]:.0%})")
    print("| final | " + " | ".join(cells) + " |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--margin", type=int, default=10_000)
    ap.add_argument("--json", default=os.path.join(HERE, "closure.json"))
    ap.add_argument("--bootstrap", type=int, default=0, help="resamples of the 101 windows")
    a = ap.parse_args()
    if a.bootstrap:
        bootstrap(a.margin, a.bootstrap)
        return
    _check_max_number()
    res = curves(a.margin)
    json.dump(res, open(a.json, "w"), indent=0)
    cols = [i / 10 for i in range(1, 11)]
    print("The frontier F(f) (every id up to it read), by the fraction f of each shard read:\n")
    print("| f | " + " | ".join(f"{c:.0%}" for c in cols) + " |\n|---|" + "---|" * len(cols))
    print("| F | " + " | ".join(f"Q{dict(res['frontier'])[c] / 1e6:.1f}M" for c in cols) + " |")
    for layout in TARGETS:
        print(f"\nShare of each table's published bytes final, by f ({layout}; claims_labels at the end):\n")
        print("| set | table | GB | parts | " + " | ".join(f"{c:.0%}" for c in cols) + " |")
        print("|---|---|---|---|" + "---|" * len(cols))
        tot_all = [0.0] * len(GRID)
        bytes_all = 0.0
        for st, tables in res["tables"].items():
            tot_set = [0.0] * len(GRID)
            b_set = 0.0
            for table, lay in tables.items():
                d = lay[layout]
                print(f"| {st} | {table} | {d['bytes'] / 1e9:.1f} | {d['parts']} | " + " | ".join(
                    f"{d['closed'][int(c * 100)] / d['bytes']:.0%}" for c in cols) + " |")
                tot_set = [x + y for x, y in zip(tot_set, d["closed"])]
                b_set += d["bytes"]
            cl = PUBLISHED_GB[st]["claims_labels"] * 1e9
            tot_set[-1] += cl
            b_set += cl
            print(f"| **{st}** | **all, claims_labels too** | {b_set / 1e9:.1f} | | " + " | ".join(
                f"**{tot_set[int(c * 100)] / b_set:.0%}**" for c in cols) + " |")
            tot_all = [x + y for x, y in zip(tot_all, tot_set)]
            bytes_all += b_set
        print(f"| **both** | **all** | {bytes_all / 1e9:.1f} | | " + " | ".join(
            f"**{tot_all[int(c * 100)] / bytes_all:.0%}**" for c in cols) + " |")


if __name__ == "__main__":
    main()
