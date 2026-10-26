"""Churn tables from churn.py's and bside.py's records (stdlib only): results/churn.txt
and results/churn.json (the rates dedup.py and model.py take).

    python3 -I analyse.py raw results
"""

import collections
import csv
import gzip
import json
import random
import sys

RANGES = [("Q0–Q25M (the oldest items)", 0, 25_000_000), ("Q25M–Q60M (mostly scholarly)", 25_000_000, 60_000_000),
          ("Q60M–Q100M", 60_000_000, 100_000_000), ("Q100M–Q142M", 100_000_000, 10**10)]
TABLES = ("claims", "labels", "descriptions", "aliases", "links", "entities")
FIELD = {"claims": "d_claims", "labels": "d_labels", "descriptions": "d_descriptions", "aliases": "d_aliases",
         "links": "d_sitelinks"}
BYTES = {"claims": "claims_bytes", "labels": "labels_bytes", "descriptions": "descriptions_bytes",
         "aliases": "aliases_bytes", "links": "sitelinks_bytes", "entities": None}


def touches(r: dict, table: str) -> bool:
    """Whether the entity's rows in `table` differ between the two dumps."""
    if r["status"] in ("new", "gone"):
        return True
    if r["status"] == "unchanged":
        return False
    if table == "entities":  # (lastrevid and modified: every edit)
        return True
    return r["status"] == "changed" and bool(r[FIELD[table]])


def ci(points: list, stat, n: int = 2000, seed: int = 1) -> tuple[float, float]:
    """A 95% interval for the ratio sum(num) / sum(den) over the points, each point's
    (num, den) = stat(its rows), the points resampled (a cluster bootstrap)."""
    agg = [stat([q]) for q in points]
    rnd = random.Random(seed)
    k = len(agg)
    xs = []
    for _ in range(n):
        s = [agg[rnd.randrange(k)] for _ in range(k)]
        xs.append(sum(a for a, _ in s) / max(1e-12, sum(b for _, b in s)))
    xs.sort()
    return xs[int(0.025 * n)], xs[int(0.975 * n) - 1]


def ratio(stat, ps) -> float:
    a = [stat([q]) for q in ps]
    return sum(x for x, _ in a) / max(1e-12, sum(y for _, y in a))


def pct(x: float, d: int = 2) -> str:
    return f"{100 * x:.{d}f}%"


def main() -> None:
    raw, out = sys.argv[1], sys.argv[2]
    rows = list(csv.DictReader(gzip.open(f"{raw}/entities.csv.gz", "rt")))
    side = {(r["point"], r["num"]): r for r in csv.DictReader(gzip.open(f"{raw}/bside.csv.gz", "rt"))}
    wins = {str(w["point"]): w for w in map(json.loads, open(f"{raw}/windows.jsonl"))}
    tails = json.load(open(f"{raw}/tails.json"))
    pairs = sorted({r["pair"] for r in rows}, key=lambda p: (p.split("-")[1], p))
    lines = []
    res: dict = {"pairs": {}}

    def emit(s=""):
        lines.append(s)

    for r in rows:  # (the set and bytes of B's entity: gone entities have none)
        b = side.get((r["point"], r["num"]))
        r["set"] = ("scholar" if b["scholarly"] == "1" else "main") if b else (
            "scholar" if r.get("scholarly") == "1" else "main")
        r["bytes"] = int(b["line_bytes"]) if b else int(r["line_a"] or 0)
        for t, k in BYTES.items():
            r[f"b_{t}"] = int(b[k]) if (b and k) else (r["bytes"] if t == "entities" else 0)

    emit("# Churn between consecutive weekly dumps (measured)\n")
    emit(f"Entities compared: {len(side):,} in {len(wins)} windows of 20261019 (each with its id range in")
    emit("20261012 and 20261005). Intervals: 95%, the 32 windows resampled (a cluster bootstrap).\n")
    emit("## Status by pair (entity-weighted; bytes: the newer line's bytes)\n")
    emit("| pair | entities | unchanged line | same lastrevid | changed | meta only | gone | new (in range) | changed+gone, entities | 95% CI | by bytes | 95% CI |")
    emit("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for p in pairs:
        rs = [r for r in rows if r["pair"] == p]
        by = collections.defaultdict(list)
        for r in rs:
            by[r["point"]].append(r)
        pts = list(by.values())
        c = collections.Counter(r["status"] for r in rs)
        same_rev = c["unchanged"] + sum(1 for r in rs if r.get("same_rev") == "1")
        hit = lambda r: r["status"] not in ("unchanged",)
        f_e = lambda ps: (sum(hit(r) for q in ps for r in q), sum(len(q) for q in ps))
        f_b = lambda ps: (sum(r["bytes"] for q in ps for r in q if hit(r)), sum(r["bytes"] for q in ps for r in q))
        lo, hi = ci(pts, f_e)
        blo, bhi = ci(pts, f_b)
        emit(f"| {p} | {len(rs):,} | {c['unchanged']:,} | {same_rev:,} | {c['changed']} | {c['meta_only']} | {c['gone']} |"
             f" {c['new']} | **{pct(ratio(f_e, pts))}** | {pct(lo)}–{pct(hi)} | {pct(ratio(f_b, pts))} | {pct(blo)}–{pct(bhi)} |")
        res["pairs"][p] = {"entities": len(rs), "status": dict(c), "same_rev": same_rev, "rate": ratio(f_e, pts),
                           "rate_ci": [lo, hi], "rate_bytes": ratio(f_b, pts), "rate_bytes_ci": [blo, bhi]}
        # by set and table
        tab = {}
        for st in ("main", "scholar"):
            q = [[r for r in pt if r["set"] == st] for pt in pts]
            q = [x for x in q if x]
            tab[st] = {}
            for t in TABLES:
                fe = lambda ps, t=t: (sum(touches(r, t) for x in ps for r in x), sum(len(x) for x in ps))
                fb = lambda ps, t=t: (sum(r[f"b_{t}"] for x in ps for r in x if touches(r, t)),
                                      sum(r[f"b_{t}"] for x in ps for r in x))
                tab[st][t] = {"entities": ratio(fe, q), "entities_ci": ci(q, fe, 1000), "bytes": ratio(fb, q),
                              "bytes_ci": ci(q, fb, 1000), "n": sum(len(x) for x in q)}
        res["pairs"][p]["tables"] = tab
    emit("\n- **same lastrevid**: the identical lines, plus any differing line whose lastrevid did not change: "
         "there are none. Every differing line has a new lastrevid.")
    emit("- **meta only**: lastrevid and modified changed, the content not (an edit and its revert in the week).\n")

    emit("## New entities (created between the dumps; the end of shard 7, tails.py)\n")
    emit("| pair | ids added (max id) | live share at the end | new bz2 bytes (×8 shards) | new entities, est. | share of 121.8M |")
    emit("|---|---|---|---|---|---|")
    for t in tails:
        est = t["new_entities_est"]
        emit(f"| {t['older']}-{t['newer']} | {t['new_ids_range']:,} | {t['live_share_window']:.2f} | "
             f"{t['new_bz2_bytes_all_shards'] / 1e6:.1f} MB | {est:,} | {pct(est / 121.8e6, 3)} |")
        res["pairs"].setdefault(f"{t['older']}-{t['newer']}", {})["new_entities"] = est
        res["pairs"][f"{t['older']}-{t['newer']}"]["new_bz2_bytes"] = t["new_bz2_bytes_all_shards"]

    emit("\n## Rows that change, by set and table (the share of each table's entities, and of its bytes, "
         "whose rows differ)\n")
    for p in pairs:
        emit(f"**{p}**\n")
        emit("| set | table | entities sampled | entities touched | 95% CI | bytes touched | 95% CI |")
        emit("|---|---|---|---|---|---|---|")
        for st, tt in res["pairs"][p]["tables"].items():
            for t, v in tt.items():
                emit(f"| {st} | {t} | {v['n']:,} | {pct(v['entities'])} | {pct(v['entities_ci'][0])}–{pct(v['entities_ci'][1])} |"
                     f" {pct(v['bytes'])} | {pct(v['bytes_ci'][0])}–{pct(v['bytes_ci'][1])} |")
        emit()

    emit("## By window: id range, shard, set mix (each pair's changed + gone share)\n")
    emit("| point | shard | ids | entities | scholarly | " + " | ".join(pairs) + " |")
    emit("|---|---|---|---|---|" + "---|" * len(pairs))
    per = collections.defaultdict(lambda: collections.Counter())
    for r in rows:
        per[(r["point"], r["pair"])][r["status"] != "unchanged"] += 1
    sch = collections.Counter(b["point"] for b in side.values() if b["scholarly"] == "1")
    nb = collections.Counter(b["point"] for b in side.values())
    disp = {p: [] for p in pairs}
    for k in sorted(wins, key=int):
        w = wins[k]
        cells = []
        for p in pairs:
            c = per[(k, p)]
            n = c[True] + c[False]
            disp[p].append((c[True], n))
            cells.append(f"{c[True]}/{n} ({pct(c[True] / n, 1)})")
        emit(f"| {k} | {w['shard']} | Q{w['lo'] / 1e6:.2f}M–Q{w['hi'] / 1e6:.2f}M | {nb[k]:,} | {pct(sch[k] / nb[k], 0)} | "
             + " | ".join(cells) + " |")
    emit()
    # by shard and by id range, pooled (with each group's windows resampled)
    groups = {"shard": lambda w: f"shard {w['shard']}",
              "id range": lambda w: next(f"{a}" for a, lo, hi in RANGES if lo <= w["lo"] < hi)}
    for gname, key in groups.items():
        emit(f"**By {gname}** (changed + gone + new in range, entities; 95% CI over the group's windows):\n")
        emit(f"| {gname} | windows | entities | " + " | ".join(pairs) + " |")
        emit("|---|---|---|" + "---|" * len(pairs))
        labels = sorted({key(w) for w in wins.values()}, key=lambda g: [x[0] for x in RANGES].index(g)
                        if gname == "id range" else g)
        for g in labels:
            ks = [k for k, w in wins.items() if key(w) == g]
            cells = []
            for p in pairs:
                agg = [(per[(k, p)][True], per[(k, p)][True] + per[(k, p)][False]) for k in ks]
                rate = sum(a for a, _ in agg) / sum(b for _, b in agg)
                lo, hi = ci([[x] for x in agg], lambda q: q[0][0], 1000)
                cells.append(f"{pct(rate)} ({pct(lo)}–{pct(hi)})")
            emit(f"| {g} | {len(ks)} | {sum(nb[k] for k in ks):,} | " + " | ".join(cells) + " |")
        emit()
    # overdispersion across windows: the variance of the counts against a binomial's
    emit("**Concentration across windows** (variance of the windows' changed counts over a binomial's at the pooled rate; "
         "1 = uniform):\n")
    for p in pairs:
        xs = disp[p]
        rate = sum(x for x, _ in xs) / sum(n for _, n in xs)
        chi = sum((x - n * rate) ** 2 / (n * rate * (1 - rate)) for x, n in xs) / (len(xs) - 1)
        top = sorted(xs, key=lambda t: -t[0])
        share4 = sum(x for x, _ in top[:4]) / max(1, sum(x for x, _ in xs))
        emit(f"- {p}: dispersion {chi:.1f}; the 4 busiest windows of 32 hold {pct(share4, 0)} of the changes")
        res["pairs"][p]["dispersion"] = chi
        res["pairs"][p]["top4_share"] = share4
    emit()

    emit("## What changed (changed entities)\n")
    for p in pairs:
        rs = [r for r in rows if r["pair"] == p and r["status"] == "changed"]
        fields = collections.Counter()
        for r in rs:
            for f in ("d_labels", "d_descriptions", "d_aliases", "d_sitelinks", "d_claims"):
                if r[f]:
                    fields[f[2:]] += 1
        only = collections.Counter(
            "+".join(f[2:] for f in ("d_labels", "d_descriptions", "d_aliases", "d_sitelinks", "d_claims") if r[f])
            for r in rs)
        props = collections.Counter(x for r in rs for x in r["d_claims"].split(";") if x)
        langs = collections.Counter(x for r in rs for x in r["d_descriptions"].split(";") if x)
        llangs = collections.Counter(x for r in rs for x in r["d_labels"].split(";") if x)
        cls = collections.Counter(r["p31"].split(";")[0] if r["p31"] else "(none)" for r in rs)
        days = collections.Counter(r["modified_b"][:10] for r in rs)
        emit(f"**{p}** ({len(rs)} changed)\n")
        emit("- fields: " + ", ".join(f"{k} {v} ({pct(v / len(rs), 0)})" for k, v in fields.most_common()))
        emit("- combinations: " + ", ".join(f"{k} {v}" for k, v in only.most_common(8)))
        emit("- properties: " + ", ".join(f"{k} {v}" for k, v in props.most_common(12)))
        emit("- description languages: " + ", ".join(f"{k} {v}" for k, v in langs.most_common(10)))
        emit("- label languages: " + ", ".join(f"{k} {v}" for k, v in llangs.most_common(10)))
        emit("- first P31: " + ", ".join(f"{k} {v}" for k, v in cls.most_common(10)))
        emit("- last modified (day): " + ", ".join(f"{k} {v}" for k, v in sorted(days.items())))
        emit()
        res["pairs"][p]["fields"] = dict(fields)
        res["pairs"][p]["desc_langs"] = dict(langs)
        res["pairs"][p]["label_langs"] = dict(llangs)

    # the two-week pair against the two weeks
    if len(pairs) == 3:
        two = next(p for p in pairs if p.split("-")[0] == min(x.split("-")[0] for x in pairs)
                   and p.split("-")[1] == max(x.split("-")[1] for x in pairs))
        a, b2 = sorted(p for p in pairs if p != two)
        ids = lambda p: {(r["point"], r["num"]) for r in rows if r["pair"] == p and r["status"] != "unchanged"}
        x, y, z = ids(a), ids(b2), ids(two)
        emit("## Two weeks against one\n")
        emit(f"- touched in {a}: {len(x)}; in {b2}: {len(y)}; in both weeks: {len(x & y)} "
             f"({pct(len(x & y) / max(1, len(y)), 0)} of the later week's)")
        emit(f"- touched over {two}: {len(z)}, against {len(x | y)} touched in either week (an entity edited and "
             f"reverted across the weeks is unchanged over two)")
        res["two_weeks"] = {"w1": len(x), "w2": len(y), "both": len(x & y), "two": len(z)}

    open(f"{out}/churn.txt", "w").write("\n".join(lines) + "\n")
    json.dump(res, open(f"{out}/churn.json", "w"), indent=1, default=list)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
