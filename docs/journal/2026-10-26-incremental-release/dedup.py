"""How much of a release's upload Xet could deduplicate against the release before
(modelled from the measured churn; stdlib only): results/dedup.txt and dedup.json.

The published files are written with Parquet content-defined chunking (compact.py:
`use_content_defined_chunking`), so a column's page boundaries follow its values, not row
counts, and a page none of whose rows changed is re-encoded to the same bytes wherever it
lands. Xet chunks the file bytes (content-defined, ~64 KiB), so an unchanged page's chunks
are found in the release before. The upload's dedup is then about the share of the
published bytes in untouched pages, less a Xet chunk at each edge of a touched run.

A page of a table, in string order, holds E consecutive entities of its set, where E is
the page's uncompressed bytes over the column's bytes an entity:
- **claims:** a statement's widest leaf columns (its id, a snak or reference hash: ~45
  bytes a statement) times the entity's statements;
- **a language or site table:** one key's rows, a row ~(value bytes + 4); its page holds
  that many rows of the key, so E = rows / (the key's rows an entity of the set);
- **entities:** a row an entity (~24 bytes in its widest columns).

String order mixes digit lengths: the block of ids starting `Q45` holds Q45000000–Q45999999
(8 digits) and Q4500000–Q4599999 (7 digits), the shorter ones a tenth as many. A page
at window w's ids therefore also holds ids at w's numbers times 10^k (k = -2, -1, +1),
in proportion to the set's density there. Each length's touched rate is that of the
window nearest in log(id).

P(untouched) = Π over lengths (1 - r)^(E × share), r the window's rate of the table's
(or the key's) touched entities among the set's entities. Weighted by the window's
bytes of the table (the windows are equal-byte samples of the dump).

A window's rate is shrunk towards its set's (a Beta(1/2, 1/2 / pooled rate) prior), so
that a window where none changed is not taken as never changing; a rate never seen in the
whole sample is taken as half an entity in it (a language or site key: its share of the
table's rows of that half, so that a table's keys never seen add up to half an entity). The 2× and 5× churn take every rate
times 2 and 5 (P to the power 2 and 5).

Two bounds and a central figure:
- **uniform:** one rate for the whole table (no concentration): the lower bound;
- **windowed:** each window's own rate (the measured concentration across id ranges),
  uniform within a window;
- the Xet edge loss: (1 - P) × 64 KiB over the page's compressed bytes (a zstd ratio
  of 4), taken off.

Not modelled: a row group's start shifting (rows inserted before it; ~1 page a column a
row group, ~0.5%), and dictionary-encoded column chunks, whose indices shift when a value
is inserted early in the chunk (their low-cardinality columns are small). Both lower the
dedup a little. Nothing here is measured but the churn.

    python3 -I dedup.py raw results ../../dataset_cards_metadata.json
"""

import collections
import csv
import gzip
import json
import math
import sys

PUBLISHED_GB = {  # bench 17.4.0's published bytes by set and table (2026-10-26-progressive-finalise)
    "main": {"claims": 41.4, "labels": 6.6, "descriptions": 3.8, "aliases": 2.3, "links": 5.6, "entities": 0.2,
             "claims_labels": 2.8},
    "scholar": {"claims": 52.8, "labels": 3.5, "descriptions": 7.5, "aliases": 0.1, "links": 0.0, "entities": 0.5,
                "claims_labels": 1.6},
}
LANG = {"labels": ("d_labels", "labels_bytes", "labels_n", 28), "descriptions": ("d_descriptions", "descriptions_bytes",
        "descriptions_n", 28), "aliases": ("d_aliases", "aliases_bytes", "aliases_n", 14),
        "links": ("d_sitelinks", "sitelinks_bytes", "sitelinks_n", 40)}
MAIN_ENTITIES_20260507 = 73_769_737
STMT_BYTES = 45
ENTITY_ROW_BYTES = 24
XET_CHUNK = 64 << 10
ZSTD_RATIO = 4.0
MAX_ID = 141_800_000
MULT = (1, 2, 5)  # churn multipliers: each window's rate times m, so P becomes P^m


def main() -> None:
    raw, out, meta_path = sys.argv[1], sys.argv[2], sys.argv[3]
    meta = json.load(open(meta_path))
    rows = list(csv.DictReader(gzip.open(f"{raw}/entities.csv.gz", "rt")))
    side = list(csv.DictReader(gzip.open(f"{raw}/bside.csv.gz", "rt")))
    wins = {w["point"]: w for w in map(json.loads, open(f"{raw}/windows.jsonl"))}
    sets = {(r["point"], r["num"]): ("scholar" if r["scholarly"] == "1" else "main") for r in side}
    by_w = collections.defaultdict(list)
    for r in side:
        by_w[int(r["point"])].append(r)
    centre = {int(k): math.sqrt(w["lo"] * w["hi"]) for k, w in wins.items()}
    span = {int(k): w["hi"] - w["lo"] for k, w in wins.items()}

    def nearest(n: float) -> int:
        return min(centre, key=lambda k: abs(math.log(max(n, 1)) - math.log(centre[k])))

    # per key: rows a main entity (dataset_cards_metadata, 20260507's main set) and bytes share
    keys = {t: {k: v["rows"] / MAIN_ENTITIES_20260507 for k, v in meta[t].items()} for t in LANG}
    kbytes = {t: {k: v["bytes"] for k, v in meta[t].items()} for t in LANG}
    pairs = ["20261012-20261019", "20261005-20261012"]
    res = {}
    lines = ["# Xet dedup of a release's upload against the one before (modelled)\n"]
    for page_kib in (256, 512, 1024):
        page = page_kib << 10
        for p in pairs:
            touched = {}  # (point, num) -> set of (table) or (table, key)
            for r in rows:
                if r["pair"] != p or r["status"] == "unchanged":
                    continue
                k = (r["point"], r["num"])
                t = touched.setdefault(k, set())
                t.add("entities")
                if r["status"] in ("new", "gone"):
                    t.update(("claims", "*labels", "*descriptions", "*aliases", "*links"))
                elif r["status"] == "changed":
                    if r["d_claims"]:
                        t.add("claims")
                    for tab, (f, _, _, _) in LANG.items():
                        for key in filter(None, r[f].split(";")):
                            t.add((tab, key))
            for st in ("main", "scholar"):
                # per window: the set's entities, its density (entities an id), and rates
                dens, n_set = {}, {}
                for w, es in by_w.items():
                    mine = [e for e in es if (sets[(e["point"], e["num"])] == st)]
                    n_set[w] = mine
                    dens[w] = 8 * len(mine) / max(1, span[w])

                tags = {}
                for w, es in n_set.items():
                    c = collections.Counter()
                    for e in es:
                        for tg in touched.get((e["point"], e["num"]), ()):
                            c[tg] += 1
                    tags[w] = c

                pooled = collections.Counter()
                for c in tags.values():
                    pooled.update(c)
                n_all = sum(len(x) for x in n_set.values())

                def pooled_rate(pred, fw: float = 1.0) -> float:
                    """The set's rate over every window, at least half an entity in the sample
                    (a key: its share `fw` of the table's rows of that half)."""
                    return max(0.5 * fw, sum(pooled[tg] for tg in pred)) / max(1, n_all)

                def rate(w: int, pred, fw: float = 1.0) -> float:
                    """The share of the set's entities in window w with a touched tag in `pred`
                    (a tuple of tags: they never overlap on one entity), shrunk towards the
                    set's rate: a Beta(1/2, 1/2 / pooled) prior, so a window with none seen is
                    not taken as never changing."""
                    n = len(n_set[w])
                    r0 = pooled_rate(pred, fw)
                    return (sum(tags[w][tg] for tg in pred) + 0.5) / (n + 0.5 / r0)

                def mix(w: int) -> list[tuple[int, float]]:
                    """(window, share of a page's entities) over the digit lengths at w's ids."""
                    c = centre[w]
                    parts = []
                    for k in (-2, -1, 0, 1):
                        n = c * 10 ** k
                        if n < 1 or n > MAX_ID:
                            continue
                        v = w if k == 0 else nearest(n)
                        parts.append((v, dens[v] * 10 ** k))
                    tot = sum(x for _, x in parts) or 1
                    return [(v, x / tot) for v, x in parts]

                bsum = {(w, k): sum(int(e[k]) for e in n_set[w]) for w in n_set
                        for k in ("labels_bytes", "descriptions_bytes", "aliases_bytes", "sitelinks_bytes",
                                  "claims_bytes", "line_bytes")}
                res.setdefault(page_kib, {}).setdefault(p, {})[st] = tab_res = {}
                for tab in ("claims", "labels", "descriptions", "aliases", "links", "entities"):
                    gb = PUBLISHED_GB[st][tab]
                    if tab in LANG:
                        f, bkey, nkey, over = LANG[tab]
                        # the table's keys by bytes; each key's E and rates
                        tot_b = sum(kbytes[tab].values())
                        tot_rows = sum(v["rows"] for v in meta[tab].values())
                        top = sorted(kbytes[tab].items(), key=lambda kv: -kv[1])
                        cover = 0.0
                        num_w = num_u = den = 0.0
                        num_k = {m: 0.0 for m in MULT}
                        es_all = [e for w in by_w for e in n_set[w]]
                        n_entries = sum(int(e[nkey]) for e in es_all) or 1
                        val = max(4.0, sum(int(e[bkey]) for e in es_all) / n_entries - over)
                        rows_page = page / (val + 4)
                        for key, b in top:
                            share_b = b / tot_b
                            per_entity = keys[tab][key]  # rows of the key an entity
                            if per_entity <= 0:
                                continue
                            E = rows_page / per_entity
                            pred = ((tab, key), f"*{tab}")
                            fw = meta[tab][key]["rows"] / tot_rows
                            glob = pooled_rate(pred, fw)
                            for w in by_w:
                                bw = bsum[(w, bkey)]
                                if not bw:
                                    continue
                                P = math.prod((1 - rate(v, pred, fw)) ** (E * s) for v, s in mix(w))
                                comp = page / ZSTD_RATIO
                                for m in MULT:
                                    Pm = P ** m
                                    num_k[m] += share_b * bw * max(0.0, Pm * (1 - (1 - Pm) * XET_CHUNK / comp))
                                num_w += share_b * bw * P
                                num_u += share_b * bw * (1 - glob) ** E
                                den += share_b * bw
                            cover += share_b
                            if cover > 0.95:
                                break
                        E_show = rows_page
                    else:
                        pred = ("claims",) if tab == "claims" else ("entities",)
                        es_all = [e for w in by_w for e in n_set[w]]
                        glob = pooled_rate(pred)
                        num_w = num_u = den = 0.0
                        num_k = {m: 0.0 for m in MULT}
                        stm = sum(int(e["statements_n"]) for e in es_all) / max(1, len(es_all))
                        E = page / (STMT_BYTES * max(1.0, stm)) if tab == "claims" else page / ENTITY_ROW_BYTES
                        for w in by_w:
                            bw = bsum[(w, "claims_bytes" if tab == "claims" else "line_bytes")]
                            if not bw:
                                continue
                            P = math.prod((1 - rate(v, pred)) ** (E * s) for v, s in mix(w))
                            comp = page / ZSTD_RATIO
                            num_w += bw * P
                            for m in MULT:
                                Pm = P ** m
                                num_k[m] += bw * max(0.0, Pm * (1 - (1 - Pm) * XET_CHUNK / comp))
                            num_u += bw * (1 - glob) ** E
                            den += bw
                        E_show = E
                    tab_res[tab] = {"gb": gb, "E": round(E_show), "uniform": num_u / den if den else 1.0,
                                    "windowed": num_w / den if den else 1.0,
                                    "xet": num_k[1] / den if den else 1.0,
                                    "xet_mult": {m: (num_k[m] / den if den else 1.0) for m in MULT}}
    # tables
    for page_kib in (256, 512, 1024):
        lines.append(f"\n## Pages of {page_kib} KiB (uncompressed)\n")
        lines.append("| pair | set | table | GB | E (claims, entities: entities a page; others: rows a page) | "
                     "untouched, uniform | untouched, windowed | dedup, windowed less Xet edges | GB deduplicated |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for p in pairs:
            tot = tot_gb = 0.0
            for st in ("main", "scholar"):
                for tab, v in res[page_kib][p][st].items():
                    d = v["gb"] * v["xet"]
                    tot += d
                    tot_gb += v["gb"]
                    lines.append(f"| {p} | {st} | {tab} | {v['gb']} | {v['E']:,} | {100 * v['uniform']:.0f}% | "
                                 f"{100 * v['windowed']:.0f}% | {100 * v['xet']:.0f}% | {d:.1f} |")
            cl = sum(PUBLISHED_GB[s]["claims_labels"] for s in PUBLISHED_GB)
            lines.append(f"| {p} | both | claims_labels (taken as the labels' dedup) | {cl} | | | | | |")
            lines.append(f"| **{p}** | **both** | **all but claims_labels** | **{tot_gb:.1f}** | | | | "
                         f"**{100 * tot / tot_gb:.0f}%** | **{tot:.1f}** |")
            res[page_kib][p]["total_dedup_gb"] = tot
            res[page_kib][p]["total_gb"] = tot_gb
            res[page_kib][p]["total_dedup_gb_mult"] = {
                m: sum(v["gb"] * v["xet_mult"][m] for st in ("main", "scholar") for v in res[page_kib][p][st].values())
                for m in MULT}
            res[page_kib][p]["total_dedup_gb_uniform"] = sum(
                v["gb"] * v["uniform"] for st in ("main", "scholar") for v in res[page_kib][p][st].values())
            lines.append(f"\nAt 2× and 5× the churn: " + ", ".join(
                f"{m}×: {res[page_kib][p]['total_dedup_gb_mult'][m]:.1f} GB ({100 * res[page_kib][p]['total_dedup_gb_mult'][m] / tot_gb:.0f}%)"
                for m in MULT) + f"; uniform churn (no concentration): {res[page_kib][p]['total_dedup_gb_uniform']:.1f} GB\n")
    open(f"{out}/dedup.txt", "w").write("\n".join(lines) + "\n")
    json.dump(res, open(f"{out}/dedup.json", "w"), indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
