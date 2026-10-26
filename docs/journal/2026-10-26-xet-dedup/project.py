"""Xet's dedup at a release's scale (modelled), from what the slice measured (stdlib only):
results/project.json and project.txt.

    python3 -I project.py results

The slice's published claims files have a release's row groups (~128 MiB of decoded rows,
99k-155k rows), but a release's per-language files are millions of rows where the
slice's are thousands, and a release's claims row groups hold more entities (the main
set's ~10 statements an entity against the slice's c0's 62). So each table's figure is
taken where the slice's geometry and churn match a release's:

- **scholar claims:** the slice's shifted row groups at the release's touched entities a
  row group. At a release a row group holds ~4.6k scholarly entities (99,469 rows at
  ~21.5 rows an entity: 52.8 GB over 46.4M entities at the slice's 53 bytes a row), so
  0.92 touched a week at 1x (0.02%); the slice's ~2.7k at 2x (0.04%: B2) has 1.1, at 5x
  (0.1%: B5) 2.7. Every row group of a release's file is shifted (below), so the figure
  is the slice's shifted row groups', Xet chunk level: B2's 31.4% for 1x, B5's 31.2% for
  2x, and for 5x (4.6 touched a row group, past what the slice reaches) 28%, taken
  down by the measured cost of a touched entity in an aligned row group.
- **main claims:** a release's main row group holds ~15.5k entities (155k rows at ~10
  statements), so ~170 touched at 1x (1.1%); the slice's B5 (5.5%) has ~165: its 1.4%
  for 1x, and ~0 beyond.
- **the language tables:** writer_shift.py's 4M-row tables at the measured churn of each
  (TERMS_CHURN), clustered as #188 measured (central) and uniform (low).
- **the entities tables:** 0 (every edit changes `lastrevid`; 22k rows a page).
- **claims_labels:** as the main set's labels.
- **the group files** (the build branch): written without content-defined chunking
  (partitioning.core, polars-genson's writer; only compact.py's writers use CDC), in
  fixed 122,880-row row groups, their rows in dump order, and each group's chunks move
  with every entity deleted or created before them in the dump (the split cuts a chunk
  every 10,000 lines): nothing after a group's first change realigns. Taken as 0
  (the slice's 2-10% for claims is its few unshifted prefixes).

**Why every row group of a release's file shifts (master):** a part's row groups are cut
every `row_group_rows` rows from its start (sort_by_id.write_parts), `row_group_rows`
recomputed from each release's own decoded bytes (row_group_rows), and a key's parts are
equal rows (KeyPlan.part_rows) or bucket packs: one row more or less anywhere before a row
group moves it. A week adds or removes rows in every part (B1's main claims: 106 of 150
touched entities with rows added or removed; at a release, ~9,000 scholarly entities
changed a week over ~100 parts).

**Content-defined row groups** (a counterfactual, writer_shift.py): row groups cut at
content-defined entity boundaries, parts at fixed id regions (#180's design B): an
untouched row group re-encodes to the same bytes (the slice: 99.8-100%), a touched one as
measured (aligned, touched: 25-34% for scholar claims). At a release's 0.92 touched a
scholarly row group (Poisson), 40% are untouched.
"""

import json
import math
import sys
from pathlib import Path

# Release published bytes, GB (#188's dedup table: dataset_cards_metadata.json)
RELEASE_GB = {("main", "claims"): 41.4, ("main", "labels"): 6.6, ("main", "descriptions"): 3.8,
              ("main", "aliases"): 2.3, ("main", "links"): 5.6, ("main", "entities"): 0.2,
              ("scholar", "claims"): 52.8, ("scholar", "descriptions"): 7.5, ("scholar", "labels"): 3.5,
              ("scholar", "other"): 0.6, ("both", "claims_labels"): 4.4}
MODEL_188 = {1: 0.55, 2: 0.47, 5: 0.35}
# scholar claims: touched scholarly entities a release row group a week at 1x
LAMBDA_SCHOLAR = 0.92


def main() -> None:
    res = Path(sys.argv[1])
    ws = json.loads((res / "writer_shift.json").read_text())
    rg = {lv: {(r["set"], r["table"]): r for r in json.loads((res / f"rowgroups-{lv}.json").read_text())}
          for lv in ("B1", "B2", "B5")}
    chunk = lambda r, c: r.get(f"{c}_chunks_same", 0) / max(1, r.get(f"{c}_chunk_bytes", 0))  # noqa: E731
    sch = {lv: rg[lv][("scholar", "claims")] for lv in rg}
    s_shift = {1: chunk(sch["B2"], "shifted_touched"), 2: chunk(sch["B5"], "shifted_touched")}
    # 5x: the shifted figure less the measured cost of touched entities in an aligned row
    # group, from 2.7 to 4.6 touched (aligned untouched 99.8% -> aligned touched, B1's one)
    s_shift[5] = 0.28
    main_claims = {1: chunk(rg["B5"][("main", "claims")], "shifted_touched"), 2: 0.0, 5: 0.0}
    aligned_touched = sum(chunk(sch[lv], "aligned_touched") for lv in sch) / len(sch)

    def terms(table: str, mult: int, variant: str, placement: str) -> float | None:
        t = ws.get("terms_clustered" if placement == "clustered" else "terms", {}).get(table)
        return t[f"{mult}x"][variant]["chunk_dedup"] if t and f"{mult}x" in t else None

    out = {}
    lines = ["# Xet's dedup at a release (modelled from the slice's measurements)\n"]
    for layout in ("master", "content_defined", "content_defined+small_pages"):
        for placement in ("clustered", "uniform"):
            for mult in (1, 2, 5):
                d = {}
                if layout == "master":
                    d[("scholar", "claims")] = s_shift[mult]
                else:
                    p = math.exp(-LAMBDA_SCHOLAR * mult)
                    d[("scholar", "claims")] = p + (1 - p) * aligned_touched
                d[("main", "claims")] = main_claims[mult]
                tv = "master" if layout == "master" else ("content_defined" if layout == "content_defined" else "small_pages")
                for t in ("labels", "descriptions", "aliases", "links"):
                    v = terms(t, mult, tv, placement)
                    d[("main", t)] = v if v is not None else 0.0
                sd = terms("scholar descriptions", mult, tv, placement)
                d[("scholar", "descriptions")] = d[("scholar", "labels")] = sd if sd is not None else 0.0
                d[("main", "entities")] = d[("scholar", "other")] = 0.0
                d[("both", "claims_labels")] = d[("main", "labels")]
                gb = sum(RELEASE_GB[k] * d[k] for k in RELEASE_GB)
                total = sum(RELEASE_GB.values())
                key = f"{layout}/{placement}/{mult}x"
                out[key] = {"published_dedup": round(gb / total, 4), "published_dedup_gb": round(gb, 1),
                            "by_table": {f"{s} {t}": round(v, 4) for (s, t), v in d.items()}, "groups_dedup": 0.0}
    tables = list(RELEASE_GB)
    lines.append("| layout, churn placement | churn | " + " | ".join(f"{s} {t} ({RELEASE_GB[(s, t)]} GB)" for s, t in tables)
                 + " | published, all (128.7 GB) | #188's model |")
    lines.append("|---|---|" + "---|" * len(tables) + "---|---|")
    for key, v in out.items():
        layout, placement, mult = key.split("/")
        lines.append(f"| {layout}, {placement} | {mult} | "
                     + " | ".join(f"{100 * v['by_table'][f'{s} {t}']:.0f}%" for s, t in tables)
                     + f" | **{100 * v['published_dedup']:.0f}%** ({v['published_dedup_gb']} GB) | "
                     + f"{100 * MODEL_188[int(mult[:-1])]:.0f}% |")
    lines.append(f"\nInputs: scholar claims shifted row groups, Xet chunk level: B2 {100 * s_shift[1]:.1f}%, B5 "
                 f"{100 * s_shift[2]:.1f}%; aligned, touched (mean of B1, B2, B5) {100 * aligned_touched:.1f}%; main claims "
                 f"B5 {100 * main_claims[1]:.1f}%. Group files: 0 (no CDC).")
    (res / "project.json").write_text(json.dumps(out, indent=1))
    (res / "project.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
