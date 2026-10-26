"""The result tables (results/tables.md) from the measurements in results/ (stdlib only).

    python3 -I analyse.py results [RAW]   # RAW: xetlocal.py's per-file outputs, summed into results/hf_xet.json
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

# #188's page-level model, 512 KiB pages, 20261012 -> 20261019 (its dedup table), by set
# and table, and its release GB
MODEL_188 = {("main", "claims"): (0.05, 41.4), ("main", "labels"): (0.73, 6.6), ("main", "descriptions"): (0.63, 3.8),
             ("main", "aliases"): (0.73, 2.3), ("main", "links"): (0.65, 5.6), ("main", "entities"): (0.0, 0.2),
             ("scholar", "claims"): (0.92, 52.8), ("scholar", "descriptions"): (0.96, 7.5),
             ("scholar", "labels"): (0.98, 3.4)}
LEVELS = ("B0", "B1", "B2", "B5", "Breal")
LABEL = {"B0": "0 (A again)", "B1": "1x", "B2": "2x", "B5": "5x", "Breal": "the slice's real week"}


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def summarise_raw(raw: Path, res: Path) -> None:
    """xetlocal.py's per-file results (xetlocal-<B>[-fresh].json in `raw`), summed by
    (kind, set, table) into results/hf_xet.json."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from measure import table_of

    out = {}
    for path in sorted(raw.glob("xetlocal-*.json")):
        g = defaultdict(lambda: [0, 0, 0])
        for r in json.loads(path.read_text())["B"]:
            kind, name = r["file"].split(":", 1)
            st = table_of(name)
            for key in ((kind, *st), (kind, st[0], "all"), (kind, "all", "all"), ("all", "all", "all")):
                g["|".join(key)][0] += r["total_bytes"]
                g["|".join(key)][1] += r["total_bytes"] - r["new_bytes"]
                g["|".join(key)][2] += r["deduped_bytes_by_global_dedup"]
        out[path.stem.removeprefix("xetlocal-")] = dict(g)
    (res / "hf_xet.json").write_text(json.dumps(out, indent=1))


def real_by_table(res: Path, label: str) -> dict:
    """hf_xet's own dedup of B against A on the local CAS, by (kind, set, table):
    [B's bytes, bytes not sent, of which found by global dedup]."""
    p = res / "hf_xet.json"
    if not p.exists():
        return {}
    return {tuple(k.split("|")): v for k, v in json.loads(p.read_text()).get(label, {}).items()}


def totals(rows: list[dict]) -> dict:
    out = defaultdict(lambda: defaultdict(float))
    for r in rows:
        for key in ((r["kind"], r["set"], r["table"]), (r["kind"], r["set"], "all"), (r["kind"], "all", "all"),
                    ("all", "all", "all")):
            for k, v in r.items():
                if isinstance(v, (int, float)):
                    out[key][k] += v
    return out


def main() -> None:
    res = Path(sys.argv[1])
    if len(sys.argv) > 2:
        summarise_raw(Path(sys.argv[2]), res)
    lines = []
    w = lines.append
    # 1. by table at 1x
    t1 = totals(json.loads((res / "dedup-B1.json").read_text()))
    real, fresh = real_by_table(res, "B1"), real_by_table(res, "B1-fresh")
    w("## B1 (1x) against A: the share of B's bytes Xet would not upload, by set and table\n")
    w("| kind | set | table | B MB | identical files | identical pages | Xet chunks in A | Xet (simulated) | "
      "hf_xet 1.6.0 (local CAS) | global only, xorb | global only, shard (simulated) | hf_xet, cache emptied | #188 model |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for key in sorted(t1, key=lambda k: (k[0] == "all", k[0], k[1] == "all", k[1], k[2] == "all", k[2])):
        r = t1[key]
        b = r["bytes"]
        m = MODEL_188.get(key[1:]) if key[0] == "published" else None
        rb, fb = real.get(key), fresh.get(key)
        w(f"| {key[0]} | {key[1]} | {key[2]} | {b / 1e6:.2f} | {pct(r['identical_files'] / b)} | {pct(r['pages'] / b)} | "
          f"{pct(r['chunks'] / b)} | **{pct(r['xet'] / b)}** | {pct(rb[1] / rb[0]) if rb else ''} | "
          f"{pct(r['xet_global_xorb'] / b)} | {pct(r['xet_global_session'] / b)} | {pct(fb[1] / fb[0]) if fb else ''} | "
          f"{pct(m[0]) if m else ''} |")
    # 2. churn levels
    w("\n## By churn level: Xet's dedup (simulated, A's shards in the local cache), and hf_xet's own\n")
    w("| B | kind | main | scholar | both | hf_xet 1.6.0, local CAS | hf_xet, cache emptied |")
    w("|---|---|---|---|---|---|---|")
    for lv in LEVELS:
        p = res / f"dedup-{lv}.json"
        if not p.exists():
            continue
        t = totals(json.loads(p.read_text()))
        rr, ff = real_by_table(res, lv), real_by_table(res, f"{lv}-fresh")
        for kind in ("groups", "published"):
            cell = lambda s: pct(t[(kind, s, "all")]["xet"] / t[(kind, s, "all")]["bytes"])  # noqa: E731
            both = pct(t[(kind, "all", "all")]["xet"] / t[(kind, "all", "all")]["bytes"])
            a, f = rr.get((kind, "all", "all")), ff.get((kind, "all", "all"))
            w(f"| {LABEL[lv]} | {kind} | {cell('main')} | {cell('scholar')} | {both} | "
              f"{pct(a[1] / a[0]) if a else ''} | {pct(f[1] / f[0]) if f else ''} |")
    # 3. row groups
    w("\n## Published row groups by category (rowgroups.py): bytes, and the share in identical pages / in Xet chunks of A's\n")
    w("| B | set | table | touched entities (rows changed in the table) | aligned, untouched | aligned, touched | "
      "shifted, untouched | shifted, touched | in shifted untouched: dictionary columns / the rest |")
    w("|---|---|---|---|---|---|---|---|---|")
    for lv in LEVELS[1:]:
        p = res / f"rowgroups-{lv}.json"
        if not p.exists():
            continue
        for r in json.loads(p.read_text()):
            if r["bytes"] < 1e6:
                continue
            cells = []
            for c in ("aligned_untouched", "aligned_touched", "shifted_untouched", "shifted_touched"):
                b = r.get(f"{c}_bytes", 0)
                cb = r.get(f"{c}_chunk_bytes", 0)
                cells.append(f"{r.get(f'rgs_{c}', 0)} rg, {b / 1e6:.1f} MB: {pct(r.get(f'{c}_pages_same', 0) / b) if b else '-'}"
                             f" / {pct(r.get(f'{c}_chunks_same', 0) / cb) if cb else '-'}")
            sd, sp = r.get("shifted_untouched_dict_bytes", 0), r.get("shifted_untouched_plain_bytes", 0)
            dp = (f"{pct(r.get('shifted_untouched_dict_same', 0) / sd)} of {sd / 1e6:.1f} MB / "
                  f"{pct(r.get('shifted_untouched_plain_same', 0) / sp)} of {sp / 1e6:.1f} MB") if sd + sp else ""
            w(f"| {LABEL[lv]} | {r['set']} | {r['table']} | {r.get('touched', 0)} of {r.get('entities_b', 0)} "
              f"({r.get('count_changed', 0)} with rows added or removed) | " + " | ".join(cells) + f" | {dp} |")
    # 4. writer layouts
    p = res / "writer_shift.json"
    if p.exists():
        ws = json.loads(p.read_text())
        w("\n## The writer's layout (writer_shift.py): B's bytes in Xet chunks of A's (pages identical)\n")
        w("| data | churn | master: row groups every N rows | content-defined row groups | master, smaller CDC pages (32-128 KiB) |")
        w("|---|---|---|---|---|")
        for s in ("main", "scholar"):
            c = ws.get(f"claims_{s}")
            if c:
                w(f"| {s} claims, the slice (B1 against A; row_group_rows {c['row_group_rows_a']} -> {c['row_group_rows_b']}) "
                  f"| 1x | {pct(c['master']['chunk_dedup'])} ({pct(c['master']['page_dedup'])}) | "
                  f"{pct(c['content_defined']['chunk_dedup'])} ({pct(c['content_defined']['page_dedup'])}) | |")
        for placement in ("terms_clustered", "terms"):
          for table, r in (ws.get(placement) or {}).items():
            for mult in ("1x", "2x", "5x"):
                x = r.get(mult)
                if not x:
                    continue
                how = "clustered" if placement == "terms_clustered" else "uniform"
                w(f"| a {r['rows'] / 1e6:.0f}M-row {table} table ({x['touched']} rows touched, {how}) | {mult} | "
                  + " | ".join(f"{pct(x[v]['chunk_dedup'])} ({pct(x[v]['page_dedup'])})"
                               for v in ("master", "content_defined", "small_pages")) + " |")
    (res / "tables.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
