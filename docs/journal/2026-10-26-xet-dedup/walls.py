"""The release walls re-modelled with the dedup measured here (modelled; stdlib only):
results/walls.txt, walls.json.

    git show origin/bench-results:runs/quick/b110f78160a96ddaa771ae7a0cedaf36c07c5968.json > rec179.json
    git show origin/docs/incremental-release:docs/journal/2026-10-26-incremental-release/model.py > model188.py
    python3 -I walls.py rec179.json ../../../bench model188.py results

- **master:** bench 17.9.0's own model (bench/overlap.py's release_wall: the build
  branch's uploads at the link's rate as a release's groups close, then finalise's three
  runs with their downloads and commits), on master b110f78's 6 runs (the record's
  `after` arm), means. Xet's dedup enters as fewer bytes sent: the group uploads'
  (`upload_eta_modelled`) times 1 - the groups' dedup, each set's and table's published
  bytes (`finalise_link_bytes["published"]`) times 1 - its own (project.py). The
  downloads are unchanged.
- **#188's variants** (F0, F1: a full release, no group files, finalise after the split
  or with the frontier; I0, I1: the incremental release), by #188's own simulator
  (model188.py's `incremental`, unchanged), its dedup argument the release-weighted
  published dedup. #188's churn by bytes, 0.93% a week (its two weeks' mean).
"""

import copy
import importlib.util
import json
import sys
from pathlib import Path

RATES = (7e6, 15e6, 30e6, 60e6, 120e6)
C_BYTES = 0.0093  # (#188: churn by bytes, the two weeks' mean, new entities included)


def hm(s: float) -> str:
    m = round(s / 60)
    return f"{m // 60}h{m % 60:02d}m"


def table_dedup(by_table: dict, set_: str, table: str) -> float:
    if table == "claims_labels":
        return by_table["both claims_labels"]
    key = f"{set_} {table}"
    if key in by_table:
        return by_table[key]
    return by_table.get(f"{set_} other", 0.0)


def with_dedup(run: dict, groups: float, by_table: dict | float) -> dict:
    r = copy.deepcopy(run)
    r["upload_eta_modelled"] = run["upload_eta_modelled"] * (1 - groups)
    pub = r["finalise_link_bytes"]["published"]
    for s, tables in pub.items():
        for t in tables:
            d = by_table if isinstance(by_table, float) else table_dedup(by_table, s, t)
            tables[t] = tables[t] * (1 - d)
    return r


def main() -> None:
    rec, bench, model188, res = sys.argv[1], sys.argv[2], sys.argv[3], Path(sys.argv[4])
    sys.path.insert(0, bench)
    import overlap

    overlap.RELEASE_LINK_RATES = RATES
    spec = importlib.util.spec_from_file_location("model188", model188)
    m188 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m188)
    runs = [u for x in json.load(open(rec)) for u in x["runs"] if u["arm"] == "after"]
    proj = json.loads((res / "project.json").read_text())
    mean = lambda xs: sum(xs) / len(xs)  # noqa: E731

    def master(groups: float, pub) -> list[float]:
        walls = [overlap.release_wall(with_dedup(r, groups, pub)) for r in runs]
        return [mean([w[f"release_wall_link_{x / 1e6:.0f}_modelled"] for w in walls]) for x in RATES]

    rows, out = [], {}

    def add(name: str, walls: list[float]) -> None:
        rows.append((name, walls))
        out[name] = [round(w) for w in walls]

    add("master, no dedup (bench 17.9.0's own)", master(0.0, 0.0))
    add("master, #188's 55% on every upload (its model, applied to master)", master(0.55, 0.55))
    for layout, label in (("master", "master's layout"), ("content_defined", "content-defined row groups"),
                          ("content_defined+small_pages", "content-defined row groups, language tables' pages 32-128 KiB")):
        for mult in (1, 2, 5):
            for placement in (("clustered", "uniform") if mult == 1 else ("clustered",)):
                p = proj[f"{layout}/{placement}/{mult}x"]
                add(f"master, measured: {label}, churn {mult}x ({placement}), published "
                    f"{100 * p['published_dedup']:.0f}%, groups 0%", master(0.0, p["by_table"]))
    for mult in (1, 2, 5):
        for label, dv in ((f"#188's {100 * m188_d:.0f}%" if (m188_d := {1: 0.55, 2: 0.47, 5: 0.35}[mult]) else "", m188_d),
                          ("measured, master's layout", proj[f"master/clustered/{mult}x"]["published_dedup"]),
                          ("measured, content-defined row groups",
                           proj[f"content_defined/clustered/{mult}x"]["published_dedup"])):
            if mult == 1:
                for vname, kw in (("F1 (full, no group files, frontier)", {"frontier": True}),):
                    walls = [mean([m188.incremental(r, x, 1.0, dv, mu=1.0, hash_share=0.0, **kw)["wall"] for r in runs])
                             for x in RATES]
                    add(f"{vname}, dedup {label} ({100 * dv:.0f}%)", walls)
            for vname, kw in (("I0 (incremental)", {"frontier": False}), ("I1 (incremental + frontier)", {"frontier": True})):
                walls = [mean([m188.incremental(r, x, C_BYTES * mult, dv, **kw)["wall"] for r in runs]) for x in RATES]
                add(f"{vname}, churn {mult}x, dedup {label} ({100 * dv:.0f}%)", walls)
    for vname, kw in (("F1, no dedup", {"frontier": True, "mu": 1.0, "hash_share": 0.0}),):
        walls = [mean([m188.incremental(r, x, 1.0, 0.0, **kw)["wall"] for r in runs]) for x in RATES]
        add(vname, walls)
    walls = [mean([m188.incremental(r, x, C_BYTES, 0.0, frontier=True)["wall"] for r in runs]) for x in RATES]
    add("I1, churn 1x, no dedup", walls)
    head = "| | " + " | ".join(f"{x / 1e6:.0f} MB/s" for x in RATES) + " |"
    lines = ["# Release walls with the measured dedup (modelled; means of bench 17.9.0's 6 runs of master b110f78)\n",
             head, "|---|" + "---|" * len(RATES)]
    lines += [f"| {n} | " + " | ".join(hm(w) for w in ws) + " |" for n, ws in rows]
    (res / "walls.json").write_text(json.dumps(out, indent=1))
    (res / "walls.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
