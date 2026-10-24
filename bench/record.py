"""Add a run to the results store (the `bench-results` branch): a line at the end of
records.jsonl, tagged with the harness version and the tier, and the rounds' raw results
as runs/<tier>/<commit>.json. Nothing already recorded changes. RECORDS.md is a view of
records.jsonl, rewritten from it: one table per harness major version and tier, newest
first.

Usage: record.py <store dir> <run url> <results.json>...  (one per round)
       record.py <store dir> --view  (rewrites RECORDS.md alone)
"""

import json
import statistics
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

from combine import paired, rounds_of
from version import VERSION

HEAD = """# Records

One entry per run, appended by the Benchmark workflow (`bench/record.py`) to
`records.jsonl`, of which this file is a view: a push to master in the quick tier (the
new commit against the previous one), or a run of the full tier from the Actions tab.
Each entry carries the version of the harness that measured it (`bench/pyproject.toml`,
changes in `bench/CHANGELOG.md`): entries compare only within a major version. Speed is
the commit before's mean wall time over the commit after's, over the rounds.
"""


def view(store: Path) -> None:
    lines = [json.loads(l) for l in (store / "records.jsonl").read_text().splitlines() if l.strip()]
    groups = defaultdict(list)
    for e in lines:
        groups[(e.get("bench", "1.0.0").split(".")[0], e.get("tier", "quick"))].append(e)
    out = [HEAD]
    for (major, tier), entries in sorted(groups.items(), key=lambda kv: (-int(kv[0][0]), kv[0][1])):
        w = entries[-1]["workload"]
        out += [f"## bench {major}.x, {tier} tier ({w['chunks']} chunks, {w['workers']} workers, groups of {w['group']})", "",
                "| date | bench | commit | previous | speed | wall (s), previous / new | peak RSS (GB), previous / new "
                "| device written (MB), previous / new | rounds | run |",
                "|---|---|---|---|---|---|---|---|---|---|"]
        for e in reversed(entries):
            sd = f" ± {e['speed_sd']:.2f}" if e.get("speed_sd") else ""
            out.append(
                f"| {e['date']} | {e.get('bench', '1.0.0')} | `{e['commit'][:7]}` | `{e['previous'][:7]}` "
                f"| {e['speed']:.2f}×{sd} ({e['wall_change_pct']:+.1f}%) "
                f"| {e['wall_previous']:.2f} / {e['wall_new']:.2f} "
                f"| {e['peak_sum_rss_mb_previous'] / 1000:.3f} / {e['peak_sum_rss_mb_new'] / 1000:.3f} "
                f"| {e['device_written_mb_previous']:,.0f} / {e['device_written_mb_new']:,.0f} "
                f"| {e.get('rounds', 1)} | [run]({e['run']}) |")
        out.append("")
    (store / "RECORDS.md").write_text("\n".join(out))


def main() -> None:
    store = Path(sys.argv[1])
    if sys.argv[2] == "--view":
        return view(store)
    url, files = sys.argv[2], [Path(p) for p in sys.argv[3:]]
    results = [json.loads(p.read_text()) for p in files]
    rounds = paired([r for res in results for r in rounds_of(res)])
    before, after = rounds[0]["names"]  # the commit before, the commit after
    fig = lambda a, k: [r["figures"][a][k] for r in rounds]
    mean = statistics.fmean  # over the rounds
    speeds = [b / a for b, a in zip(fig(before, "wall"), fig(after, "wall"))]
    sha = lambda a: rounds[0]["arms"][a]["sha"] or "?"
    tier = rounds[0]["workload"].get("tier", "quick")
    line = {
        "date": str(date.today()), "bench": VERSION, "tier": tier, "commit": sha(after), "previous": sha(before),
        "speed": round(mean(speeds), 4), "speed_sd": round(statistics.stdev(speeds), 4) if len(speeds) > 1 else None,
        "wall_change_pct": round(100 * (1 / mean(speeds) - 1), 2), "run": url, "rounds": len(rounds),
        "workload": rounds[0]["workload"], "parity_ok": all(r["parity_ok"] for r in rounds),
        **{f"{k}_{a}": mean(fig(arm, k)) for a, arm in (("previous", before), ("new", after))
           for k in ("wall", "cpu_total", "peak_sum_rss_mb", "device_written_mb")},
        "cpus": [r["cpu"] for r in rounds], "packages": rounds[0]["arms"][after]["packages"],
    }
    with open(store / "records.jsonl", "a") as f:
        f.write(json.dumps(line) + "\n")
    (store / "runs" / tier).mkdir(parents=True, exist_ok=True)
    (store / "runs" / tier / f"{sha(after)}.json").write_text(json.dumps(results))
    view(store)
    print(json.dumps(line))


if __name__ == "__main__":
    main()
