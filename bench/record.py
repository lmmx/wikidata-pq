"""Append a push to master's run to the results store (the `bench-results` branch): a row
at the end of RECORDS.md, a line at the end of records.jsonl, and the rounds' raw results
as runs/<commit>.json. Nothing already in the store is changed.

Usage: record.py <store dir> <run url> <results.json>...  (one per round)
"""

import json
import statistics
import sys
from datetime import date
from pathlib import Path

from combine import rounds_of


def main() -> None:
    store, url, files = Path(sys.argv[1]), sys.argv[2], [Path(p) for p in sys.argv[3:]]
    store.mkdir(parents=True, exist_ok=True)
    if not (store / "RECORDS.md").exists():  # a new tier's ledger
        (store / "RECORDS.md").write_text(
            "| date | commit | previous | speed | mean wall (s), previous / new | peak RSS (GB), previous / new "
            "| device written (MB), previous / new | run |\n|---|---|---|---|---|---|---|---|\n")
    results = [json.loads(p.read_text()) for p in files]
    rounds = [r for res in results for r in rounds_of(res)]
    before, after = rounds[0]["names"]  # the previous commit, the new one
    fig = lambda a, k: [r["figures"][a][k] for r in rounds]
    med = statistics.fmean  # the mean over the rounds
    speeds = [b / a for b, a in zip(fig(before, "wall"), fig(after, "wall"))]
    speed = med(speeds)
    sha = lambda a: rounds[0]["arms"][a]["sha"] or "?"
    gb = lambda v: f"{v / 1000:.3f}"
    row = (
        f"| {date.today()} | `{sha(after)[:7]}` | `{sha(before)[:7]}` | {speed:.2f}× ({100 * (1 / speed - 1):+.1f}%) "
        f"| {med(fig(before, 'wall')):.2f} / {med(fig(after, 'wall')):.2f} "
        f"| {gb(med(fig(before, 'peak_sum_rss_mb')))} / {gb(med(fig(after, 'peak_sum_rss_mb')))} "
        f"| {med(fig(before, 'device_written_mb')):,.0f} / {med(fig(after, 'device_written_mb')):,.0f} | [run]({url}) |"
    )
    line = {
        "date": str(date.today()), "commit": sha(after), "previous": sha(before),
        "speed": round(speed, 4), "speed_sd": round(statistics.stdev(speeds), 4) if len(speeds) > 1 else None,
        "wall_change_pct": round(100 * (1 / speed - 1), 2), "run": url, "rounds": len(rounds),
        "workload": rounds[0]["workload"], "parity_ok": all(r["parity_ok"] for r in rounds),
        **{f"{k}_{a}": med(fig(arm, k)) for a, arm in (("previous", before), ("new", after))
           for k in ("wall", "cpu_total", "peak_sum_rss_mb", "device_written_mb")},
        "cpus": [r["cpu"] for r in rounds], "packages": rounds[0]["arms"][after]["packages"],
    }
    with open(store / "RECORDS.md", "a") as f:
        f.write(row + "\n")
    with open(store / "records.jsonl", "a") as f:
        f.write(json.dumps(line) + "\n")
    (store / "runs").mkdir(exist_ok=True)
    (store / "runs" / f"{sha(after)}.json").write_text(json.dumps(results))
    print(row)


if __name__ == "__main__":
    main()
