"""Append a push to master's run to the results store (the `bench-results` branch): a row
at the end of RECORDS.md, a line at the end of records.jsonl, and the run's raw results
as runs/<commit>.json. Nothing already in the store is changed.

Usage: record.py <results.json> <store dir> <run url>
"""

import json
import statistics
import sys
from datetime import date
from pathlib import Path


def main() -> None:
    results, store, url = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    res = json.loads(results.read_text())
    before, after = list(res["arms"])  # the previous commit, the new one
    vals = lambda a, k: [r[k] for r in res["runs"] if r["arm"] == a and r[k] is not None]
    med = lambda a, k: statistics.median(vals(a, k))
    rng = lambda a, k, f=lambda v: f"{v:,.0f}": "–".join(dict.fromkeys(f(v) for v in (min(vals(a, k)), max(vals(a, k)))))
    gb = lambda v: f"{v / 1024:.1f}"
    sha = lambda a: res["arms"][a]["sha"] or "?"
    speed = med(before, "wall") / med(after, "wall")
    change = 100 * (med(after, "wall") / med(before, "wall") - 1)
    row = (
        f"| {date.today()} | `{sha(after)[:7]}` | `{sha(before)[:7]}` | {speed:.2f}× ({change:+.1f}%) "
        f"| {med(before, 'wall'):.1f} / {med(after, 'wall'):.1f} "
        f"| {rng(before, 'peak_sum_rss_mb', gb)} / {rng(after, 'peak_sum_rss_mb', gb)} "
        f"| {rng(before, 'device_written_mb')} / {rng(after, 'device_written_mb')} | [run]({url}) |"
    )
    line = {
        "date": str(date.today()), "commit": sha(after), "previous": sha(before),
        "speed": round(speed, 4), "wall_change_pct": round(change, 2), "run": url,
        "workload": res["workload"], "rounds": res["rounds"], "parity_ok": res["parity_ok"],
        **{f"{k}_{a}": med(arm, k) for a, arm in (("previous", before), ("new", after))
           for k in ("wall", "cpu_total", "peak_sum_rss_mb", "device_written_mb")},
        "cpu": res["machine"]["cpu"], "packages": res["arms"][after]["packages"],
    }
    with open(store / "RECORDS.md", "a") as f:
        f.write(row + "\n")
    with open(store / "records.jsonl", "a") as f:
        f.write(json.dumps(line) + "\n")
    (store / "runs").mkdir(exist_ok=True)
    (store / "runs" / f"{sha(after)}.json").write_text(results.read_text())
    print(row)


if __name__ == "__main__":
    main()
