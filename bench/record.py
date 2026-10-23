"""Add a push to master's run to RECORDS.md's CI table, newest first: the new commit
against the previous one.

Usage: record.py <results.json> <RECORDS.md> <run url>
"""

import json
import statistics
import sys
from datetime import date
from pathlib import Path

HEADER = "| date | commit | previous |"


def main() -> None:
    res = json.loads(Path(sys.argv[1]).read_text())
    records, url = Path(sys.argv[2]), sys.argv[3]
    ref, cand = list(res["arms"])  # the previous commit, the new one
    vals = lambda a, k: [r[k] for r in res["runs"] if r["arm"] == a and r[k] is not None]
    med = lambda a, k: statistics.median(vals(a, k))
    rng = lambda a, k, f=lambda v: f"{v:,.0f}": "–".join(dict.fromkeys(f(v) for v in (min(vals(a, k)), max(vals(a, k)))))
    gb = lambda v: f"{v / 1024:.1f}"
    sha = lambda a: (res["arms"][a]["sha"] or "?")[:7]
    change = 100 * (med(cand, "wall") / med(ref, "wall") - 1)
    row = (
        f"| {date.today()} | `{sha(cand)}` | `{sha(ref)}` | {med(ref, 'wall') / med(cand, 'wall'):.2f}× ({change:+.1f}%) "
        f"| {med(ref, 'wall'):.1f} / {med(cand, 'wall'):.1f} "
        f"| {rng(ref, 'peak_sum_rss_mb', gb)} / {rng(cand, 'peak_sum_rss_mb', gb)} "
        f"| {rng(ref, 'device_written_mb')} / {rng(cand, 'device_written_mb')} | [run]({url}) |"
    )
    lines = records.read_text().splitlines()
    at = next(i for i, l in enumerate(lines) if l.startswith(HEADER)) + 2  # past the rule
    if lines[at].strip("| ") == "":  # the empty placeholder row
        del lines[at]
    lines.insert(at, row)
    records.write_text("\n".join(lines) + "\n")
    print(row)


if __name__ == "__main__":
    main()
