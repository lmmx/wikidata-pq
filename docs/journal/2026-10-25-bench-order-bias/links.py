"""The arm effect, from the ledger's links: a commit measured as the commit after in one
master run and as the commit before in the next (the same pipeline: src, reshape,
pyproject.toml and uv.lock the same trees; and the same bench major, the same workload).
Without an arm effect the two read alike, on the same CPU model, but for runner noise;
with the commit after reading high by b, the next run's before reads lower by b.

With --prs DIR (gather.sh's runs of pull requests), the links are also a pull request's
commit before (its merge base, a master commit) against that commit's own master run, where
it was the commit after, on the same bench version.

Usage: python3 -I links.py [--prs DIR] REPO RECORDS.jsonl RUNS_DIR [FIGURE...]
"""

import json
import math
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

T95 = [None, 12.71, 4.30, 3.18, 2.78, 2.57, 2.45, 2.36, 2.31, 2.26]


def trees(repo, sha):
    out = []
    for p in ("src", "reshape", "pyproject.toml", "uv.lock"):
        r = subprocess.run(["git", "-C", repo, "rev-parse", f"{sha}:{p}"], capture_output=True, text=True)
        out.append(r.stdout.strip())
    return tuple(out)


def levels(jobs, arm, k):
    """The arm's figure by CPU model, the mean over that model's jobs (log)."""
    by = defaultdict(list)
    for j in jobs:
        for r in j["runs"]:
            if r["arm"] == arm and r.get(k):
                by[j["machine"]["cpu"]].append(math.log(r[k]))
    return {c: statistics.fmean(v) for c, v in by.items()}


def main():
    argv = sys.argv[1:]
    prs = None
    if argv[0] == "--prs":
        prs, argv = Path(argv[1]), argv[2:]
    repo, records, runs = argv[0], argv[1], Path(argv[2])
    figures = argv[3:] or ["split_cpu_total", "finalise_cpu", "measured_cpu", "measured_wall", "cpu_total", "wall",
                               "eta", "pipeline_eta", "finalise_eta", "total_wall"]
    recs = []
    for line in open(records):
        e = json.loads(line)
        p = runs / e["tier"] / f"{e['commit']}.json"
        if e["tier"] != "quick" or not p.exists() or not isinstance(jobs := json.loads(p.read_text()), list):
            continue
        recs.append({**e, "jobs": jobs, "major": e["bench"].split(".")[0]})
    for e in recs:
        e["t_after"], e["t_before"] = trees(repo, e["commit"]), trees(repo, e["previous"])
    pairs = [(prev, cur) for prev, cur in zip(recs, recs[1:])
             if prev["major"] == cur["major"] and prev["t_after"] == cur["t_before"]]
    if prs:
        by_commit = {e["commit"]: e for e in recs}
        for p in sorted(prs.glob("*.json")):
            jobs = [j for j in json.loads(p.read_text()) if j.get("order")]
            if not jobs or not (m := by_commit.get(jobs[0]["arms"]["before"]["sha"])) or m["bench"] != jobs[0]["bench"]:
                continue
            pairs.append((m, {"jobs": jobs}))
    print(f"{len(pairs)} links\n")
    print("| figure | links | next run's before over this run's after (same commit, same CPU model) "
          "| the earlier run's paired change, mean |")
    print("|---|---|---|---|")
    for k in figures:
        gaps, changes = [], []
        for prev, cur in pairs:
            a, b = levels(prev["jobs"], "after", k), levels(cur["jobs"], "before", k)
            common = set(a) & set(b)
            if not common:
                continue
            gaps.append(statistics.fmean(b[c] - a[c] for c in common))
            # (the run's own paired change, as combine reports it)
            lr = [math.log(r1[k] / r0[k]) for j in prev["jobs"]
                  for r0 in j["runs"] if r0["arm"] == "before" and r0.get(k)
                  for r1 in j["runs"] if r1["arm"] == "after" and r1.get(k)]
            changes.append(statistics.fmean(lr))
        if len(gaps) < 2:
            continue
        n = len(gaps)
        h = T95[min(n, len(T95)) - 1] * statistics.stdev(gaps) / n**0.5
        hc = T95[min(n, len(T95)) - 1] * statistics.stdev(changes) / n**0.5
        print(f"| `{k}` | {n} | {100 * statistics.fmean(gaps):+.2f}% ±{h * 100:.2f} | "
              f"{100 * statistics.fmean(changes):+.2f}% ±{hc * 100:.2f} |")


if __name__ == "__main__":
    main()
