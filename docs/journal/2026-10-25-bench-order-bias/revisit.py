"""A commit's split CPU where it was the commit after, against where it was the commit
before in a later run (master's next run, or a pull request on it), by CPU model: what a
rise in its own run became when the same code was measured again.

Usage: python3 -I revisit.py SHA_PREFIX... -- RUN.json...
"""
import json
import statistics
import sys
from collections import defaultdict

i = sys.argv.index("--")
shas, paths = sys.argv[1:i], sys.argv[i + 1:]
K = "split_cpu_total"
for sha in shas:
    seen = defaultdict(lambda: defaultdict(list))  # (role, cpu) -> values
    other = defaultdict(list)
    for p in paths:
        jobs = json.load(open(p))
        if not isinstance(jobs, list) or not jobs or not jobs[0].get("order"):
            continue
        for role in ("after", "before"):
            if jobs[0]["arms"][role]["sha"].startswith(sha):
                for j in jobs:
                    cpu = j["machine"]["cpu"].split()[2]
                    for r in j["runs"]:
                        if r["arm"] == role and r.get(K):
                            seen[role][cpu].append(r[K])
                        if role == "after" and r["arm"] == "before" and r.get(K):
                            other[cpu].append(r[K])
    common = set(seen["after"]) & set(seen["before"])
    m = lambda v: f"{statistics.fmean(v):.1f} (n={len(v)})"
    print(f"{sha}: " + "; ".join(f"{c}: its run's before {m(other[c])}, it {m(seen['after'][c])} -> later as before "
                                 f"{m(seen['before'][c])}" for c in sorted(common)))
