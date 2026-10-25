"""Finalise's modelled release time by stage, from a bench run's artifacts.

Each finalise stage's own CPU (phases.jsonl `self`) on the run on three times the rows
less the first run (the part per byte the model scales), split into the claims (any call
inside the claims' compact_table or sort write_table) and the rest, then converted to
modelled release seconds at bench 13.0.0's rates: the claims at their bytes' scale over
2 (the 3x run adds two samples' rows) over 4 cores, the rest at the entities' scale.

Usage: python account.py ARTIFACTS_DIR [arm]   (gh run download RUN -D ARTIFACTS_DIR)
"""

import collections
import glob
import json
import statistics
import sys

root, arm = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "after")
runs = []
for path in sorted(glob.glob(f"{root}/*/{arm}_r1.phases.jsonl")):
    rows = [json.loads(line) for line in open(path)]
    res = json.load(open(path.replace(".phases.jsonl", ".json")))
    m = {r["marker"]: r["t"] for r in rows if "marker" in r}
    spans = {"1x": (m["finalise"], m["finalise end"]), "3x": (m["finalise x3"], m["finalise x3 end"])}
    acc = {}
    for k, (a, b) in spans.items():
        win = [(r["t0"], r["t"]) for r in rows
               if r.get("fn") in ("sort_by_id.write_table", "compact.compact_table")
               and (r.get("of") or [""])[0] == "claims" and a <= r["t0"] <= b]
        c = collections.Counter()
        for r in rows:
            if "fn" in r and a <= r["t0"] <= b:
                part = "claims" if any(s <= r["t0"] <= e for s, e in win) else "rest"
                c[(part, r["fn"])] += r["self"]
        acc[k] = c
    rate = {"claims": res["claims_bytes_scale"] / 2 / 4, "rest": res["scale"] / 2 / 4}
    eta = {key: (acc["3x"][key] - acc["1x"][key]) * rate[key[0]] for key in set(acc["1x"]) | set(acc["3x"])}
    eta[("model", "fixed (per key)")] = res["finalise_fixed"]
    eta[("model", "claims workers' starts")] = res["release_claims_workers_eta"]
    eta[("model", "finalise ETA")] = res["finalise_eta"]
    runs.append(eta)

total = statistics.mean(r[("model", "finalise ETA")] for r in runs)
keys = set().union(*runs)
print(f"{len(runs)} runs; finalise ETA {total / 60:.1f} min")
for key in sorted(keys, key=lambda k: (k[0], -statistics.mean(r.get(k, 0) for r in runs))):
    v = [r.get(key, 0) for r in runs]
    mu = statistics.mean(v)
    if abs(mu) >= 15:
        print(f"{key[0]:7s} {key[1]:40s} {mu / 60:6.1f} min ± {statistics.stdev(v) / 60:4.1f}  {100 * mu / total:5.1f}%")
for part in ("claims", "rest"):
    v = [sum(x for k, x in r.items() if k[0] == part) for r in runs]
    print(f"{part:7s} {'(all)':40s} {statistics.mean(v) / 60:6.1f} min  {100 * statistics.mean(v) / total:5.1f}%")
