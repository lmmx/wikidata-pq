"""Summarise WIKIDATA_PHASES lines: each phase's own CPU time (exclusive of the listed
phases it calls) in the 1x finalise run and the 3x run, and the difference, the per-byte
part the ETA scales."""
import collections
import json
import sys

rows = [json.loads(line) for line in open(sys.argv[1])]
marks = {r["marker"]: r["t"] for r in rows if "marker" in r}
runs = {"x1": (marks["x1"], marks["x1 end"]), "x3": (marks["x3"], marks["x3 end"])}
pipe = (marks["pipeline"], marks["pipeline end"])
pc = collections.Counter()
for r in rows:
    if "marker" not in r and pipe[0] <= r["t"] <= pipe[1]:
        pc[r["fn"]] += r["self"]
ptot = sum(pc.values())
print(f"pipeline, instrumented CPU (all workers): {ptot:.1f}s")
for fn, v in pc.most_common():
    print(f"  {fn:28s} {v:7.1f}s {100 * v / ptot:5.1f}%")
cpu = {k: collections.Counter() for k in runs}
calls = {k: collections.Counter() for k in runs}
for r in rows:
    if "marker" in r:
        continue
    for k, (a, b) in runs.items():
        if a <= r["t"] <= b:
            cpu[k][r["fn"]] += r["self"]
            calls[k][r["fn"]] += 1
tot = {k: sum(c.values()) for k, c in cpu.items()}
print(f"instrumented CPU: x1 {tot['x1']:.1f}s, x3 {tot['x3']:.1f}s, difference {tot['x3'] - tot['x1']:.1f}s")
print(f"{'phase (own CPU)':28s} {'x1':>8s} {'x3':>8s} {'x3-x1':>8s} {'share':>6s}  calls x1/x3")
d = tot["x3"] - tot["x1"]
for fn in sorted(set(cpu["x1"]) | set(cpu["x3"]), key=lambda f: -(cpu["x3"][f] - cpu["x1"][f])):
    a, b = cpu["x1"][fn], cpu["x3"][fn]
    print(f"{fn:28s} {a:8.1f} {b:8.1f} {b - a:8.1f} {100 * (b - a) / d:5.1f}%  {calls['x1'][fn]}/{calls['x3'][fn]}")
