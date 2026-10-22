"""Per-step wall and CPU across the runs of profile_process.py outputs (warm runs: all but each process's run 0)."""
import json, statistics as st, sys
from collections import defaultdict
w, c, pk, tot = defaultdict(list), defaultdict(list), defaultdict(float), []
for f in sys.argv[1:]:
    d = json.load(open(f))
    for r in d["runs"]:
        if r["run"] == 0:
            continue
        tot.append((r["wall"], r["cpu"]))
        for k, v in r["steps"].items():
            w[k].append(v["wall"]); c[k].append(v["cpu"]); pk[k] = max(pk[k], v["peak_over_base_mb"])
print(f"warm runs {len(tot)}: wall {st.median(x[0] for x in tot):.2f} ({min(x[0] for x in tot):.2f}-{max(x[0] for x in tot):.2f}) cpu {st.median(x[1] for x in tot):.2f}")
for k in sorted(w, key=lambda k: -st.median(w[k])):
    print(f"{st.median(w[k]):6.3f} ({min(w[k]):.3f}-{max(w[k]):.3f}) wall  {st.median(c[k]):6.2f} cpu  peak+{pk[k]:6.0f} MB  {k}")
