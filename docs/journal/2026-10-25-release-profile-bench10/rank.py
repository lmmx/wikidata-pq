"""The release's modelled hours by stage (bench 10.0.0), from rerank.py's outputs: the
pipeline's and finalise's from every run given with --pf, the split's from the runs of
--split whose arm is --arm (the split's code differs between #127's arms; nothing else
does). Means over the runs, with a 95% interval over the runners.

Usage: rank.py --pf res128.json res127.json --split res127.json --arm after"""
import argparse, collections, json, math, statistics as st

p = argparse.ArgumentParser()
p.add_argument("--pf", nargs="+", required=True)
p.add_argument("--split", required=True)
p.add_argument("--arm", default="after")
a = p.parse_args()
pf = [r for f in a.pf for r in json.load(open(f))]
split = [r for r in json.load(open(a.split)) if f"/{a.arm}_" in r["run"]]


def hm(s):
    s = round(s)
    return f"{s // 3600}h{(s % 3600) // 60:02d}m" if s >= 3600 else f"{s // 60}m"


def agg(runs, span):
    keys = {k for r in runs for k in r["out"] if k.startswith(span + "|")}
    return {k: (st.mean(v := [r["out"].get(k, 0) for r in runs]), 1.96 * st.stdev(v) / math.sqrt(len(v)),
                st.mean(r["cpu"].get(k, 0) for r in runs)) for k in keys}


rows = {**agg(pf, "pipeline"), **agg(pf, "finalise"), **agg(split, "split")}
m = collections.defaultdict(lambda: [0.0, 0.0, 0.0])
for k, (v, ci, c) in rows.items():
    span, fn, cls = k.split("|")
    e = m[span, fn, "claims" if cls in ("main", "scholar", "claims") else cls]
    e[0] += v; e[1] = math.hypot(e[1], ci); e[2] += c
total = sum(e[0] for e in m.values())
print(f"runs: pipeline and finalise {len(pf)}, split {len(split)}; modelled release {hm(total)}")
for span, runs in (("pipeline", pf), ("finalise", pf), ("split", split)):
    v = [r[span + "_eta"] for r in runs]
    print(f"  {span}: {hm(st.mean(v))} ± {hm(1.96 * st.stdev(v) / math.sqrt(len(v)))}")
cum = 0.0
for i, ((span, fn, cls), (v, ci, c)) in enumerate(sorted(m.items(), key=lambda kv: -kv[1][0]), 1):
    cum += v
    print(f"{i:2d} {span:8s} {fn:60.60s} {cls:10s} {hm(v):>6s} ±{hm(ci):>4s} {100 * v / total:5.1f}%"
          f" (cum {100 * cum / total:5.1f}%) sample CPU {c:5.2f} s")
