"""Per-span, per-fn self CPU: before vs after, mean over rounds, CI from round means
(forward/reverse of a round averaged, as the bench pairs them)."""
import json, sys, glob, os, collections, statistics as st
root, span = sys.argv[1], sys.argv[2]
top = int(sys.argv[3]) if len(sys.argv) > 3 else 25
T = {2: 12.71, 3: 4.30, 4: 3.18}
def load(p):
    rows = [json.loads(l) for l in open(p)]
    m = {r["marker"]: r["t"] for r in rows if "marker" in r}
    a, b = m[span], m[span + " end"]
    c = collections.Counter()
    for r in rows:
        if "fn" in r and a <= r["t"] <= b:
            c[r["fn"]] += r["self"]
    return c
runs = {}
for d in sorted(glob.glob(os.path.join(root, "round-*"))):
    runs[os.path.basename(d)] = (load(f"{d}/before_r1.phases.jsonl"), load(f"{d}/after_r1.phases.jsonl"))
fns = collections.Counter()
for b, a in runs.values():
    fns.update(b); fns.update(a)
tot = {k: (sum(b.values()), sum(a.values())) for k, (b, a) in runs.items()}
def row(name, get):
    bs = [get(b) for b, a in runs.values()]; as_ = [get(a) for b, a in runs.values()]
    d = [a - b for a, b in zip(as_, bs)]
    rounds = collections.defaultdict(list)
    for k, x in zip(runs, d):
        rounds[k.split("-")[1]].append(x)
    rm = [st.mean(v) for v in rounds.values()]
    e = T[len(rm)] * st.stdev(rm) / len(rm) ** 0.5 if len(rm) > 1 else 0
    mb = st.mean(bs)
    print(f"{name[:46]:46} {mb:8.3f} {st.mean(as_):8.3f} {st.mean(d):+7.3f} ±{e:.3f}  ({100*st.mean(d)/mb if mb else 0:+.1f}%)")
print(f"{'fn':46} {'before':>8} {'after':>8} {'delta':>7}")
row("TOTAL " + span, lambda c: sum(c.values()))
for f, _ in fns.most_common(top):
    row(f, lambda c, f=f: c.get(f, 0.0))
