import json, sys, glob, os, statistics as st
root = sys.argv[1]
keys = sys.argv[2].split(",")
rows = []
for d in sorted(glob.glob(os.path.join(root, "round-*"))):
    a = json.load(open(os.path.join(d, "after_r1.json")))
    b = json.load(open(os.path.join(d, "before_r1.json")))
    rows.append((os.path.basename(d), a, b))
print("key".ljust(28), "before".rjust(9), "after".rjust(9), "  d%  per round")
for k in keys:
    bs = [r[2].get(k) for r in rows]; as_ = [r[1].get(k) for r in rows]
    if None in bs or None in as_: print(k, "missing"); continue
    ds = [100*(a-b)/b if b else 0 for a, b in zip(as_, bs)]
    print(k.ljust(28), f"{st.mean(bs):9.2f}", f"{st.mean(as_):9.2f}", f"{st.mean(ds):+6.1f}  ", " ".join(f"{x:+.1f}" for x in ds))
