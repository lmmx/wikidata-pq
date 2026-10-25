import json, sys, glob, os, collections
root, span = sys.argv[1], sys.argv[2]
def load(p):
    rows = [json.loads(l) for l in open(p)]
    m = {r["marker"]: r["t"] for r in rows if "marker" in r}
    a, b = m[span], m[span + " end"]
    procs = {}
    for r in rows:
        if "proc" in r and a <= r["t"] <= b:
            procs.setdefault(r["pid"], {})[r["proc"]] = r["cpu"]
    tot = sum(v.get("exit", v.get("end", 0)) - v.get("start", 0) for v in procs.values())
    traced = sum(r["self"] for r in rows if "fn" in r and a <= r["t"] <= b)
    kinds = collections.Counter(k for v in procs.values() for k in v)
    return len(procs), round(tot, 2), round(traced, 2), kinds
for d in sorted(glob.glob(os.path.join(root, "round-*"))):
    b = load(f"{d}/before_r1.phases.jsonl"); a = load(f"{d}/after_r1.phases.jsonl")
    jb = json.load(open(f"{d}/before_r1.json")); ja = json.load(open(f"{d}/after_r1.json"))
    print(os.path.basename(d), "procs", b[0], a[0], "proc cpu", b[1], a[1], "traced", b[2], a[2], "finalise_cpu", jb["finalise_cpu"], ja["finalise_cpu"], "split", jb["split_cpu_total"], ja["split_cpu_total"], b[3])
