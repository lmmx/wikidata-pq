"""A release's modelled time by stage under bench 10.0.0's model, from the bench's own
artifacts: each run's results JSON (`<arm>_r<k>.json`), phases log, chunks log and
split-from-gz JSON. Each stage's own CPU is scaled as bench/pipeline.py scales it:

- the pipeline: wall/CPU times the stage's CPU, the claims' calls (inside
  normalise_claims_direct or place_unsplit, by their chunk's set) by the set's claims
  rows, the rest (and the CPU outside the timed stages) by entities;
- finalise: the 3x run less the first by stage and table (phases.table_cpu's rule),
  halved, over the cores, the claims by the claims' bytes and the rest by entities; the
  per-byte CPU outside the stages at the blended scale; the fixed part and the release's
  claims workers' starts as the harness has them;
- the split: its CPU by the dump's bytes over the slice's over the cores, lbzip2's part
  its decode of the slice alone.

Prints, per run, the parts' sums against the harness's own ETAs.

Usage: rerank.py <out.json> <artifact dirs (round-*)...>"""
import collections, json, sys, glob, os, statistics as st

RELEASE_DUMP_BYTES = 103_488_697_311
FILES = ["c0", "r50", "s30", "s90"]  # quick tier, chunk i -> FILES[i]
SET = {f: ("scholar" if f.startswith("s") else "main") for f in FILES}
CLAIMS_CALLS = ("process.normalise_claims_direct", "partitioning.core.place_unsplit")
TABLES = {"claims", "labels", "descriptions", "aliases", "links", "entities", "claims_labels"}

def table_of(e):
    for a in e.get("of", []):
        t = a.split("/")[0].split(" ")[0]
        if t in TABLES:
            return t
    return None

def analyse(js, ph, chunks):
    R = json.load(open(js))
    rows = [json.loads(l) for l in open(ph)]
    marks = {r["marker"]: r["t"] for r in rows if "marker" in r}
    calls = [r for r in rows if "fn" in r]
    out = collections.Counter()  # (span, stage, class) -> release seconds
    cpu = collections.Counter()  # (span, stage, class) -> sample CPU-s
    # ---- pipeline
    a, b = marks["pipeline"], marks["pipeline end"]
    pc = [r for r in calls if a <= r["t"] <= b]
    order = collections.defaultdict(list)
    for c in [json.loads(l) for l in open(chunks)]:
        order[c["pid"]].append(c["chunk"])
    by_pid = collections.defaultdict(list)
    for r in pc:
        by_pid[r["pid"]].append(r)
    seen = collections.Counter()
    for pid, cs in by_pid.items():
        claims = sorted([r for r in cs if r["fn"] in CLAIMS_CALLS], key=lambda r: r["t"])
        spans = []
        for r in claims:
            k = seen[pid, r["fn"]]; seen[pid, r["fn"]] += 1
            chunk = order[pid][k]
            spans.append((r["t0"], r["t"], SET[FILES[chunk % 4]]))
        for r in cs:
            cls = next((s for t0, t1, s in spans if t0 <= r["t0"] and r["t"] <= t1), "other")
            cpu["pipeline", r["fn"], cls] += r["self"]
    stage_sum = sum(v for k, v in cpu.items() if k[0] == "pipeline")
    cpu["pipeline", "(outside the timed stages: worker starts, imports, the pool)", "other"] += R["cpu_total"] - stage_sum
    f = R["wall"] / R["cpu_total"]
    sc = {"main": R["claims_rows_scale_main"], "scholar": R["claims_rows_scale_scholar"], "other": R["scale"]}
    for k, v in list(cpu.items()):
        if k[0] == "pipeline":
            out[k] += v * f * sc[k[2]]
    # ---- finalise
    def tcpu(a, b):
        c = collections.Counter()
        bp = collections.defaultdict(list)
        for r in calls:
            if a <= r["t"] <= b:
                bp[r["pid"]].append(r)
        for cs in bp.values():
            cs.sort(key=lambda r: (r["t0"], -r["t"]))
            open_named = []
            for r in cs:
                open_named = [e for e in open_named if e["t"] >= r["t0"]]
                t = table_of(r)
                if t is None:
                    t = next((table_of(e) for e in reversed(open_named) if r["t"] <= e["t"]), "other")
                else:
                    open_named.append(r)
                c[r["fn"], t] += r["self"]
        return c
    one = tcpu(marks["finalise"], marks["finalise end"])
    three = tcpu(marks["finalise x3"], marks["finalise x3 end"])
    d = {k: three.get(k, 0) - one.get(k, 0) for k in set(one) | set(three)}
    D = sum(d.values())
    Dc = sum(v for k, v in d.items() if k[1] == "claims")
    share = Dc / D
    source = R["finalise_source_mb"] * 1e6
    per_byte = R["finalise_cpu_per_mb"] / 1e6
    P = per_byte * 2 * source
    w = R["workers"]
    cs_, es = R["claims_bytes_scale"], R["scale"]
    blend = share * cs_ + (1 - share) * es
    for (fn, t), v in d.items():
        cls = "claims" if t == "claims" else "other"
        cpu["finalise", fn, cls] += v
        out["finalise", fn, cls] += v / 2 * (cs_ if cls == "claims" else es) / w
        cpu["finalise_by_table", fn, t] += v
        out["finalise_by_table", fn, t] += v / 2 * (cs_ if cls == "claims" else es) / w
    cpu["finalise", "(per byte, outside the timed stages)", "blend"] += P - D
    out["finalise", "(per byte, outside the timed stages)", "blend"] += (P - D) / 2 * blend / w
    out["finalise", "(fixed part, per key)", "fixed"] += R["finalise_fixed"]
    out["finalise", "(the release's claims workers' starts)", "fixed"] += R["release_claims_workers_eta"]
    # ---- split
    a, b = marks["split"], marks["route end"] if "route end" in marks else marks["split end"]
    s = collections.Counter()
    for r in calls:
        if a <= r["t"] <= b and r["fn"].startswith("dump."):
            s[r["fn"]] += r["self"]
    hz = RELEASE_DUMP_BYTES / (R["split_source_mb"] * 1e6) / w
    gz = json.load(open(js.replace(".json", ".split-gz.json")))
    dec = gz.get("bz2_decode_cpu", 0)
    tot = R["split_cpu_total"]
    parts = {"lbzip2 decode (measured alone)": dec, **{k: v for k, v in s.items()}}
    parts["dump.route (route-release)"] = R["route_cpu"]
    parts["(the rest: Polars' write threads (zstd), line reads, is_scholarly, start-up)"] = tot - sum(parts.values())
    for k, v in parts.items():
        cpu["split", k, "dump bytes"] += v
        out["split", k, "dump bytes"] += v * hz
    checks = {
        "pipeline": (sum(v for k, v in out.items() if k[0] == "pipeline"), R["pipeline_eta"]),
        "finalise": (sum(v for k, v in out.items() if k[0] == "finalise"), R["finalise_eta"]),
        "split": (sum(v for k, v in out.items() if k[0] == "split"), R["split_eta"]),
        "eta": (sum(v for k, v in out.items() if k[0] != "finalise_by_table"), R["eta"]),
        "share": (share, R.get("finalise_claims_share")),
    }
    return out, cpu, checks, R

res = []
for d in sys.argv[2:]:
    for js in sorted(glob.glob(os.path.join(d, "*_r[0-9].json"))):
        ph = js.replace(".json", ".phases.jsonl"); ch = js.replace(".json", ".chunks.jsonl")
        out, cpu, checks, R = analyse(js, ph, ch)
        res.append({"run": js, "cpu_model": R["cpu_total"], "out": {"|".join(k): v for k, v in out.items()},
                    "cpu": {"|".join(k): v for k, v in cpu.items()}, "checks": checks,
                    "eta": R["eta"], "pipeline_eta": R["pipeline_eta"], "finalise_eta": R["finalise_eta"], "split_eta": R["split_eta"]})
json.dump(res, open(sys.argv[1], "w"))
for r in res:
    print(os.path.basename(os.path.dirname(r["run"])), os.path.basename(r["run"]), {k: (round(a), b) if k != "share" else (round(a, 4), b) for k, (a, b) in r["checks"].items()})
