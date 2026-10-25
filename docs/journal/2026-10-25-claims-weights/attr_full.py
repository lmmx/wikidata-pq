"""Per-chunk per-table pipeline CPU for runs where a worker takes several chunks: a
worker's calls split into its chunks at every third normalise_map_direct (labels,
descriptions, aliases: three a chunk), in the order its chunks finished."""
import json, sys, glob, os, collections
art, files, outp = sys.argv[1], sys.argv[2].split(","), sys.argv[3]
res = {}
for d in sorted(glob.glob(f"{art}/round-*")):
    for arm in ("before", "after"):
        for f in glob.glob(f"{d}/{arm}_r*.phases.jsonl"):
            base = f[:-len(".phases.jsonl")]
            if not os.path.exists(base + ".json"):
                continue
            rows = [json.loads(l) for l in open(f)]
            marks = {r["marker"]: r["t"] for r in rows if "marker" in r}
            a, b = marks["pipeline"], marks["pipeline end"]
            calls = [r for r in rows if "fn" in r and a <= r["t"] <= b]
            chunks = [json.loads(l) for l in open(base + ".chunks.jsonl")]
            bypid_chunks = collections.defaultdict(list)
            for c in chunks:
                bypid_chunks[c["pid"]].append(c["chunk"])
            out = collections.defaultdict(float)
            bypid = collections.defaultdict(list)
            for r in calls:
                bypid[r["pid"]].append(r)
            bad = 0
            for pid, cs in bypid.items():
                cs.sort(key=lambda r: r["t0"])
                if pid not in bypid_chunks:
                    for r in cs:
                        out[("parent", "merge" if r["fn"] == "push.core.merge_group" else "misc")] += r["self"]
                    continue
                starts = [r["t0"] for r in cs if r["fn"] == "process.normalise_map_direct"][::3]
                order = bypid_chunks[pid]
                if len(starts) != len(order):
                    bad += 1
                    continue
                claims_iv = [(r["t0"], r["t"]) for r in cs if r["fn"] in ("process.normalise_claims_direct",
                                                                         "partitioning.core.place_unsplit")]
                last_prep = prev = None
                for r in cs:
                    j = max(i for i, s in enumerate(starts) if s <= r["t0"]) if r["t0"] >= starts[0] else 0
                    fname = files[order[j] % len(files)]
                    fn = r["fn"]
                    if any(x0 <= r["t0"] and r["t"] <= x1 for x0, x1 in claims_iv):
                        t = "claims"
                    elif fn == "process.normalise_in_schema":
                        t = r["of"][0]
                    elif fn == "partitioning.transforms.prepare_for_partition":
                        t = last_prep = r["of"][0]
                    elif fn == "partitioning.core.partition_parquet":
                        t = last_prep
                    elif fn == "process.normalise_sitelinks":
                        t = "links"
                    elif fn == "process.atomic_sink_parquet":
                        t = "links" if prev == "process.normalise_sitelinks" else "entities"
                    else:
                        t = "misc"
                    out[(fname, t)] += r["self"]
                    out[(f"chunk{order[j]}", t)] += r["self"]
                    prev = fn
            run = json.load(open(base + ".json"))
            res[os.path.basename(d) + "/" + arm] = {"bad_pids": bad, "out": {"|".join(k): v for k, v in out.items()},
                "res": {k: run.get(k) for k in ("wall", "cpu_total", "chunk_cpu", "pipeline_eta", "scale", "n")},
                "chunk_cpu": [[c["chunk"], c["cpu"]] for c in chunks]}
            print(base, "bad pids", bad)
json.dump(res, open(outp, "w"))
