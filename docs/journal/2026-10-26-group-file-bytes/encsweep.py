"""Each claims leaf's bytes under each encoding family, the sample's 5 claims partition
files each written as the split writes it (pq_glue parallel == genson's writer)."""
import sys, time, os, json, resource, glob
from collections import defaultdict
import pyarrow.parquet as pq
import pq_glue
sys.path.insert(0, os.environ.get("SRC", "src"))
from wikidata.partitioning.core import ARROW_ROW_GROUP_ROWS
P = os.environ["WORK"]  # split.py's work dir (parts/{set}/claims/chunk_*.parquet)
files = sorted(glob.glob(f"{P}/parts/*/claims/chunk_*.parquet"))
tables = [pq.read_table(f) for f in files]
meta = pq.read_metadata(files[0])
leaves = {meta.schema.column(i).path: meta.schema.column(i).physical_type for i in range(meta.num_columns)}
FAM = {
  "base": {},
  "dba": {"BYTE_ARRAY": "DELTA_BYTE_ARRAY", "INT64": "DELTA_BINARY_PACKED", "INT32": "DELTA_BINARY_PACKED", "DOUBLE": "BYTE_STREAM_SPLIT"},
  "dlba": {"BYTE_ARRAY": "DELTA_LENGTH_BYTE_ARRAY", "INT64": "BYTE_STREAM_SPLIT", "INT32": "BYTE_STREAM_SPLIT", "DOUBLE": "ALP"},
  "plain": {"BYTE_ARRAY": "PLAIN", "INT64": "PLAIN", "INT32": "PLAIN", "DOUBLE": "PLAIN", "BOOLEAN": "PLAIN"},
}
level = int(os.environ.get("LEVEL", "1"))
res = {}
for name, fam in FAM.items():
    enc = {p: fam[t] for p, t in leaves.items() if t in fam}
    if name == "base": enc = {"statement_id": "DELTA_BYTE_ARRAY"}
    else: enc["statement_id"] = "DELTA_BYTE_ARRAY"
    sizes = defaultdict(int); total = 0; cpu = 0.0; wall = 0.0
    for i, t in enumerate(tables):
        dst = f"{P}/sw_{i}.parquet"
        r0 = resource.getrusage(resource.RUSAGE_SELF); w0 = time.perf_counter()
        pq_glue.write_parquet(t, dst, row_group_size=ARROW_ROW_GROUP_ROWS, compression_level=level, column_encoding=enc)
        r1 = resource.getrusage(resource.RUSAGE_SELF); wall += time.perf_counter() - w0
        cpu += r1.ru_utime + r1.ru_stime - r0.ru_utime - r0.ru_stime
        total += os.path.getsize(dst)
        m = pq.read_metadata(dst)
        for g in range(m.num_row_groups):
            rg = m.row_group(g)
            for c in range(rg.num_columns):
                sizes[rg.column(c).path_in_schema] += rg.column(c).total_compressed_size
        os.unlink(dst)
    res[name] = {"total": total, "cpu": round(cpu, 2), "wall": round(wall, 2), "cols": dict(sizes)}
    print(name, total, round(cpu, 2), round(wall, 2), flush=True)
json.dump({"leaves": leaves, "res": res}, open(f"{P}/encsweep_l{level}.json", "w"))
base = res["base"]["cols"]
best = {}
for p in leaves:
    opts = {n: res[n]["cols"][p] for n in res}
    b = min(opts, key=opts.get)
    best[p] = b
    if b != "base" and base[p] - opts[b] > 2000:
        print(f"{p:100s} {leaves[p]:10s} base={base[p]:9d} {b}={opts[b]:9d} ({100*(opts[b]-base[p])/base[p]:+.1f}%)  " + " ".join(f"{n}={v}" for n, v in opts.items()))
gain = sum(base[p] - res[best[p]]["cols"][p] for p in leaves)
print("total base", res["base"]["total"], "best per column saves", gain, f"{100*gain/res['base']['total']:.2f}%")
