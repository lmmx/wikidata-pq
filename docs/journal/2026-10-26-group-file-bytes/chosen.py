"""Base (#184: statement_id DBA) against a chosen per-column map: bytes by set and column,
write CPU, and read CPU by each of finalise's readers; values checked equal."""
import sys, time, os, json, resource, glob
from collections import defaultdict
import pyarrow.parquet as pq, pyarrow as pa, polars as pl
import pq_glue
sys.path.insert(0, os.environ.get("SRC", "src"))
from wikidata.partitioning.core import ARROW_ROW_GROUP_ROWS
P = os.environ["WORK"]  # split.py's work dir (parts/{set}/claims/chunk_*.parquet)
files = sorted(glob.glob(f"{P}/parts/*/claims/chunk_*.parquet"))
tables = [pq.read_table(f) for f in files]
CHOSEN = json.loads(os.environ["CHOSEN"]) if "CHOSEN" in os.environ else {
    "statement_id": "DELTA_BYTE_ARRAY",
    "id": "DELTA_LENGTH_BYTE_ARRAY",
    "datavalue.id": "DELTA_BYTE_ARRAY",
    "datavalue.datavalue__string": "DELTA_LENGTH_BYTE_ARRAY",
    "datavalue.text": "DELTA_LENGTH_BYTE_ARRAY",
    "datavalue.amount": "DELTA_BYTE_ARRAY",
    "datavalue.numeric-id": "BYTE_STREAM_SPLIT",
    "references.list.element.hash": "DELTA_LENGTH_BYTE_ARRAY",
    "qualifiers.list.element.value.list.element.datavalue.datavalue__string": "DELTA_LENGTH_BYTE_ARRAY",
    "qualifiers.list.element.value.list.element.datavalue.numeric-id": "PLAIN",
}
level = int(os.environ.get("LEVEL", "1"))
cfgs = {"base": {"statement_id": "DELTA_BYTE_ARRAY"}, "chosen": CHOSEN}
def cpu(): r = resource.getrusage(resource.RUSAGE_SELF); return r.ru_utime + r.ru_stime
out = {}
for name, enc in cfgs.items():
    bset = defaultdict(int); cols = defaultdict(lambda: defaultdict(int)); wcpu = []
    paths = []
    for i, (f, t) in enumerate(zip(files, tables)):
        dst = f"{P}/{name}_{i}.parquet"; paths.append(dst)
        s = "scholar" if "/scholar/" in f else "main"
        best = 1e9
        for _ in range(3):
            c0 = cpu(); pq_glue.write_parquet(t, dst, row_group_size=ARROW_ROW_GROUP_ROWS, compression_level=level, column_encoding=enc); best = min(best, cpu() - c0)
        wcpu.append(best)
        bset[s] += os.path.getsize(dst)
        m = pq.read_metadata(dst)
        for g in range(m.num_row_groups):
            rg = m.row_group(g)
            for c in range(rg.num_columns):
                cols[s][rg.column(c).path_in_schema] += rg.column(c).total_compressed_size
    reads = {}
    for rname, fn in {
        "pyarrow": lambda p: pq.read_table(p),
        "pyarrow_rowgroups": lambda p: [pq.ParquetFile(p).read_row_group(g) for g in range(pq.ParquetFile(p).num_row_groups)],
        "pq_glue": lambda p: pq_glue.read_parquet(p),
        "polars": lambda p: pl.read_parquet(p),
        "polars_id": lambda p: pl.scan_parquet(p).select("id").collect(),
    }.items():
        best = 1e9
        for _ in range(3):
            c0 = cpu()
            for p in paths: fn(p)
            best = min(best, cpu() - c0)
        reads[rname] = round(best, 3)
    for t, p in zip(tables, paths):
        assert pq.read_table(p).equals(t), p
        assert pl.read_parquet(p).equals(pl.from_arrow(t)), p
    out[name] = {"bytes": dict(bset), "total": sum(bset.values()), "write_cpu": round(sum(wcpu), 3), "read_cpu": reads,
                 "cols": {s: dict(v) for s, v in cols.items()}}
    print(name, out[name]["bytes"], out[name]["total"], "write", out[name]["write_cpu"], "read", reads, flush=True)
b, c = out["base"], out["chosen"]
print(f"total {c['total']-b['total']:+d} ({100*(c['total']-b['total'])/b['total']:+.2f}%)", {s: f"{100*(c['bytes'][s]-b['bytes'][s])/b['bytes'][s]:+.2f}%" for s in b['bytes']})
for col in CHOSEN:
    print(f"  {col:80s}", {s: f"{100*(c['cols'][s][col]-b['cols'][s][col])/b['cols'][s][col]:+.1f}% ({c['cols'][s][col]-b['cols'][s][col]:+d})" for s in b['cols']})
json.dump(out, open(f"{P}/chosen_l{level}.json", "w"))
