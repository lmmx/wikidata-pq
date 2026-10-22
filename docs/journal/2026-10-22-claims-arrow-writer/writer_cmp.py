import os, resource, sys, time
from pathlib import Path
os.environ.setdefault("WIKIDATA_RELEASE", "20261012")
sys.path.insert(0, sys.argv[1])
import polars as pl, pyarrow.parquet as pq
from wikidata.partitioning.claims import claims_base
def cpu():
    r = resource.getrusage(resource.RUSAGE_SELF); return r.ru_utime + r.ru_stime
df = claims_base(pl.scan_parquet(sys.argv[2]).drop_nulls()).collect()
tbl = df.to_arrow()
for r in range(2):
    for name, f in (("polars write_parquet", lambda: df.write_parquet("/tmp/scratch/cmp/w/wc_pl.parquet")),
                    ("pyarrow write_table zstd", lambda: pq.write_table(tbl, "/tmp/scratch/cmp/w/wc_pa.parquet", compression="zstd"))):
        t, c = time.perf_counter(), cpu(); f(); print(f"  {name:26s} {time.perf_counter()-t:5.2f}s wall {cpu()-c:5.2f}s cpu", flush=True)
