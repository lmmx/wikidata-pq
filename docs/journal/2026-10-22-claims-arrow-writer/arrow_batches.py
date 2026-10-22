"""pyarrow's ParquetWriter on the claims frame, fed in batches of n rows: row groups made,
wall, CPU, RSS peak and after, file size. Usage: arrow_batches.py <src> <claims.parquet> <n> [rg rows]"""
import os, resource, sys, time
from pathlib import Path
os.environ.setdefault("WIKIDATA_RELEASE", "20261012")
sys.path.insert(0, sys.argv[1])
import polars as pl, pyarrow.parquet as pq
from wikidata.config import UNSPLIT_COL, Table
from wikidata.partitioning import prepare_for_partition
def kb(f): return next(int(l.split()[1]) for l in open("/proc/self/status") if l.startswith(f)) // 1024
def cpu():
    r = resource.getrusage(resource.RUSAGE_SELF); return r.ru_utime + r.ru_stime
n = int(sys.argv[3]); rg = int(sys.argv[4]) if len(sys.argv) > 4 else 122_880
f = "/tmp/scratch/cmp/w/ab.parquet"
for r in range(2):
    df = prepare_for_partition(Path(sys.argv[2]).resolve(), Table.CLAIMS).collect().drop(UNSPLIT_COL)
    open("/proc/self/clear_refs", "w").write("5"); r0 = kb("VmRSS"); t, c = time.perf_counter(), cpu()
    with pq.ParquetWriter(f, df.head(0).to_arrow().schema, compression="zstd") as w:
        for part in df.iter_slices(n):
            for b in part.to_arrow().to_batches():
                w.write_batch(b, row_group_size=rg)
    m = pq.read_metadata(f)
    print(f"batch {n:6d}: {time.perf_counter()-t:5.2f}s wall {cpu()-c:5.2f}s cpu  RSS {r0} -> peak {kb('VmHWM')} -> {kb('VmRSS')} MB  {os.path.getsize(f)/2**20:.2f} MB  rgs {[m.row_group(i).num_rows for i in range(m.num_row_groups)][:8]}", flush=True)
    del df
