"""The claims partition write alone, Polars' sink or pyarrow (partition_parquet arrow=),
on a claims file in this process, 3 times: wall, CPU, RSS before/peak/after, and pyarrow's
pool (allocated now, max). Usage: arrow_mem.py <src> <claims.parquet> <arrow 0|1>"""
import os, resource, shutil, sys, time
from pathlib import Path
os.environ.setdefault("WIKIDATA_RELEASE", "20261012")
sys.path.insert(0, sys.argv[1])
import polars as pl, pyarrow as pa
from wikidata.config import UNSPLIT_COL, Table
from wikidata.partitioning import prepare_for_partition, partition_parquet
def kb(f): return next(int(l.split()[1]) for l in open("/proc/self/status") if l.startswith(f)) // 1024
def cpu():
    r = resource.getrusage(resource.RUSAGE_SELF); return r.ru_utime + r.ru_stime
arrow = sys.argv[3] == "1"; W = Path("/tmp/scratch/cmp/w/amem")
for r in range(3):
    shutil.rmtree(W, ignore_errors=True)
    df = prepare_for_partition(Path(sys.argv[2]).resolve(), Table.CLAIMS).collect()
    open("/proc/self/clear_refs", "w").write("5"); r0 = kb("VmRSS"); t, c = time.perf_counter(), cpu()
    partition_parquet(by=UNSPLIT_COL, lf=df.lazy(), source_name="chunk_0.parquet", dst_dir=W, log_dir=W / "audit", arrow=arrow)
    pool = pa.default_memory_pool()
    print(f"arrow={arrow} {r}: {time.perf_counter()-t:5.2f}s wall {cpu()-c:5.2f}s cpu  RSS {r0} -> peak {kb('VmHWM')} -> {kb('VmRSS')} MB | pyarrow pool now {pool.bytes_allocated()//2**20} MB, max {pool.max_memory()//2**20} MB", flush=True)
    del df
