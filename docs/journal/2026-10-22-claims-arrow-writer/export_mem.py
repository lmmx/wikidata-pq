"""Where the pyarrow claims write's memory goes: RSS peak/after for the export alone (whole
frame; per 122,880-row slice), and for the write of an already exported table.
Usage: export_mem.py <src> <claims.parquet> <mode>"""
import os, resource, sys, time
from pathlib import Path
os.environ.setdefault("WIKIDATA_RELEASE", "20261012")
sys.path.insert(0, sys.argv[1])
import polars as pl, pyarrow as pa, pyarrow.parquet as pq
from wikidata.config import UNSPLIT_COL, Table
from wikidata.partitioning import prepare_for_partition
def kb(f): return next(int(l.split()[1]) for l in open("/proc/self/status") if l.startswith(f)) // 1024
f = "/tmp/scratch/cmp/w/em.parquet"
mode = sys.argv[3]
df = prepare_for_partition(Path(sys.argv[2]).resolve(), Table.CLAIMS).collect().drop(UNSPLIT_COL)
if mode == "write":
    tb = df.to_arrow(); print("exported table", tb.nbytes // 2**20, "MB")
open("/proc/self/clear_refs", "w").write("5"); r0 = kb("VmRSS"); t = time.perf_counter()
if mode == "whole":
    tb = df.to_arrow(); print("  table nbytes", tb.nbytes // 2**20, "MB"); del tb
elif mode == "slices":
    for part in df.iter_slices(122_880):
        x = part.to_arrow(); print("  slice nbytes", x.nbytes // 2**20, "MB", "rss", kb("VmRSS")); del x
elif mode == "slices_rechunk":
    for part in df.iter_slices(122_880):
        x = part.rechunk().to_arrow(); print("  slice nbytes", x.nbytes // 2**20, "MB", "rss", kb("VmRSS")); del x
elif mode == "write":
    pq.write_table(tb, f, compression="zstd", row_group_size=122_880)
print(f"{mode:15s} {time.perf_counter()-t:5.2f}s  RSS {r0} -> peak {kb('VmHWM')} -> {kb('VmRSS')} MB  (frame est {df.estimated_size()//2**20} MB)")
