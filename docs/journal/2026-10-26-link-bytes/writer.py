"""Size and CPU of zstd levels on the bench's quick-tier sample: each table's files
(group files or published files) rewritten with pq_glue's parallel writer at each level,
the way the pipeline writes them (group: no CDC; published: CDC, sorted by id), with
the whole write's CPU (encode and compress) and the read's (pyarrow, Polars). A check
on pages.py: its compress CPU is the part of the write that the level changes.

Usage: writer.py ROOT KIND(staging|final) OUT_DIR LEVELS [ROUNDS] [TABLES]"""
import json, resource, sys, time, os
from pathlib import Path
from statistics import median
import pyarrow.parquet as pq
import polars as pl
import pq_glue

def cpu():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime

root, kind, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
levels = [int(x) for x in sys.argv[4].split(",")]
rounds = int(sys.argv[5]) if len(sys.argv) > 5 else 3
tables = sys.argv[6].split(",") if len(sys.argv) > 6 else None
out.mkdir(parents=True, exist_ok=True)
res = []
by = {}
for p in sorted(root.rglob("*.parquet")):
    t = p.relative_to(root).parts[1 if kind == "staging" else 0]
    by.setdefault(t, []).append(p)
for t, files in sorted(by.items()):
    if tables and t not in tables:
        continue
    data = [(pq.read_table(f), pq.ParquetFile(f).metadata) for f in files]
    orig = sum(f.stat().st_size for f in files)
    for cdc, lv in [(c, l) for c in ((False,) if kind == "staging" else (True, False)) for l in levels]:
        ws, wc, rs, rc, plc, size = [], [], [], [], [], 0
        for r in range(rounds):
            c0, w0 = cpu(), time.perf_counter()
            size = 0
            for i, (tb, md) in enumerate(data):
                dst = out / f"{i}.parquet"
                rg = md.row_group(0).num_rows if md.num_row_groups else None
                sc = [("id", False, False)] if kind == "final" and "id" in tb.schema.names else None
                pq_glue.write_parquet(tb, dst, engine="parallel", row_group_size=rg, compression="zstd",
                                      compression_level=lv, content_defined_chunking=cdc, sorting_columns=sc)
                size += dst.stat().st_size
            wc.append(cpu() - c0); ws.append(time.perf_counter() - w0)
            c0, w0 = cpu(), time.perf_counter()
            for i in range(len(data)):
                pq.read_table(out / f"{i}.parquet")
            rc.append(cpu() - c0); rs.append(time.perf_counter() - w0)
            c0 = cpu()
            for i in range(len(data)):
                pl.read_parquet(out / f"{i}.parquet")
            plc.append(cpu() - c0)
        for i in range(len(data)):
            (out / f"{i}.parquet").unlink()
        row = {"kind": kind, "table": t, "files": len(files), "orig_mb": round(orig / 1e6, 3), "cdc": cdc, "level": lv,
               "mb": round(size / 1e6, 3), "write_cpu": round(median(wc), 3), "write_wall": round(median(ws), 3),
               "read_cpu_pa": round(median(rc), 3), "read_cpu_pl": round(median(plc), 3)}
        print(json.dumps(row), flush=True)
        res.append(row)
