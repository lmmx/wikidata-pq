"""Sidecar rows from Parquet footers (row counts, the id column's min/max statistics) against
write_sidecar's scans, on a chunk's partition output. Usage: sidecar_stats.py <src> <results> <stem>"""
import sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import polars as pl, pyarrow.parquet as pq
res, stem = Path(sys.argv[2]), sys.argv[3]

def footer_stats(f, id_col):
    md = pq.read_metadata(f)
    col = next(i for i in range(md.num_columns) if md.schema.column(i).path == id_col)
    lo = hi = None
    for g in range(md.num_row_groups):
        s = md.row_group(g).column(col).statistics
        if s is None or not s.has_min_max:
            if md.row_group(g).num_rows and (s is None or s.null_count != md.row_group(g).num_rows):
                raise ValueError(f"{f}: no min/max for {id_col} in row group {g}")
            continue
        lo = s.min if lo is None else min(lo, s.min)
        hi = s.max if hi is None else max(hi, s.max)
    return md.num_rows, lo, hi

bad = n = 0; ts = tf = 0.0
for tbl in sorted(p for p in res.iterdir() if p.is_dir()):
    files = sorted(tbl.glob(f"*/{stem}.parquet")) + sorted(tbl.glob(f"*/{stem}_*.parquet"))
    for f in files:
        lf = pl.scan_parquet(f); id_col = "id" if "id" in lf.collect_schema() else "ref"
        t = time.perf_counter(); a = lf.select(pl.len(), pl.col(id_col).min().alias("min"), pl.col(id_col).max()).collect().row(0); ts += time.perf_counter() - t
        t = time.perf_counter(); b = footer_stats(f, id_col); tf += time.perf_counter() - t
        n += 1
        if tuple(a) != tuple(b):
            bad += 1; print("DIFF", f, a, b)
print(f"{n} files, {bad} differ; scans {ts:.2f}s, footers {tf:.2f}s")
md = pq.read_metadata(next(res.glob("claims/*/*.parquet"))); print("claims row groups", md.num_row_groups, "stats exact?", md.row_group(0).column(0).statistics)
