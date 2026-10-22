"""One sidecar variant over all of a chunk's partition tables, in this fresh process: wall
and peak RSS. Usage: sidecar_mem.py <results> <stem> <perfile|concat|concatN|footer>"""
import sys, time, resource
from pathlib import Path
import polars as pl, pyarrow.parquet as pq
res, stem, mode = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
def q(f, id_col): return pl.scan_parquet(f).select(pl.len(), pl.col(id_col).min().alias("min"), pl.col(id_col).max())
t = time.perf_counter(); rows = 0
for tbl in sorted(p for p in res.iterdir() if p.is_dir()):
    files = sorted(tbl.glob(f"*/{stem}.parquet")) + sorted(tbl.glob(f"*/{stem}_*.parquet"))
    if not files: continue
    id_col = "id" if "id" in pl.scan_parquet(files[0]).collect_schema() else "ref"
    if mode == "perfile":
        out = [q(f, id_col).collect().row(0) for f in files]
    elif mode.startswith("concat"):
        n = int(mode[6:] or len(files)); out = []
        for i in range(0, len(files), n):
            out += pl.concat([q(f, id_col) for f in files[i:i+n]]).collect().rows()
    elif mode == "footer":
        out = []
        for f in files:
            md = pq.read_metadata(f); c = next(i for i in range(md.num_columns) if md.schema.column(i).path == id_col)
            st = [md.row_group(g).column(c).statistics for g in range(md.num_row_groups)]
            out.append((md.num_rows, min(s.min for s in st), max(s.max for s in st)))
    rows += len(out)
print(f"{mode:9s} {rows} files {time.perf_counter()-t:.2f}s, maxrss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss//1024} MB")
