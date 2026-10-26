"""The split tables' rows of the sample, by key (stable), in row groups of 122,880 as a
large key's group file holds them: PLAIN (the fused writer's) against pq-glue's adaptive
dictionary, at zstd 3; bytes and write CPU (3 rounds, min), and pyarrow read CPU."""
import glob, os, resource
import pyarrow as pa, pyarrow.parquet as pq, pq_glue, polars as pl
P = os.environ["WORK"]  # split.py's work dir (parts/{set}/claims/chunk_*.parquet)
def cpu(): r = resource.getrusage(resource.RUSAGE_SELF); return r.ru_utime + r.ru_stime
for tbl, key in (("labels", "language"), ("descriptions", "language"), ("aliases", "language"), ("links", "site")):
    for s in ("main", "scholar"):
        fs = sorted(glob.glob(f"{P}/parts/{s}/{tbl}/chunk_*.parquet"))
        t = pl.concat([pl.read_parquet(f) for f in fs]).with_row_index("_i").sort([key, "_i"]).drop("_i").to_arrow()
        leaves = [c for c in pq.read_metadata(fs[0]).schema.to_arrow_schema().names]
        meta = pq.read_metadata(fs[0]); paths = [meta.schema.column(i).path for i in range(meta.num_columns)]
        res = {}
        for name, ce in (("plain", {p: "PLAIN" for p in paths if p != key}), ("dict", None)):
            w = 1e9
            for _ in range(3):
                c0 = cpu(); pq_glue.write_parquet(t, f"{P}/f2.parquet", row_group_size=122_880, compression_level=3, column_encoding=ce); w = min(w, cpu() - c0)
            r = 1e9
            for _ in range(3):
                c0 = cpu(); pq.ParquetFile(f"{P}/f2.parquet").read(); r = min(r, cpu() - c0)
            res[name] = (os.path.getsize(f"{P}/f2.parquet"), round(w, 3), round(r, 3))
        b = res["plain"][0]
        print(f"{tbl:12s} {s:7s} rows {t.num_rows:8d} plain {res['plain']}  dict {res['dict']}  ({100*(res['dict'][0]-b)/b:+.1f}%)", flush=True)
os.unlink(f"{P}/f2.parquet")
