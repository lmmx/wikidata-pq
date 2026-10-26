import os
import glob, pyarrow.parquet as pq, pyarrow.compute as pc, polars as pl
P = os.environ["WORK"]  # split.py's work dir (parts/{set}/claims/chunk_*.parquet)
bad = tot = 0
for f in sorted(glob.glob(f"{P}/parts/*/claims/chunk_*.parquet")):
    r = pq.read_table(f, columns=["references"]).column(0).combine_chunks()
    dv = r.flatten().field("snaks").flatten().field("value").flatten().field("datavalue")
    d = pl.DataFrame({k: pl.from_arrow(dv.field(k)) for k in ("id", "numeric-id", "entity-type")})
    d = d.filter(pl.col("id").is_not_null() | pl.col("numeric-id").is_not_null())
    num = pl.when(pl.col("id").str.contains(r"^[QPLM]\d+$")).then(pl.col("id").str.slice(1).cast(pl.Int64))
    tot += d.height; bad += d.filter(~pl.col("numeric-id").eq_missing(num)).height
print("references", tot, "rows with an entity value; numeric-id not derivable:", bad)
