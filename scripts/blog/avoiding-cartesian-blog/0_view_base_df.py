import polars as pl
from pathlib import Path

claims_file = Path("results/claims/chunk_0-00400-of-00546.parquet")

lf = pl.scan_parquet(claims_file)

base = (
    lf.select(pl.col("claims").explode().struct.unnest())
    .drop("key")
    .explode("value")
    .unnest("value")
    .unnest("mainsnak")
    .head(1)
    .collect()
)

print(base)
print()
print(base.schema)
