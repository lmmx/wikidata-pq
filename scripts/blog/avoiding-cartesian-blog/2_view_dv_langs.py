import polars as pl
from pathlib import Path

claims_file = Path("results/claims/chunk_0-00400-of-00546.parquet")

lf = pl.scan_parquet(claims_file)

df = (
    lf.select(pl.col("claims").explode().struct.unnest())
    .drop("key")
    .explode("value")
    .unnest("value")
    .unnest("mainsnak")
    .head(3)
    .with_columns(
        pl.col("datavalue")
        .struct.field("labels")
        .list.eval(pl.element().struct.field("key"))
        .alias("_dv_langs")
    )
    .select(["property", "_dv_langs"])
    .collect()
)

print(df)
