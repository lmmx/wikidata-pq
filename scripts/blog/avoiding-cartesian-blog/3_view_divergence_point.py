import polars as pl
from pathlib import Path

claims_file = Path("results/claims/chunk_0-00400-of-00546.parquet")
WIKIBASE_TYPES = ["wikibase-item", "wikibase-property"]

lf = pl.scan_parquet(claims_file)

base = (
    lf.select(pl.col("claims").explode().struct.unnest())
    .drop("key")
    .explode("value")
    .unnest("value")
    .unnest("mainsnak")
    .filter(pl.col("datatype").is_in(WIKIBASE_TYPES))
    .head(1)
)

efficient_stage = (
    base
    .with_columns(
        pl.col("datavalue")
        .struct.field("labels")
        .list.eval(pl.element().struct.field("key"))
        .alias("_dv_langs")
    )
    .explode("property-labels")
    .with_columns(
        pl.col("property-labels")
        .struct.rename_fields(["language", "property_label"])
    )
    .unnest("property-labels")
    .filter(pl.col("language").is_in(pl.col("_dv_langs")))
    .select(["language", "property_label"])
    .collect()
)

print(efficient_stage)
print(f"\nrows kept: {efficient_stage.height}")
