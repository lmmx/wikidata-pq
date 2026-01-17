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
    .collect()
)

cartesian = (
    base.lazy()
    .explode("property-labels")
    .with_columns(
        pl.col("property-labels")
        .struct.rename_fields(["prop_lang", "property_label"])
    )
    .unnest("property-labels")
    .with_columns(pl.col("datavalue").struct.field("labels").alias("dv_labels"))
    .explode("dv_labels")
    .with_columns(
        pl.col("dv_labels")
        .struct.rename_fields(["dv_lang", "datavalue_label"])
    )
    .unnest("dv_labels")
    .select(["prop_lang", "dv_lang", "property_label", "datavalue_label"])
    .collect()
)

print(cartesian)
print(f"\nrows produced: {cartesian.height}")
