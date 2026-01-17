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
    .head(5)
    .collect()
)

from debug_claims5 import transform_cartesian, transform_efficient

r1 = transform_cartesian(base).select(
    ["language", "property_label", "datavalue_label"]
)

r2 = transform_efficient(base).select(
    ["language", "property_label", "datavalue_label"]
)

print("Cartesian result:")
print(r1.head(10))

print("\nEfficient result:")
print(r2.head(10))

print("\nSchemas identical:", r1.schema == r2.schema)
