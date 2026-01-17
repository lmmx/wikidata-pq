#!/usr/bin/env python3
"""Test claims transformation for partitioning."""

from pathlib import Path

import polars as pl

claims_file = Path("results/claims/chunk_0-00400-of-00546.parquet")
lf = pl.scan_parquet(claims_file).head(2)

print("=== Step-by-step claims transformation ===\n")

# Step 1: Explode outer claims list
step1 = lf.select(pl.col("claims").explode().struct.unnest())
print(f"Step 1 - Explode claims list:\n{step1.head(2).collect()}\n")
print(f"Columns: {step1.collect_schema().keys()}\n")

# Step 2: Drop redundant 'key' (same as property), explode inner value list
step2 = step1.drop("key").explode("value").unnest("value")
print(f"Step 2 - Explode value, unnest:\n{step2.head(2).collect()}\n")
print(f"Columns: {step2.collect_schema().keys()}\n")

# Step 3: Unnest mainsnak to access property-labels
step3 = step2.unnest("mainsnak")
print(f"Step 3 - Unnest mainsnak:\n{step3.head(2).collect()}\n")
print(f"Columns: {step3.collect_schema().keys()}\n")

# Step 4: Explode property-labels, rename fields to get 'language' column
step4 = (
    step3.explode("property-labels")
    .with_columns(
        pl.col("property-labels").struct.rename_fields(["language", "property_label"])
    )
    .unnest("property-labels")
)
print(
    f"Step 4 - Explode property-labels, extract language:\n{step4.head(5).collect()}\n"
)
print(f"Columns: {step4.collect_schema().keys()}\n")

# Check we have the language column
assert "language" in step4.collect_schema().keys(), "Missing language column!"

# Sample the language distribution
print("=== Language distribution (top 10) ===")
lang_dist = (
    step4.group_by("language").len().sort("len", descending=True).head(10).collect()
)
print(lang_dist)

print("\n✓ Claims transformation produces 'language' column suitable for partitioning")
