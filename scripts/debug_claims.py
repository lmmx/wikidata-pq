#!/usr/bin/env python3
"""Check claims structure to plan transformation."""
import polars as pl
from pathlib import Path

claims_file = Path("results/claims/chunk_0-00400-of-00546.parquet")
lf = pl.scan_parquet(claims_file)

print("=== Claims schema ===")
schema = lf.collect_schema()
print(schema)

print("\n=== Top-level columns ===")
print(list(schema.keys()))

# If it's a single 'claims' column, peek inside
if list(schema.keys()) == ["claims"]:
    print("\n=== Sample of claims column (1 row) ===")
    sample = lf.head(1).collect()
    print(sample)
    
    # Try to see what's inside
    print("\n=== Attempting to explode and unnest ===")
    try:
        exploded = lf.select(pl.col("claims").explode().struct.unnest()).head(3).collect()
        print(exploded)
        print(f"\nColumns after explode+unnest: {exploded.columns}")
    except Exception as e:
        print(f"Failed: {e}")