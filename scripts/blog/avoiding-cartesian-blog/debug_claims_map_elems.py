#!/usr/bin/env python3
"""Benchmark map_elements vs join-based lookup for label extraction."""

import time
from pathlib import Path

import polars as pl

claims_file = Path("results/claims/chunk_0-00400-of-00546.parquet")
WIKIBASE_TYPES = ["wikibase-item", "wikibase-property"]


def get_base(n: int | None = None) -> pl.LazyFrame:
    """Get base claims data, optionally limited to n rows."""
    lf = (
        pl.scan_parquet(claims_file)
        .select(pl.col("claims").explode().struct.unnest())
        .drop("key")
        .explode("value")
        .unnest("value")
        .unnest("mainsnak")
        .filter(pl.col("datatype").is_in(WIKIBASE_TYPES))
    )
    if n:
        lf = lf.head(n)
    return lf


def transform_map_elements(base: pl.LazyFrame) -> pl.LazyFrame:
    """Current approach using map_elements (slow, no parallelization)."""
    return (
        base.with_columns(
            pl.col("datavalue")
            .struct.field("labels")
            .list.eval(pl.element().struct.field("key"))
            .alias("dv_langs")
        )
        .explode("property-labels")
        .with_columns(
            pl.col("property-labels").struct.rename_fields(
                ["language", "property_label"]
            )
        )
        .unnest("property-labels")
        .filter(pl.col("language").is_in(pl.col("dv_langs")))
        .with_columns(
            pl.struct(["datavalue", "language"])
            .map_elements(
                lambda row: next(
                    (
                        item["value"]
                        for item in (row["datavalue"]["labels"] or [])
                        if item["key"] == row["language"]
                    ),
                    None,
                ),
                return_dtype=pl.String,
            )
            .alias("datavalue_label")
        )
        .drop("dv_langs")
    )


def transform_join_lookup(base: pl.LazyFrame) -> pl.LazyFrame:
    """Native Polars approach: join-based lookup."""
    # Add row index to correlate after exploding
    indexed = base.with_row_index("_row_id")

    # Create lookup table: explode labels, extract key/value
    labels_lookup = (
        indexed.select(
            "_row_id", pl.col("datavalue").struct.field("labels").alias("dv_labels")
        )
        .explode("dv_labels")
        .with_columns(
            pl.col("dv_labels").struct.field("key").alias("_lang"),
            pl.col("dv_labels").struct.field("value").alias("datavalue_label"),
        )
        .select("_row_id", "_lang", "datavalue_label")
    )

    # Main table: explode property-labels
    main = (
        indexed.explode("property-labels")
        .with_columns(
            pl.col("property-labels").struct.rename_fields(
                ["language", "property_label"]
            )
        )
        .unnest("property-labels")
    )

    # Join to get matching labels - inner join naturally filters to matches only
    return main.join(
        labels_lookup,
        left_on=["_row_id", "language"],
        right_on=["_row_id", "_lang"],
        how="inner",
    ).drop("_row_id")


# Benchmark
for n in [1, 5, 10, 50, 500, None]:
    print(f"\n{'='*60}")
    print(f"Testing with {n} claims")
    print("=" * 60)

    base = get_base(n)
    count = base.select(pl.len()).collect().item()
    print(f"  Wikibase claims: {count}")

    # map_elements approach
    t0 = time.perf_counter()
    r1 = transform_map_elements(base).collect()
    t1 = time.perf_counter()
    map_time = t1 - t0
    print(f"  map_elements:  {map_time:.3f}s, {len(r1)} rows")

    # Join lookup approach
    t0 = time.perf_counter()
    r2 = transform_join_lookup(base).collect()
    t1 = time.perf_counter()
    join_time = t1 - t0
    print(f"  join_lookup:   {join_time:.3f}s, {len(r2)} rows")

    speedup = map_time / join_time if join_time > 0 else float("inf")
    print(f"  Speedup: {speedup:.1f}x")

    # Verify results match
    if len(r1) == len(r2):
        print(f"  ✓ Row counts match")
        # Check values
        cols = ["property", "language", "property_label", "datavalue_label"]
        r1_check = r1.sort(cols[:2]).select(cols).head(10)
        r2_check = r2.sort(cols[:2]).select(cols).head(10)
        if r1_check.equals(r2_check):
            print(f"  ✓ Values match")
        else:
            print(f"  ⚠️  Values differ!")
            print("map_elements:")
            print(r1_check)
            print("join_lookup:")
            print(r2_check)
    else:
        print(f"  ⚠️  Row count mismatch: {len(r1)} vs {len(r2)}")
