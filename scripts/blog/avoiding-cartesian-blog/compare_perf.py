#!/usr/bin/env python3
"""Compare performance of cartesian vs efficient approach."""

import time
from pathlib import Path

import polars as pl

claims_file = Path("results/claims/chunk_0-00400-of-00546.parquet")
WIKIBASE_TYPES = ["wikibase-item", "wikibase-property"]


def get_base(lf: pl.LazyFrame, n: int) -> pl.DataFrame:
    return (
        lf.select(pl.col("claims").explode().struct.unnest())
        .drop("key")
        .explode("value")
        .unnest("value")
        .unnest("mainsnak")
        .head(n)
        .collect()
    )


def transform_cartesian(base: pl.DataFrame) -> pl.DataFrame:
    """Original approach - cartesian product then filter."""
    return (
        base.lazy()
        .filter(pl.col("datatype").is_in(WIKIBASE_TYPES))
        .explode("property-labels")
        .with_columns(
            pl.col("property-labels").struct.rename_fields(
                ["prop_lang", "property_label"]
            )
        )
        .unnest("property-labels")
        .with_columns(pl.col("datavalue").struct.field("labels").alias("dv_labels"))
        .explode("dv_labels")
        .with_columns(
            pl.col("dv_labels").struct.rename_fields(["dv_lang", "datavalue_label"])
        )
        .unnest("dv_labels")
        .filter(pl.col("prop_lang") == pl.col("dv_lang"))
        .rename({"prop_lang": "language"})
        .drop("dv_lang")
        .collect()
    )


def transform_efficient(base: pl.DataFrame) -> pl.DataFrame:
    """Optimized - filter before extraction, use map_elements for lookup."""
    return (
        base.lazy()
        .filter(pl.col("datatype").is_in(WIKIBASE_TYPES))
        # Get available languages in datavalue.labels
        .with_columns(
            pl.col("datavalue")
            .struct.field("labels")
            .list.eval(pl.element().struct.field("key"))
            .alias("_dv_langs")
        )
        # Explode property-labels only
        .explode("property-labels")
        .with_columns(
            pl.col("property-labels").struct.rename_fields(
                ["language", "property_label"]
            )
        )
        .unnest("property-labels")
        # Filter to languages that exist in datavalue.labels
        .filter(pl.col("language").is_in(pl.col("_dv_langs")))
        # Extract matching label without exploding
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
        .drop("_dv_langs")
        .collect()
    )


# Test with increasing sizes
lf = pl.scan_parquet(claims_file)

for n in [10, 50, 100, 500]:
    print(f"\n{'='*60}")
    print(f"Testing with {n} base claims")
    print("=" * 60)

    base = get_base(lf, n)
    wb_count = base.filter(pl.col("datatype").is_in(WIKIBASE_TYPES)).height
    print(f"  ({wb_count} are wikibase-item/property)")

    t0 = time.perf_counter()
    r1 = transform_cartesian(base)
    t1 = time.perf_counter()
    print(f"Cartesian:  {t1-t0:.3f}s, {len(r1)} rows")

    t0 = time.perf_counter()
    r2 = transform_efficient(base)
    t1 = time.perf_counter()
    print(f"Efficient:  {t1-t0:.3f}s, {len(r2)} rows")

    # Verify same results
    assert len(r1) == len(r2), f"Row count mismatch: {len(r1)} vs {len(r2)}"
