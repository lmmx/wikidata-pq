#!/usr/bin/env python3
"""Verify transformation patterns for each table type."""

from pathlib import Path

import polars as pl

RESULTS = Path("results")
TEST_FILE = "chunk_0-00400-of-00546.parquet"


def transform_map_record(lf: pl.LazyFrame, col: str) -> pl.LazyFrame:
    """Labels, descriptions: Map<Record{language, value}>"""
    return lf.select(pl.col(col).explode().struct.unnest()).unnest("value").drop("key")


def transform_map_list_record(lf: pl.LazyFrame, col: str) -> pl.LazyFrame:
    """Aliases: Map<List<Record{language, value}>>"""
    return (
        lf.select(pl.col(col).explode().struct.unnest())
        .explode("value")
        .unnest("value")
        .drop("key")
    )


def transform_sitelinks(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Links: Map<Record{site, title}>"""
    return (
        lf.select(pl.col("sitelinks").explode().struct.unnest())
        .unnest("value")
        .drop("key")
    )


# Expected outcomes
specs = [
    ("labels", "labels", transform_map_record, ["language", "value"]),
    ("descriptions", "descriptions", transform_map_record, ["language", "value"]),
    ("aliases", "aliases", transform_map_list_record, ["language", "value"]),
    ("links", "sitelinks", transform_sitelinks, ["site", "title"]),
]

for tbl_name, col, fn, expected_cols in specs:
    path = RESULTS / tbl_name / TEST_FILE
    if not path.exists():
        print(f"⚠️  {tbl_name}: not found at {path}")
        continue

    lf = pl.scan_parquet(path).drop_nulls()
    print(f"\n{'='*60}\n{tbl_name.upper()} (column: {col})\n{'='*60}")
    print(f"Input schema: {lf.collect_schema()}")

    result = fn(lf, col) if col != "sitelinks" else fn(lf)
    result_schema = result.collect_schema()

    print(f"Output schema: {result_schema}")
    print(f"Expected columns: {expected_cols}")

    # Verify
    actual_cols = list(result_schema.keys())
    assert actual_cols == expected_cols, f"Mismatch: got {actual_cols}"

    sample = result.head(5).collect()
    print(f"\nSample:\n{sample}")
    print(f"✓ {tbl_name} OK")
