"""Pre-partition transforms for flattening nested structures.

Each table type has nested JSON that needs flattening before partitioning.
Simple tables (labels, descriptions, aliases, links) are handled here.
Claims are delegated to the claims module due to their complexity.
"""

from pathlib import Path

import polars as pl

from ..config import Table
from .claims import prepare_claims

TABLE_COLS = {
    Table.LABEL: "labels",
    Table.DESC: "descriptions",
    Table.ALIAS: "aliases",
    Table.LINKS: "sitelinks",
}


def prepare_map_string(lf: pl.LazyFrame, col: str) -> pl.LazyFrame:
    """Labels, descriptions: Map<String>, keyed by language."""
    return (
        lf.explode(col, empty_as_null=True)
        .select("id", pl.col(col).struct.unnest())
        .rename({"key": "language"})
    )


def prepare_map_list_string(lf: pl.LazyFrame, col: str) -> pl.LazyFrame:
    """Aliases: Map<List<String>>, keyed by language."""
    return prepare_map_string(lf, col).explode("value", empty_as_null=True)


def prepare_map_record(lf: pl.LazyFrame, col: str) -> pl.LazyFrame:
    """Links: Map<Record{site, title}>."""
    return (
        lf.explode(col, empty_as_null=True)
        .select("id", pl.col(col).struct.unnest())
        .unnest("value")
        .drop("key")
    )


def prepare_for_partition(table_file: Path, table: Table) -> pl.LazyFrame:
    """Flatten nested structure for partitioning. Streaming-safe (no collect).

    Returns lazyframe with scalar columns ready for language/site partitioning.
    """
    lf = pl.scan_parquet(table_file).drop_nulls()

    if table == Table.CLAIMS:
        tables_dir = table_file.parent.parent
        lookup = pl.scan_parquet(tables_dir / Table.CLAIMS_LABELS / table_file.name)
        labels = pl.scan_parquet(tables_dir / Table.LABEL / table_file.name)
        return prepare_claims(lf, lookup, labels)

    if table == Table.CLAIMS_LABELS:  # Already one row per language
        return lf

    col = TABLE_COLS[table]

    if table == Table.ALIAS:
        return prepare_map_list_string(lf, col)

    if table == Table.LINKS:
        return prepare_map_record(lf, col)

    # Labels, descriptions: Map<String>
    return prepare_map_string(lf, col)
