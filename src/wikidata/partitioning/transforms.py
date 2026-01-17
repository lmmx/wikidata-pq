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


def prepare_map_record(lf: pl.LazyFrame, col: str) -> pl.LazyFrame:
    """Labels, descriptions, links: Map<Record{language/site, value/title}>."""
    return lf.select(pl.col(col).explode().struct.unnest()).unnest("value").drop("key")


def prepare_map_list_record(lf: pl.LazyFrame, col: str) -> pl.LazyFrame:
    """Aliases: Map<List<Record{language, value}>>."""
    return (
        lf.select(pl.col(col).explode().struct.unnest())
        .explode("value")
        .unnest("value")
        .drop("key")
    )


def prepare_for_partition(table_file: Path, table: Table) -> pl.LazyFrame:
    """Flatten nested structure for partitioning. Streaming-safe (no collect).

    Returns lazyframe with scalar columns ready for language/site partitioning.
    """
    lf = pl.scan_parquet(table_file).drop_nulls()

    if table == Table.CLAIMS:
        return prepare_claims(lf)

    col = TABLE_COLS[table]

    if table == Table.ALIAS:
        return prepare_map_list_record(lf, col)

    # Labels, descriptions, links: all Map<Record>
    return prepare_map_record(lf, col)
