"""Partitioning helpers. Includes pre-partition transformation.

The `partition_parquet` function writes language-partitioned subset dirs with parquets
named the same as the source file they came from.

Subdirectories are made automatically, and sidecar reports are written in "audit/".
"""

from functools import partial
from pathlib import Path

import polars as pl
import polars.selectors as cs

from .config import Table

# Column names per table (claims is not included, handled separately)
_TABLE_COLS = {
    Table.LABEL: "labels",
    Table.DESC: "descriptions",
    Table.ALIAS: "aliases",
    Table.LINKS: "sitelinks",
}


def prepare_for_partition(table_file: Path, table: Table) -> pl.LazyFrame:
    """Flatten nested structure for partitioning.

    Returns lazyframe with scalar columns ready for language/site partitioning.
    Claims are passed through unchanged (their transformation is different).
    """
    lf = pl.scan_parquet(table_file).drop_nulls()
    if table == Table.CLAIMS:
        # Claims have complex structure; handled in dedicated code path
        result = lf
    else:
        # Base: explode the list, unnest the key/value struct
        base = lf.select(pl.col(_TABLE_COLS[table]).explode().struct.unnest())
        if table == Table.ALIAS:
            # Aliases: Map<List<Record>> needs double explode
            result = base.explode("value")
        else:
            # Labels, descriptions, links: Map<Record>
            result = base
        result = result.unnest("value").drop("key")
    return result


def custom_file_path(
    ctx: pl.KeyedPartitionContext, source: Path, ext: str = ".parquet"
) -> str:
    """Partition files keep source filename under language/site subdirs."""
    partition_dir = Path(ctx.keys[0].str_value)
    stem = source.stem
    if ctx.in_part_idx > 0:
        stem += f"_{ctx.in_part_idx}"
    return str((partition_dir / stem).with_suffix(ext))


def sink_sidecar(report: pl.DataFrame, *, source: Path, log_dir: Path) -> None:
    """Write audit sidecar with row counts and ID bounds per partition."""
    sidecar = log_dir / source.name
    (
        report.lazy()
        .select(cs.by_index(range(5)))
        .unnest(cs.struct())
        .drop(cs.matches("count"))
        .rename({"lower_bound": "min_id", "upper_bound": "max_id"})
        .sink_parquet(sidecar, mkdir=True)
    )
    return


def partition_parquet(
    by: str,
    lf: pl.LazyFrame,
    source_name: str,
    dst_dir: Path,
    log_dir: Path,
) -> None:
    """Sink a lazyframe to language/site-partitioned parquets.

    Args:
        by: Partition column name (either "language" or "site")
        lf: LazyFrame already transformed for partitioning
        source_name: Original filename (for partition file naming and audit)
        dst_dir: Output directory for partitioned files
        log_dir: Directory for audit sidecar files
    """
    source_pq = Path(source_name)
    cb = partial(sink_sidecar, source=source_pq, log_dir=log_dir)
    fp = partial(custom_file_path, source=source_pq)
    partition = pl.PartitionByKey(dst_dir, by=[by], file_path=fp, finish_callback=cb)
    lf.sink_parquet(partition, mkdir=True)
    return
