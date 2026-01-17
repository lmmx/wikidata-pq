"""Core partitioning sink logic.

Writes language/site-partitioned parquet files with source filename preservation
and audit sidecar generation.
"""

from functools import partial
from pathlib import Path

import polars as pl
import polars.selectors as cs


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
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    (
        report.select(cs.by_index(range(5)))
        .unnest(cs.struct())
        .drop(cs.matches("count"))
        .rename({"lower_bound": "min_id", "upper_bound": "max_id"})
        .write_parquet(sidecar)
    )


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
