"""Partitioning helpers.

The `partition_parquet` function writes language-partitioned subset dirs with parquets
named the same as the source file they came from.

Subdirectories are made automatically, and sidecar reports are written in "audit/".
"""

from functools import partial
from pathlib import Path

import polars as pl
import polars.selectors as cs

# log_dir = Path("audit")


def custom_file_path(
    ctx: pl.KeyedPartitionContext, source: Path, ext: str = ".parquet"
) -> str:
    """If just one file per partition as expected, filename = the source file's."""
    partition_dir = Path(ctx.keys[0].str_value)  # the language code e.g. "en", "fr"
    stem = source.stem  # name the subset file the same as the original source file
    if ctx.in_part_idx > 0:
        stem += f"_{ctx.in_part_idx}"
    filepath = (partition_dir / stem).with_suffix(ext)
    return str(filepath)


def sink_sidecar(report: pl.DataFrame, *, source: Path, log_dir: Path) -> None:
    """Write the report to sidecar file named the same as the source file."""
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


def partition_parquet(by: str, source_pq: Path, dst_dir: Path, log_dir: Path) -> None:
    """Loop over the source files and call this to make partitions.

    The files will keep the same filename as `source_pq`, under `dst_dir` in subdirs
    named from the partition `key`.
    """
    cb = partial(sink_sidecar, source=source_pq, log_dir=log_dir)
    fp = partial(custom_file_path, source=source_pq)
    partition = pl.PartitionByKey(dst_dir, by=[by], file_path=fp, finish_callback=cb)
    pl.scan_parquet(source_pq).sink_parquet(partition, mkdir=True)
