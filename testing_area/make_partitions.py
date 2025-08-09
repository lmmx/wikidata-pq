from functools import partial
from pathlib import Path

import polars as pl
import polars.selectors as cs

src_dir = Path("data")
dst_dir = Path("./out")
sidecar = Path("monitoring")


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


def callback(report: pl.DataFrame, *, source: Path) -> None:
    """Write the report to sidecar file named under the subset"""
    (
        report.lazy()
        .select(cs.by_index(range(5)))
        .unnest(cs.struct())
        .drop(cs.matches("count"))
        .rename({"lower_bound": "min_id", "upper_bound": "max_id"})
        .sink_parquet(sidecar / source.name, mkdir=True)
    )
    return


for source_pq in sorted(src_dir.glob("*.parquet")):
    cb = partial(callback, source=source_pq)
    fp = partial(custom_file_path, source=source_pq)
    partition = pl.PartitionByKey(
        dst_dir, by=["language"], file_path=fp, finish_callback=cb
    )
    pl.scan_parquet(source_pq).sink_parquet(partition, mkdir=True)
