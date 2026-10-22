"""Core partitioning sink logic.

Writes language/site-partitioned parquet files with source filename preservation
and audit sidecar generation.
"""

from functools import partial
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
from polars.io.partition import FileProviderArgs

from ..config import UNSPLIT_COL


def custom_file_path(
    args: FileProviderArgs, source: Path, ext: str = ".parquet"
) -> str:
    """Partition files keep source filename under language/site subdirs."""
    partition_dir = Path(str(args.partition_keys.item(0, 0)))
    stem = source.stem
    if args.index_in_partition > 0:
        stem += f"_{args.index_in_partition}"
    return str((partition_dir / stem).with_suffix(ext))


# Polars truncates string statistics to 64 bytes: ones shorter than this are exact
EXACT_STAT_BYTES = 60


def file_stats(f: Path) -> tuple[int, str | None, str | None]:
    """A partition file's row count and the min and max of its ids.

    Read from the footer the sink has just written, without reading the data, unless a
    row group has no min/max for the ids or one may be truncated: then the file is read.
    """
    meta = pq.read_metadata(f)
    paths = [meta.schema.column(i).path for i in range(meta.num_columns)]
    # The claims label lookup is keyed by ref (the labelled id/property/unit)
    id_col = "id" if "id" in paths else "ref"
    col = paths.index(id_col)
    stats = [
        meta.row_group(g).column(col).statistics for g in range(meta.num_row_groups)
    ]
    if all(s is not None and s.has_min_max for s in stats):
        lows, highs = [s.min for s in stats], [s.max for s in stats]
        if all(len(b.encode()) < EXACT_STAT_BYTES for b in lows + highs):
            return meta.num_rows, min(lows, default=None), max(highs, default=None)
    lf = pl.scan_parquet(f).select(
        pl.len(), pl.col(id_col).min().alias("min"), pl.col(id_col).max()
    )
    return lf.collect().row(0)


def write_sidecar(by: str, *, dst_dir: Path, source: Path, log_dir: Path) -> None:
    """Write audit sidecar with row counts and ID bounds per partition file."""
    stem = source.stem
    files = sorted(dst_dir.glob(f"*/{stem}.parquet"))
    files += sorted(dst_dir.glob(f"*/{stem}_*.parquet"))
    schema = {
        "path": pl.String,
        "num_rows": pl.UInt64,
        "file_size": pl.UInt64,
        by: pl.String,
        "min_id": pl.String,
        "max_id": pl.String,
    }
    rows = []
    for f in files:
        num_rows, min_id, max_id = file_stats(f)
        rows.append((str(f), num_rows, f.stat().st_size, f.parent.name, min_id, max_id))
    sidecar = log_dir / source.name
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows, schema=schema, orient="row").write_parquet(sidecar)


def partition_parquet(
    by: str,
    lf: pl.LazyFrame,
    source_name: str,
    dst_dir: Path,
    log_dir: Path,
) -> None:
    """Sink a lazyframe to language/site-partitioned parquets.

    Args:
        by: Partition column name ("language", "site", or UNSPLIT_COL, which is not
            written to the files)
        lf: LazyFrame already transformed for partitioning
        source_name: Original filename (for partition file naming and audit)
        dst_dir: Output directory for partitioned files
        log_dir: Directory for audit sidecar files
    """
    source_pq = Path(source_name)
    fp = partial(custom_file_path, source=source_pq)
    partition = pl.PartitionBy(
        dst_dir, key=by, file_path_provider=fp, include_key=by != UNSPLIT_COL
    )
    lf.sink_parquet(partition, mkdir=True)
    write_sidecar(by, dst_dir=dst_dir, source=source_pq, log_dir=log_dir)
