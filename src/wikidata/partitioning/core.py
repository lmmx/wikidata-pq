"""Core partitioning sink logic.

Writes language/site-partitioned parquet files with source filename preservation
and audit sidecar generation.
"""

from functools import partial
from pathlib import Path

import polars as pl
from polars.io.partition import FileProviderArgs


def custom_file_path(
    args: FileProviderArgs, source: Path, ext: str = ".parquet"
) -> str:
    """Partition files keep source filename under language/site subdirs."""
    partition_dir = Path(str(args.partition_keys.item(0, 0)))
    stem = source.stem
    if args.index_in_partition > 0:
        stem += f"_{args.index_in_partition}"
    return str((partition_dir / stem).with_suffix(ext))


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
        lf = pl.scan_parquet(f)
        # The claims label lookup is keyed by ref (the labelled id/property/unit)
        id_col = "id" if "id" in lf.collect_schema() else "ref"
        stats = (
            lf.select(pl.len(), pl.col(id_col).min().alias("min"), pl.col(id_col).max())
            .collect()
            .row(0)
        )
        rows.append((str(f), stats[0], f.stat().st_size, f.parent.name, *stats[1:]))
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
        by: Partition column name (either "language" or "site")
        lf: LazyFrame already transformed for partitioning
        source_name: Original filename (for partition file naming and audit)
        dst_dir: Output directory for partitioned files
        log_dir: Directory for audit sidecar files
    """
    source_pq = Path(source_name)
    fp = partial(custom_file_path, source=source_pq)
    partition = pl.PartitionBy(dst_dir, key=by, file_path_provider=fp, include_key=True)
    lf.sink_parquet(partition, mkdir=True)
    write_sidecar(by, dst_dir=dst_dir, source=source_pq, log_dir=log_dir)
