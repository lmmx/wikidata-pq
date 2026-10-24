"""Core partitioning sink logic.

Writes language/site-partitioned parquet files with source filename preservation
and audit sidecar generation.
"""

import os
from functools import partial
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from polars.io.partition import FileProviderArgs
from polars_genson import write_parquet

from ..config import UNSPLIT_COL, UNSPLIT_KEY


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


# Rows per row group in a file written with pyarrow: as many as Polars' sink puts in one
ARROW_ROW_GROUP_ROWS = 122_880


def write_arrow(lf: pl.LazyFrame, dst: Path) -> None:
    """Write a frame to one Parquet file with pyarrow, as Polars' sink would write it.

    The claims' many nested leaves make Polars' writer slow: computing their repetition
    and definition levels was 30% of the pipeline's CPU, and pyarrow writes the same
    data with a third to a sixth of the CPU
    (docs/journal/2026-10-22-claims-arrow-writer.md).

    The file holds the same values, the Arrow schema Polars stores (Arrow's large
    types), row groups of the same size, and zstd at its default level. It is written a
    row group at a time, so the exported copy of one is freed before the next, and each
    slice is rechunked before its export: exported as it was, it took 5 times as long
    with twice the memory. It goes to a temporary file that replaces `dst` once
    complete. A frame with no rows writes no file, as the sink does.
    """
    df = lf.collect()
    if df.is_empty():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp")
    with pq.ParquetWriter(tmp, df.head(0).to_arrow().schema, compression="zstd") as w:
        for part in df.iter_slices(ARROW_ROW_GROUP_ROWS):
            w.write_table(part.rechunk().to_arrow())
    tmp.replace(dst)


def write_table(table: pa.Table, dst: Path) -> None:
    """Write an Arrow table to `dst` with genson's Parquet writer, a file with no rows
    included: row groups of ARROW_ROW_GROUP_ROWS rows, zstd, by way of a temporary file.

    genson encodes each row group's leaf columns in parallel, where pyarrow encodes them
    one at a time: the claims' statements, with their deeply nested references and
    qualifiers, write in a quarter of pyarrow's time on 4 threads and 5% smaller, as the
    same data in the same Arrow schema with the same id statistics
    (docs/journal/2026-10-24-genson-writer.md)."""
    tmp = dst.with_suffix(".tmp")
    write_parquet(table, tmp, row_group_size=ARROW_ROW_GROUP_ROWS)
    tmp.replace(dst)


def place_unsplit(src: Path, *, dst_dir: Path, log_dir: Path) -> None:
    """Partition a processed file that is already laid out as its table's one unsplit
    partition file (claims, see process.normalise_claims_direct): a hard link to it in
    place of a copy, then its audit sidecar. A file with no rows gives no partition file,
    as the sink gives none."""
    if pq.read_metadata(src).num_rows:
        dst = dst_dir / UNSPLIT_KEY / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".tmp")
        tmp.unlink(missing_ok=True)
        os.link(src, tmp)
        tmp.replace(dst)
    write_sidecar(UNSPLIT_COL, dst_dir=dst_dir, source=Path(src.name), log_dir=log_dir)


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
    arrow: bool = False,
) -> None:
    """Sink a lazyframe to language/site-partitioned parquets.

    Args:
        by: Partition column name ("language", "site", or UNSPLIT_COL, which is not
            written to the files)
        lf: LazyFrame already transformed for partitioning
        source_name: Original filename (for partition file naming and audit)
        dst_dir: Output directory for partitioned files
        log_dir: Directory for audit sidecar files
        arrow: Write the one unsplit file with pyarrow (see `write_arrow`), not
            Polars' sink
    """
    source_pq = Path(source_name)
    if arrow:
        if by != UNSPLIT_COL:
            raise ValueError(f"arrow writes one unsplit file, not files by {by}")
        dst = dst_dir / UNSPLIT_KEY / f"{source_pq.stem}.parquet"
        write_arrow(lf.drop(UNSPLIT_COL), dst)
    else:
        fp = partial(custom_file_path, source=source_pq)
        partition = pl.PartitionBy(
            dst_dir, key=by, file_path_provider=fp, include_key=by != UNSPLIT_COL
        )
        lf.sink_parquet(partition, mkdir=True)
    write_sidecar(by, dst_dir=dst_dir, source=source_pq, log_dir=log_dir)


from .._phases import instrument  # noqa: E402 (profile branch only)

instrument(globals(), ["file_stats", "write_arrow", "write_table", "place_unsplit", "write_sidecar",
                       "partition_parquet"])
