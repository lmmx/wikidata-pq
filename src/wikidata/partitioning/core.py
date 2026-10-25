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


def _write_sidecar_rows(
    by: str, rows: list[tuple], *, source: Path, log_dir: Path, row_groups: bool = False
) -> None:
    """Write a sidecar's rows: (path, num_rows, file_size, key, min_id, max_id), and
    with `row_groups` the key's row groups in `path` (see `write_one_file`)."""
    schema = {
        "path": pl.String,
        "num_rows": pl.UInt64,
        "file_size": pl.UInt64,
        by: pl.String,
        "min_id": pl.String,
        "max_id": pl.String,
    }
    if row_groups:
        schema["row_groups"] = pl.List(pl.UInt32)
    sidecar = log_dir / source.name
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows, schema=schema, orient="row").write_parquet(sidecar)


def write_sidecar(by: str, *, dst_dir: Path, source: Path, log_dir: Path) -> None:
    """Write audit sidecar with row counts and ID bounds per partition file."""
    stem = source.stem
    files = sorted(dst_dir.glob(f"*/{stem}.parquet"))
    files += sorted(dst_dir.glob(f"*/{stem}_*.parquet"))
    rows = []
    for f in files:
        num_rows, min_id, max_id = file_stats(f)
        rows.append((str(f), num_rows, f.stat().st_size, f.parent.name, min_id, max_id))
    _write_sidecar_rows(by, rows, source=source, log_dir=log_dir)


def key_stats(by: str, lf: pl.LazyFrame) -> pl.LazyFrame:
    """Each key's row count and the min and max of its ids (`ref` for the claims label
    lookup, as in `file_stats`): exact, as `file_stats` gives them."""
    id_col = "id" if "id" in lf.collect_schema() else "ref"
    return lf.group_by(by).agg(
        pl.len(), pl.col(id_col).min().alias("min"), pl.col(id_col).max().alias("max")
    )


def write_frame_sidecar(
    by: str, stats: pl.DataFrame, *, dst_dir: Path, source: Path, log_dir: Path
) -> None:
    """`write_sidecar` for the files a partition sink has just written, one per key
    (see `partition_parquet`), from the `key_stats` of the frame it wrote: the key dirs
    are not listed and no footer is read (a chunk writes a file to each of a table's
    hundreds of keys), only each file's size is taken."""
    # In the order the sidecar's glob was sorted in
    files = sorted(
        (dst_dir / str(key) / f"{source.stem}.parquet", n, lo, hi)
        for key, n, lo, hi in stats.iter_rows()
    )
    rows = [
        (str(f), n, f.stat().st_size, f.parent.name, lo, hi) for f, n, lo, hi in files
    ]
    _write_sidecar_rows(by, rows, source=source, log_dir=log_dir)


def one_file_path(dst_dir: Path, source: Path, by: str) -> Path:
    """The one partition file of a chunk's table split by `by` (see `write_one_file`),
    beside the chunk's processed file and not in any key's dir."""
    return dst_dir / f"{source.stem}.by_{by}.parquet"


def write_one_file(
    by: str, lf: pl.LazyFrame, *, dst_dir: Path, source: Path, log_dir: Path
) -> None:
    """Partition a chunk's table into ONE file, a row group per key, in place of a file
    per key (a chunk writes a few hundred keys a table, and a file per key cost about
    0.9 ms of writer set-up each: docs/journal/2026-10-25-partition-costs.md).

    The keys' rows are in the order of the sidecar rows a file per key had (their key
    dirs' paths sorted), each key's rows in the frame's order: a key in one row group,
    as the sink wrote a key's file (a chunk's keys are each of a few thousand rows), or
    one of more than ARROW_ROW_GROUP_ROWS rows in row groups of that many (where the
    sink split it at its own morsels). The group merge glues a key's row groups out of
    each chunk's file (pq_glue.concat_many with row groups), so a key's group file has
    the row groups, rows and Arrow schema it had. Written with pyarrow's writer (one
    writer for all the keys), by way of a temporary file.

    The sidecar has a row per key, as before: its `path` is the one file, `row_groups`
    the key's row groups in it, and `file_size` the bytes of those row groups' column
    chunks (the first key's also the file's header and footer, so a chunk's sizes still
    sum to its files'). Row counts and id bounds are taken from the frame, as
    `key_stats` takes them.
    """
    df = lf.collect().rechunk()
    dst = one_file_path(dst_dir, source, by)
    id_col = "id" if "id" in df.columns else "ref"
    # Each key's rows (their positions, in the frame's order) and `key_stats`, in one
    # group_by
    keys = (
        df.select(pl.int_range(pl.len(), dtype=pl.UInt32).alias("_row"), by, id_col)
        .group_by(by)
        .agg("_row", pl.len(), pl.col(id_col).min().alias("min"), pl.col(id_col).max().alias("max"))
    )
    # The order of a file per key's sidecar rows: by its path, the key's dir
    names = [str(k) for k in keys[by].to_list()]
    keys = keys[sorted(range(len(names)), key=lambda j: Path(names[j]))]
    rows = []
    if keys.height:
        # The rows key by key, each key's in a row group (or more) of its own
        table = df.to_arrow().take(keys["_row"].explode().to_arrow())
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".tmp")
        groups, start, offset = [], 0, 0
        # Dictionaries for every column but the ids (unique within a key, so a dictionary
        # only costs a page more): the key and the values, as Polars' sink encoded most
        # of them, so the sort reads them as quickly (written plain, the finalise's sort
        # took 6.6% longer, #106). zstd at Polars' level
        dictionary = [c for c in table.column_names if c != id_col]
        with pq.ParquetWriter(tmp, table.schema, compression="zstd", compression_level=3, use_dictionary=dictionary) as w:
            for n in keys["len"].to_list():
                w.write_table(table.slice(offset, n), row_group_size=ARROW_ROW_GROUP_ROWS)
                k = -(-n // ARROW_ROW_GROUP_ROWS)
                groups.append(list(range(start, start + k)))
                start, offset = start + k, offset + n
        meta = pq.read_metadata(tmp)
        if meta.num_row_groups != start or meta.num_rows != df.height:
            raise RuntimeError(f"[partition] {source.name}: {meta.num_row_groups} row groups,"
                               f" {meta.num_rows} rows written, not {start} and {df.height}")
        sizes = [
            sum(meta.row_group(g).column(c).total_compressed_size for c in range(meta.num_columns))
            for g in range(start)
        ]
        tmp.replace(dst)
        rest = dst.stat().st_size - sum(sizes)
        stats = keys.select(by, "len", "min", "max").iter_rows()
        for i, ((key, n, lo, hi), rgs) in enumerate(zip(stats, groups)):
            size = sum(sizes[g] for g in rgs) + (rest if i == 0 else 0)
            rows.append((str(dst), n, size, str(key), lo, hi, rgs))
    _write_sidecar_rows(by, rows, source=source, log_dir=log_dir, row_groups=True)


def partition_parquet(
    by: str,
    lf: pl.LazyFrame,
    source_name: str,
    dst_dir: Path,
    log_dir: Path,
    arrow: bool = False,
    one_file: bool = False,
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
        one_file: Write one file, a row group per key, not a file per key (see
            `write_one_file`)
    """
    source_pq = Path(source_name)
    if one_file:
        if by == UNSPLIT_COL:
            raise ValueError("an unsplit table has one file already")
        write_one_file(by, lf, dst_dir=dst_dir, source=source_pq, log_dir=log_dir)
        return
    if arrow:
        if by != UNSPLIT_COL:
            raise ValueError(f"arrow writes one unsplit file, not files by {by}")
        dst = dst_dir / UNSPLIT_KEY / f"{source_pq.stem}.parquet"
        write_arrow(lf.drop(UNSPLIT_COL), dst)
        write_sidecar(by, dst_dir=dst_dir, source=source_pq, log_dir=log_dir)
        return
    fp = partial(custom_file_path, source=source_pq)
    # One file per key, however large, so each key's file is the one the sidecar lists
    partition = pl.PartitionBy(
        dst_dir,
        key=by,
        file_path_provider=fp,
        include_key=by != UNSPLIT_COL,
        approximate_bytes_per_file=None,
    )
    # The sink and the sidecar's stats in one query, sharing its scan
    sink = lf.sink_parquet(partition, mkdir=True, lazy=True)
    _, stats = pl.collect_all([sink, key_stats(by, lf)])
    write_frame_sidecar(by, stats, dst_dir=dst_dir, source=source_pq, log_dir=log_dir)
