"""The labels, descriptions, aliases and links partitions written by the split, straight
from a part's reshaped columns (stage 1 of "fusion A": not yet called by dump.py or
process.py; docs/journal/2026-10-25-fused-terms-links.md).

Today these four tables take three passes after the split has written their JSON into the
chunk file: genson normalises each column (process.normalise_map_direct,
normalise_sitelinks), Polars explodes it (transforms.prepare_for_partition), and Polars'
sink writes a file per key (core.partition_parquet). Here one compiled call
(reshape/src/fuse.rs) does all three on the columns the split has in memory.

In design B's shape (claims_build.write_claims, process.place_built_claims):
- the split, for each part: `write_terms` writes each table's files to
  `routed/{table}/chunk_N/{key}.parquet`, then the table's `index.json` (its keys' rows,
  bytes and ids' bounds), the mark that the table is complete;
- `_move_routed` would move each `routed/{table}/chunk_N/` to `{table}/chunk_M/` beside the
  part, as it moves `routed/claims/chunk_N.parquet`;
- the pipeline: `place_built_terms` renames each key's file to
  `results/{table}/{key}/chunk_M.parquet` (partition_parquet's path) and writes the audit
  sidecar as `write_frame_sidecar` would, so merge_group and everything after are as now.

The partition files hold the same rows in the same order, in the same Parquet schema,
Arrow schema and row groups as the sink's (a key of up to 100,000 rows a chunk: see the
journal), under the same names; only their encoded bytes (and so `file_size`) differ.
scripts/test_fused_terms.py checks this on the bench's chunks.
"""

import os
import shutil
from pathlib import Path

import orjson
import polars as pl
import pyarrow as pa
from wikidata_reshape import fused_partitions

from ..config import PARTITION_COLS, Table
from .core import _write_sidecar_rows

# The tables built, in the order fused_partitions takes them, and their columns
FUSED = {
    Table.LABEL: "labels",
    Table.DESC: "descriptions",
    Table.ALIAS: "aliases",
    Table.LINKS: "sitelinks",
}
# A built table's mark of completion, beside its files (no key file has this name)
INDEX = "index.json"


def large_strings(column: pl.Series | pa.Array) -> pa.Array:
    """A string column as one Arrow large-string array (Polars' string views copied out)."""
    arr = column.to_arrow(compat_level=pl.CompatLevel.oldest()) if isinstance(column, pl.Series) else column
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks()
    return arr if arr.type == pa.large_string() else arr.cast(pa.large_string())


def _buffers(arr: pa.Array) -> tuple[pa.Buffer, pa.Buffer, int]:
    _, offsets, data = arr.buffers()
    return data, offsets, arr.offset


def build_terms(
    ids: pa.Array,
    texts: dict[Table, pa.Array],
    rows: pa.Array | None,
    dsts: dict[Table, Path],
    *,
    one_file: bool = False,
) -> tuple[dict[Table, list[tuple]] | None, str | None]:
    """The tables in `texts` built from a part's rows and written to `dsts`: each key's
    file at `dsts[table]/{key}.parquet` (or, with `one_file`, the table's one file at
    `dsts[table]`, a row group a key). `ids` and each text are the chunk's (large
    strings), `rows` the part's int64 positions in them (or None for all their rows).

    Returns each table's keys, by key: (key, rows, min id, max id, bytes, first row group,
    row groups), and None; or, if any cell of any table is outside what the build
    reproduces, None and why (nothing written)."""
    for name, arr in [("id", ids), *((t.value, a) for t, a in texts.items())]:
        if arr.null_count:
            return None, f"{name}: {arr.null_count} null cells"
    if rows is not None and (rows.offset or rows.null_count or rows.type != pa.int64()):
        rows = pa.array(rows.to_numpy(zero_copy_only=False), pa.int64())
    order = list(FUSED)
    columns = [_buffers(texts[t]) if t in texts else None for t in order]
    targets = [str(dsts[t]) if t in dsts else None for t in order]
    index = None if rows is None else rows.buffers()[1]
    status, result = fused_partitions(len(ids), _buffers(ids), columns, targets, one_file, index)
    if status == "misfit":
        return None, "; ".join(f"{order[t]}: row {row}: {why}" for t, row, why in result)
    return {t: files for t, files in zip(order, result) if files is not None}, None


def write_terms(
    ids: pl.Series | pa.Array,
    texts: dict[Table, pa.Array],
    rows: pa.Array | None,
    routed: Path,
    file: str,
) -> str | None:
    """Write a part's term and sitelink partitions in the split, as claims_build.write_claims
    writes its claims: `ids` and `texts` the chunk's (`texts` by table, of FUSED), `rows`
    the part's positions in them (or None), `routed` the set's routed directory, `file`
    the chunk's file name. Each table goes to `routed/{table}/{stem}/`, its INDEX written
    once all its files are. None once written, else why the part's tables stay as JSON
    (and nothing is left of an earlier attempt)."""
    stem = Path(file).stem
    dirs = {t: routed / t / stem for t in texts}
    for d in dirs.values():  # an earlier attempt's, interrupted
        shutil.rmtree(d, ignore_errors=True)
    tables, why = build_terms(large_strings(ids), texts, rows, dirs)
    if tables is None:
        return why
    for table, keys in tables.items():
        d = dirs[table]
        d.mkdir(parents=True, exist_ok=True)  # (a table with no rows has no files)
        tmp = d / f"{INDEX}.tmp"
        tmp.write_bytes(orjson.dumps([k[:5] for k in keys]))
        tmp.replace(d / INDEX)
    return None


def place_built_terms(built: Path, name: str, output_dir: Path, audit_dir: Path, tables=tuple(FUSED)) -> None:
    """Partition a chunk whose term and sitelink tables the split built (`write_terms`):
    each key's file renamed from `built/{table}/{stem}/` to
    `output_dir/{table}/{key}/{name}`, where partition_parquet writes it, then the audit
    sidecar written as write_frame_sidecar writes it, then the built directory removed. A
    rename already made is skipped, so this resumes."""
    stem = Path(name).stem
    for table in tables:
        d = built / table / stem
        sidecar = audit_dir / table / name
        if not d.exists() and sidecar.exists():
            continue  # placed before an interruption
        if not (d / INDEX).exists():
            raise RuntimeError(f"{table} {name}: no built partitions in {d}: split again")
        keys = orjson.loads((d / INDEX).read_bytes())
        dst_dir = output_dir / table
        rows = []
        for key, n, lo, hi, size in keys:
            src, dst = d / f"{key}.parquet", dst_dir / key / name
            if not os.path.exists(src) and os.path.exists(dst):
                pass  # renamed before an interruption
            else:
                try:
                    os.replace(src, dst)
                except FileNotFoundError:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(src, dst)
            # (path, num_rows, file_size, key, min_id, max_id), by key: as write_frame_sidecar
            rows.append((str(dst), n, size, key, lo, hi))
        _write_sidecar_rows(PARTITION_COLS[table], rows, source=Path(name), log_dir=audit_dir / table)
        (d / INDEX).unlink()
        d.rmdir()


def write_fused_partitions(frame: pl.DataFrame, name: str, output_dir: Path, audit_dir: Path, work: Path) -> str | None:
    """A whole chunk's frame through both steps (`write_terms` to `work`, then
    `place_built_terms`): the four tables partitioned as partition_chunk lays them out.
    None once written, else why not (nothing written)."""
    texts = {t: large_strings(frame.get_column(c)) for t, c in FUSED.items()}
    why = write_terms(frame.get_column("id"), texts, None, work, name)
    if why is None:
        place_built_terms(work, name, output_dir, audit_dir)
    return why
