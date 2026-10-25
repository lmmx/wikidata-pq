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


def place_built_terms(
    built: Path, name: str, output_dir: Path, audit_dir: Path, tables=tuple(FUSED), *, in_place: bool = False
) -> None:
    """Partition a chunk whose term and sitelink tables the split built (`write_terms`):
    each key's file renamed from `built/{table}/{stem}/` to
    `output_dir/{table}/{key}/{name}`, where partition_parquet writes it, then the audit
    sidecar written as write_frame_sidecar writes it, then the built directory removed. A
    rename already made is skipped, so this resumes.

    With `in_place`, nothing is renamed: the sidecar names each key's file where the split
    wrote it (merge_group reads the partition files from the sidecars' paths, and deletes
    them once merged; the emptied built directory is left for the run's clean-up)."""
    stem = Path(name).stem
    for table in tables:
        d = built / table / stem
        sidecar = audit_dir / table / name
        if not (d / INDEX).exists():
            if sidecar.exists():
                continue  # placed before an interruption
            raise RuntimeError(f"{table} {name}: no built partitions in {d}: split again")
        keys = orjson.loads((d / INDEX).read_bytes())
        dst_dir = output_dir / table
        rows = []
        for key, n, lo, hi, size in keys:
            src, dst = d / f"{key}.parquet", dst_dir / key / name
            if in_place:
                dst = src
            elif not os.path.exists(src) and os.path.exists(dst):
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
        if not in_place:
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


# The one-file layout, wired into the split and the pipeline (fusion A, stage 2): a built
# table is one file a part, `{stem}.parquet`, its row groups by key (in the keys' byte
# order), the key column kept as the sink keeps it; beside it `{stem}.json`, its index (a
# list of (key, rows, min id, max id, bytes, first row group, row groups)), written last.
# A table with no rows has its index (empty) and no file. The group's close
# (push.core._partition_table) makes each key's group file from its row groups in its
# chunks' files, copied as bytes from the spans the indexes give (partitioning.rowgroups),
# with no sink; a group with a chunk not built reads the files, in place of processed
# files, by the sink as before.

# The split builds these tables (WIKIDATA_SPLIT_TERMS=json: it leaves them as JSON)
SPLIT_TERMS = os.environ.get("WIKIDATA_SPLIT_TERMS", "built")
if SPLIT_TERMS not in ("built", "json"):
    raise SystemExit(f"WIKIDATA_SPLIT_TERMS is built or json, not {SPLIT_TERMS!r}")


def index_path(table_dir: Path, name: str) -> Path:
    """Where a built table's index is beside its file `name` (`chunk_N.parquet`)."""
    return table_dir / f"{Path(name).stem}.json"


def write_one_file(
    ids: pa.Array, texts: dict[Table, pa.Array], rows: pa.Array | None, routed: Path, file: str
) -> tuple[dict[Table, int] | None, str | None]:
    """The split's step (in the one-file layout): each table's file to
    `routed/{table}/{file}`, then its index beside it. Returns each table's file bytes (0
    for a table with no rows) and None; or None and why the part's tables stay as JSON (and
    nothing is left written)."""
    if SPLIT_TERMS == "json":
        return None, "not built (WIKIDATA_SPLIT_TERMS=json)"
    dsts = {t: routed / t / file for t in texts}
    for t, f in dsts.items():  # an earlier attempt's, interrupted
        f.unlink(missing_ok=True)
        index_path(f.parent, file).unlink(missing_ok=True)
        f.parent.mkdir(parents=True, exist_ok=True)
    tables, why = build_terms(large_strings(ids), texts, rows, dsts, one_file=True)
    if tables is None:
        for f in dsts.values():
            f.unlink(missing_ok=True)
        return None, why
    sizes = {}
    for t, f in dsts.items():
        keys = tables.get(t, [])
        tmp = index_path(f.parent, file).with_suffix(".json.tmp")
        tmp.write_bytes(orjson.dumps([list(k) for k in keys]))
        tmp.replace(index_path(f.parent, file))
        sizes[t] = f.stat().st_size if keys else 0
    return sizes, None


def built_index(table_dir: Path, name: str) -> list | None:
    """A built table's index for the chunk file `name` in `table_dir`, or None if its table
    there is not built (a processed file, or nothing)."""
    idx = index_path(table_dir, name)
    return orjson.loads(idx.read_bytes()) if idx.exists() else None


def place_one_file(built_dir: Path, table_dir: Path, name: str) -> None:
    """The pipeline's process step for a built table: its file and then its index moved
    from beside the chunk (`built_dir`, the set's `{table}/`) to where the processed file
    goes (`table_dir/name`). Resumes: a table whose index is in place already is left."""
    if index_path(table_dir, name).exists():
        return
    src_index = index_path(built_dir, name)
    if not src_index.exists():
        raise RuntimeError(f"{built_dir.name} {name}: no built table in {built_dir}: split again")
    table_dir.mkdir(parents=True, exist_ok=True)
    if (built_dir / name).exists():
        _move(built_dir / name, table_dir / name)
    _move(src_index, index_path(table_dir, name))


def _move(src: Path, dst: Path) -> None:
    try:
        src.replace(dst)
    except OSError:  # another file system: copied, then renamed into place
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        shutil.copyfile(src, tmp)
        tmp.replace(dst)
        src.unlink()


def write_built_sidecar(table: Table, table_dir: Path, name: str, audit_dir: Path) -> None:
    """The partition step's sidecar for a built table (its group partitioned at its close,
    push.core.merge_group): write_stats_sidecar's columns, a row a key, from the index.
    Each row's `path` is the one file, its `file_size` the key's row groups' bytes, the
    file's other bytes (its magic and footer) added to the first key's, so a chunk's sizes
    add up to its file's bytes (push.groups.chunk_partition_bytes). The rows are in the
    order write_stats_sidecar's are (Path order of each key's would-be file: the keys'
    order); a table with no rows has no file and no rows."""
    keys = built_index(table_dir, name)
    f = table_dir / name
    rows = [[str(f), n, size, key, lo, hi] for key, n, lo, hi, size, *_ in keys]
    if rows:
        # (the index counts the magic with the first key's row groups, the footer with the
        # last's: both to the first)
        with open(f, "rb") as fh:
            fh.seek(-8, os.SEEK_END)
            rows[-1][2] -= int.from_bytes(fh.read(4), "little") + 8
        rows[0][2] += f.stat().st_size - sum(r[2] for r in rows)
    rows.sort(key=lambda r: Path(r[3]))
    _write_sidecar_rows(PARTITION_COLS[table], [tuple(r) for r in rows], source=Path(name), log_dir=audit_dir)
