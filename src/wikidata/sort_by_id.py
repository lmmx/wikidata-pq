"""Sort each table's repo on the Hub by id, once it is compacted.

Compaction leaves each key's rows in source chunk order, which runs through the id space
several times, so a reader looking up an id can skip no file and few row groups. This
stage sorts each key's rows by its sort column (`id`, or `ref` for claims_labels) in
string order, stably so each id's rows keep their order, across all the key's files, and
declares the order in each row group (`sorting_columns`). Files are renamed
`part-{i}-of-{n}.parquet` (docs/journal/2026-09-30-sort-by-id.md).

A table goes through these stages, recorded in `sort.jsonl` in the state dir:

- sourced: HUB_COPY_DIR/{table} has exactly the compacted files (compaction's manifest),
  by size and sha256; the emptied compaction source directory is then removed. (When
  compaction uploaded its files, before 2026-10-05, it is checked against the Hub's.)
- written: each key sorted to SORT_DIR/out/{table}, checked, and listed in the manifest.
  How each key is sorted and laid out is planned from its data alone (plan_key: its
  rows, and its decoded bytes estimated from its footers), not from its files' sizes.
  A key over SORT_IN_MEMORY_BYTES so estimated (claims) goes through id-range buckets: each source
  file's row groups planned (each read directly by the buckets of most of its rows, the
  rest spilled to fragments), and each bucket sorted into the key's part files, planned
  before the sort (consecutive buckets each): a part file of one bucket written by its
  sort, one of several glued from its buckets sorted to files of the Hub's settings
  (their encoded row groups copied, not decoded) as soon as they are all sorted, while
  the others are; each step FINALISE_WORKERS at once and resumable
  (docs/journal/2026-10-05-sort-speed.md, docs/journal/2026-10-25-sort-into-parts.md)
- committed: each key's part files added and its old files (group files, or compacted
  files uploaded before 2026-10-05) deleted in one commit
- verified: the Hub has exactly the part files of every key
- done: the metadata JSON rewritten, and HUB_COPY_DIR/{table} holds the sorted files
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
import polars as pl
import pq_glue
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
from huggingface_hub import (
    CommitOperationAdd,
    CommitOperationDelete,
    HfApi,
    snapshot_download,
)
from tqdm import tqdm

from .compact import (
    _check_glued,
    _same,
    _write_file_parallel,
)
from .compact import _src_dir as _compact_src_dir
from .compact import last_stage as compact_stage
from .compact import read_manifest as compact_manifest
from .config import (
    HUB_REVISION,
    CLEAN_UP_LOCAL,
    COMPACT_COMMIT_MAX_ADDS,
    COMPACT_COMMIT_MAX_OPS,
    COMPACT_DOWNLOAD_WORKERS,
    COMPACT_FILE_BYTES,
    COMPACT_ROW_GROUP_BYTES,
    DATASET_CARDS_METADATA,
    HUB_COPY_DIR,
    SORT_BUCKET_BYTES,
    SORT_BUCKET_WRITE_BYTES,
    SORT_DIR,
    FINALISE_LARGE_WORKERS,
    FINALISE_WORKERS,
    SORT_HUB_BYTES_PER_DECODED,
    SORT_IN_MEMORY_BYTES,
    SORT_STRING_BYTES,
    Table,
)
from .parallel import in_parallel
from .push.core import _sha256

STAGES = ["sourced", "written", "committed", "verified", "done"]

SORT_COLUMN = {Table.CLAIMS_LABELS: "ref"}

# A key's file on the Hub, compacted or sorted
FILE_RE = re.compile(r"^([^/]+)/((?:chunks-\d+-\d+|part-\d+-of-\d+)\.parquet)$")


def sort_column(table: Table) -> str:
    return SORT_COLUMN.get(table, "id")


def _ledger(state_dir: Path) -> Path:
    return state_dir / "sort.jsonl"


def record_stage(state_dir: Path, table: Table, stage: str) -> None:
    assert stage in STAGES, stage
    path = _ledger(state_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps({"table": str(table), "stage": stage}) + "\n")


def last_stage(state_dir: Path, table: Table) -> str | None:
    path = _ledger(state_dir)
    if not path.exists():
        return None
    stages = [json.loads(line) for line in path.read_text().splitlines() if line]
    return next((r["stage"] for r in reversed(stages) if r["table"] == table), None)


def _src_dir(table: Table) -> Path:
    return HUB_COPY_DIR / table


def _out_dir(table: Table) -> Path:
    return SORT_DIR / "out" / table


def _bucket_dir(table: Table, key: str) -> Path:
    return SORT_DIR / "buckets" / table / key


def _manifest_path(table: Table) -> Path:
    return _out_dir(table) / "manifest.jsonl"


def _part_name(i: int, n: int) -> str:
    w = len(str(n))
    return f"part-{i:0{w}d}-of-{n:0{w}d}.parquet"


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _append_jsonl(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def _name_order(p: Path) -> list[int | str]:
    """Name order with digit runs compared as numbers, as a release's group names can
    mix widths (see compact.FILE_RE)."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p.name)]


def _local_keys(table: Table) -> dict[str, list[Path]]:
    """Each key's files in the local copy, in name order (chunk order for compacted
    files)."""
    keys: dict[str, list[Path]] = {}
    for p in sorted(_src_dir(table).glob("*/*.parquet"), key=_name_order):
        keys.setdefault(p.parent.name, []).append(p)
    return keys


def _remote_files(repo_id: str, api: HfApi) -> dict[str, dict[str, object]]:
    """Each key's Parquet files on the Hub, by name."""
    keys: dict[str, dict[str, object]] = {}
    for info in api.list_repo_tree(repo_id, repo_type="dataset", recursive=True, revision=HUB_REVISION):
        if m := FILE_RE.match(info.path):
            keys.setdefault(m.group(1), {})[m.group(2)] = info
    return keys


def source_table(table: Table, repo_id: str, api: HfApi) -> None:
    """Bring the local copy up to date with the Hub, and check it has exactly the Hub's
    files: the sorted files are built from it."""
    print(f"[sort] {table}: checking {_src_dir(table)} against {repo_id}", flush=True)
    snapshot_download(
        repo_id,
        repo_type="dataset",
        revision=HUB_REVISION,
        local_dir=_src_dir(table),
        allow_patterns="*/*.parquet",
        max_workers=COMPACT_DOWNLOAD_WORKERS,
    )
    remote = _remote_files(repo_id, api)
    local = _local_keys(table)
    if set(remote) != set(local):
        raise RuntimeError(f"[sort] {table}: local keys differ from the Hub's")
    for key, files in tqdm(local.items(), desc=f"[sort] {table}: hashing", unit="key"):
        if {f.name for f in files} != set(remote[key]):
            raise RuntimeError(f"[sort] {table}/{key}: local files differ from the Hub's")
        for f in files:
            if not _same(remote[key][f.name], f, _sha256(f)):
                raise RuntimeError(f"[sort] {table}/{key}/{f.name} differs from the Hub's")
    print(f"[sort] {table}: local copy matches the Hub", flush=True)


def _uploaded_by_compaction(state_dir: Path, table: Table) -> bool:
    """Compaction uploaded the table's files (before 2026-10-05), so the Hub has them."""
    path = state_dir / "compact.jsonl"
    if not path.exists():
        return False
    lines = (json.loads(line) for line in path.read_text().splitlines() if line)
    return any(r == {"table": str(table), "stage": "committed"} for r in lines)


def check_compacted_copy(table: Table) -> None:
    """The local copy has exactly the files of compaction's manifest, by size and sha256
    (hashed FINALISE_WORKERS files at once): the sorted files are built from it."""
    print(f"[sort] {table}: checking {_src_dir(table)} against the compaction manifest", flush=True)
    manifest = compact_manifest(table)
    local = _local_keys(table)
    if set(local) != set(manifest):
        raise RuntimeError(f"[sort] {table}: local keys differ from the compacted keys")
    want = {}
    for key, files in local.items():
        entries = {f["name"]: f for f in manifest[key]["files"]}
        if {f.name for f in files} != set(entries):
            raise RuntimeError(f"[sort] {table}/{key}: local files differ from the compacted files")
        want.update({f: entries[f.name] for f in files})
    with ThreadPoolExecutor(FINALISE_WORKERS) as pool:
        hashes = pool.map(_sha256, want)
        bar = tqdm(zip(want, hashes), total=len(want), desc=f"[sort] {table}: hashing", unit="file")
        for f, sha in bar:
            if f.stat().st_size != want[f]["bytes"] or sha != want[f]["sha256"]:
                raise RuntimeError(f"[sort] {table}/{f.parent.name}/{f.name} differs from the compacted file")
    print(f"[sort] {table}: local copy matches the compacted files", flush=True)


# Checks


def _schema(sources: list[Path]) -> pa.Schema:
    schema = pq.read_schema(sources[0]).remove_metadata()
    for src in sources[1:]:
        if not pq.read_schema(src).remove_metadata().equals(schema):
            raise RuntimeError(f"[sort] {src} has another schema")
    return schema


def _check_sort(what: str, t: pa.Table, idx: pa.Array, st: pa.Table, col: str) -> None:
    """`st` is `t` stably sorted by `col` through the sort indices `idx`: `idx` takes each
    of `t`'s rows once, `st`'s sort column is `t`'s taken by `idx` and ascends (nulls
    last), and the rows of one value keep their order in `t`. Checked on the indices and
    the sort column; the other columns are taken by the same indices (pyarrow's take),
    and the files written from `st` are checked against it row for row (_check_written)."""
    n = t.num_rows
    order = idx.to_numpy().astype(np.int64)  # (uint64: a step back would wrap around)
    if len(order) != n or st.num_rows != n or (n and not (np.bincount(order, minlength=n) == 1).all()):
        raise RuntimeError(f"[sort] {what}: the sort is not a permutation of the rows")
    ids = st.column(col).combine_chunks()
    if not ids.equals(t.column(col).take(idx).combine_chunks()):
        raise RuntimeError(f"[sort] {what}: the sorted {col} column is not the rows' own")
    if n < 2:
        return
    nulls = pc.is_null(ids).to_numpy(zero_copy_only=False)
    if nulls.any() and not nulls[int(np.argmax(nulls)):].all():
        raise RuntimeError(f"[sort] {what}: a null {col} before a value")
    prev, cur = ids.slice(0, n - 1), ids.slice(1)
    if not pc.all(pc.less_equal(prev, cur)).as_py():
        raise RuntimeError(f"[sort] {what}: not sorted by {col}")
    same = pc.or_(pc.fill_null(pc.equal(prev, cur), False), pc.and_(pc.is_null(prev), pc.is_null(cur)))
    if not (np.diff(order)[same.to_numpy(zero_copy_only=False)] > 0).all():
        raise RuntimeError(f"[sort] {what}: rows of one {col} changed their order (not a stable sort)")


def _release() -> None:
    """Return the pages pyarrow's pool (mimalloc) holds free to the OS. Memory freed on
    one thread that another allocated (a row group decoded on Arrow's threads, freed on
    this one) is not reused by this thread's allocations, so without this a bucket's
    worker held each stage's freed pages into the next: its peak RSS was about three
    times its rows' Arrow bytes, at the sorted bucket's write. A call costs about a
    microsecond when there is nothing to return. The allocator is unchanged
    (docs/journal/2026-10-25-finalise-memory.md)."""
    pa.default_memory_pool().release_unused()


def _check_written(
    what: str, outputs: list[Path], t: pa.Table, schema: pa.Schema, threads: bool = True
) -> None:
    """The files have the schema and hold exactly `t`'s rows, in order: each row group
    read back by pyarrow (another implementation than the writer's, arrow-rs) and compared
    with its rows of `t`, value for value, then freed, so the check holds a row group at
    a time. Read on Arrow's threads if `threads`, its pages then returned (_release), or
    else on this thread, whose next row group reuses them: a bucket's worker, one of
    FINALISE_WORKERS, spent less CPU time so, faulting in no fresh pages."""
    at = 0
    for out in outputs:
        f = pq.ParquetFile(out)
        if not f.schema_arrow.remove_metadata().equals(schema):
            raise RuntimeError(f"[sort] {what}: {out.name} has another schema")
        for g in range(f.metadata.num_row_groups):
            got = f.read_row_group(g, use_threads=threads)
            if not got.equals(t.slice(at, got.num_rows)):
                raise RuntimeError(f"[sort] {what}: {out.name} row group {g} differs from the sorted rows")
            at += got.num_rows
            del got
            if threads:
                _release()  # (decoded on Arrow's threads: their pages kept otherwise)
    if at != t.num_rows:
        raise RuntimeError(f"[sort] {what}: the files hold {at} rows, {t.num_rows} sorted")


def _check_schemas(what: str, outputs: list[Path], schema: pa.Schema) -> None:
    for out in outputs:
        if not pq.read_schema(out).remove_metadata().equals(schema):
            raise RuntimeError(f"[sort] {what}: {out.name} has another schema")


def _check_sorted(what: str, outputs: list[Path], col: str, schema: pa.Schema) -> None:
    """Each file has the schema, and the sort column ascends within and across files."""
    prev = None
    for out in outputs:
        if not pq.read_schema(out).remove_metadata().equals(schema):
            raise RuntimeError(f"[sort] {what}: {out.name} has another schema")
        s = pl.read_parquet(out, columns=[col])[col]
        if not s.is_sorted():
            raise RuntimeError(f"[sort] {what}: {out.name} is not sorted by {col}")
        if len(s) and prev is not None and s[0] < prev:
            raise RuntimeError(f"[sort] {what}: {out.name} starts before the previous file ends")
        if len(s):
            prev = s[-1]


# Planning: how a key is sorted and laid out, from its data alone
#
# Every decision that shapes the published files (sorted in memory or through buckets,
# the bucket count, the part count and which buckets make each part, the rows a row
# group) is taken from the key's rows and decoded bytes, never from the bytes its files
# happen to be encoded in, so the same rows give the same files whichever writer, encoding
# or file layout (group files linked or glued, one partition file or several) brought them
# here, and in whatever order their rows came within an id range. Before 2026-10-25 these
# were taken from the source files' sizes and from a sample row group (the largest file's
# first), which a re-encoding or a reordering of the same rows changed
# (docs/journal/2026-10-25-decoupled-thresholds.md).

# The planning rule, recorded with a bucketed key's boundaries: a key bucketed by
# another rule is bucketed again
PLAN = "decoded-1"

# A leaf value's nominal decoded width by physical type; a string's is SORT_STRING_BYTES
_NOMINAL_BYTES = {"BOOLEAN": 1, "INT32": 4, "FLOAT": 4, "INT64": 8, "DOUBLE": 8, "INT96": 12}


def _nominal_width(column: pq.ColumnSchema) -> int:
    if column.physical_type == "BYTE_ARRAY":
        return SORT_STRING_BYTES
    if column.physical_type == "FIXED_LEN_BYTE_ARRAY":
        return column.length
    return _NOMINAL_BYTES[column.physical_type]


def footer_size(sources: list[Path]) -> tuple[int, int]:
    """The files' rows, and their decoded bytes estimated from their footers alone: each
    leaf column chunk's values (`num_values`, its definition and repetition levels: one a
    value, null or empty list) at its type's nominal width. Both are sums over the row
    groups of counts that depend only on the rows and the schema: the same whichever
    writer, encoding or compression wrote the files (on the benchmark's claims, pyarrow
    with and without dictionaries, Polars and pq-glue's arrow-rs wrote the same counts,
    and uncompressed sizes 2.7 times apart), however the rows are split into files and
    row groups (a glued file has its sources' row groups), and in any order."""
    rows = decoded = 0
    for src in sources:
        meta = pq.read_metadata(src)
        widths = [_nominal_width(meta.schema.column(c)) for c in range(meta.num_columns)]
        rows += meta.num_rows
        for g in range(meta.num_row_groups):
            rg = meta.row_group(g)
            decoded += sum(rg.column(c).num_values * w for c, w in enumerate(widths))
    return rows, decoded


@dataclass(frozen=True)
class KeyPlan:
    """How a key is sorted and laid out (plan_key)."""

    rows: int
    decoded: int  # bytes, estimated from the footers (footer_size)
    in_memory: bool  # sorted in memory (sort_in_memory), else through buckets
    larger: bool  # sorted in memory in a pool of its own (an execution choice alone)
    buckets: int  # id-range buckets of equal rows (_boundaries), if not in memory
    parts: int  # part files

    def rows_for(self, nbytes: int) -> int:
        """The rows of about `nbytes` decoded bytes, at the key's mean (for execution
        choices: flushes, memory estimates)."""
        return max(1, int(nbytes * self.rows / max(self.decoded, 1)))

    def part_rows(self) -> list[int]:
        """An in-memory key's rows a part file: `parts` files of equal rows, the last the
        rest."""
        per = math.ceil(self.rows / self.parts) if self.rows else 0
        return [min(per, self.rows - i * per) for i in range(self.parts)]

    def pack(self, bucket_rows: list[int]) -> list[list[int]]:
        """A bucketed key's part files, each consecutive buckets: `parts` files (or one a
        bucket, if fewer), cut where the buckets' cumulative rows are nearest each
        multiple of rows / parts. The buckets' rows are their ids' rows counted
        (_boundaries), so the cuts are known before any bucket is sorted."""
        total = sum(bucket_rows)
        n = max(1, min(self.parts, len(bucket_rows)))
        cum = np.cumsum(bucket_rows)
        cuts = [0]
        for k in range(1, n):
            c = int(np.argmin(np.abs(cum - total * k / n))) + 1
            if cuts[-1] < c < len(bucket_rows):
                cuts.append(c)
        cuts.append(len(bucket_rows))
        return [list(range(a, b)) for a, b in zip(cuts, cuts[1:])]


def plan_key(table: Table, sources: list[Path]) -> KeyPlan:
    """A key's plan, from its files' footers alone (footer_size), so the same from its
    group files as from the files compacted from them:

    - in memory if its decoded bytes are at most SORT_IN_MEMORY_BYTES, in a pool of its
      own (FINALISE_LARGE_WORKERS at once, a worker each) if over 3/16 of it;
    - else through `ceil(decoded / SORT_BUCKET_BYTES)` buckets;
    - `ceil(decoded × its table's Hub bytes a decoded byte / COMPACT_FILE_BYTES)` part
      files (SORT_HUB_BYTES_PER_DECODED), at most one a row.

    The rows a row group are taken from the rows themselves as they are written
    (row_group_rows)."""
    rows, decoded = footer_size(sources)
    in_memory = decoded <= SORT_IN_MEMORY_BYTES
    hub = decoded * SORT_HUB_BYTES_PER_DECODED[str(table)]
    return KeyPlan(
        rows=rows,
        decoded=decoded,
        in_memory=in_memory,
        larger=in_memory and decoded > SORT_IN_MEMORY_BYTES * 3 // 16,
        buckets=max(1, math.ceil(decoded / SORT_BUCKET_BYTES)),
        parts=max(1, min(math.ceil(hub / COMPACT_FILE_BYTES), rows)),
    )


def sorted_in_memory(table: Table, sources: list[Path]) -> bool:
    """Whether the key of these files (group files or compacted files: the same answer)
    is sorted in memory."""
    return plan_key(table, sources).in_memory


def decoded_bytes(data: pa.Table | pa.ChunkedArray | pa.Array) -> int:
    """The rows' decoded bytes, exactly, as their values would take in Arrow's standard
    layout: a fixed-width value its width (a boolean a byte), a string or binary value its
    bytes and a 4-byte offset (8 for the large types), a list its offset and its values',
    a struct its fields'; validity bitmaps not counted. A function of the values and the
    schema alone: not of how they are chunked or sliced, which reader decoded them
    (pyarrow and pq-glue allocate validity bitmaps differently, so `nbytes` differed), or
    their order."""
    if isinstance(data, pa.Table):
        return sum(decoded_bytes(c) for c in data.columns)
    if isinstance(data, pa.ChunkedArray):
        return sum(decoded_bytes(c) for c in data.chunks)
    t, n = data.type, len(data)
    if pa.types.is_dictionary(t):
        return decoded_bytes(data.dictionary_decode())
    if pa.types.is_null(t):
        return 0
    if pa.types.is_boolean(t):
        return n
    if pa.types.is_struct(t):
        return sum(decoded_bytes(data.field(i)) for i in range(t.num_fields))
    if pa.types.is_fixed_size_list(t):
        return decoded_bytes(data.flatten())
    if pa.types.is_list(t) or pa.types.is_large_list(t) or pa.types.is_map(t):
        width = 8 if pa.types.is_large_list(t) else 4
        if not n:
            return 0
        offsets = data.offsets
        lo, hi = offsets[0].as_py(), offsets[n].as_py()
        return width * n + decoded_bytes(data.values.slice(lo, hi - lo))
    if pa.types.is_string(t) or pa.types.is_binary(t) or pa.types.is_large_string(t) or pa.types.is_large_binary(t):
        large = pa.types.is_large_string(t) or pa.types.is_large_binary(t)
        if not n:
            return 0
        offsets = np.frombuffer(data.buffers()[1], dtype=np.int64 if large else np.int32)
        return (8 if large else 4) * n + int(offsets[data.offset + n] - offsets[data.offset])
    if pa.types.is_string_view(t) or pa.types.is_binary_view(t):
        return 4 * n + int(pc.sum(pc.binary_length(data)).as_py() or 0)
    return n * (t.bit_width // 8)


def row_group_rows(t: pa.Table, nbytes: int | None = None) -> int:
    """Rows a row group for about COMPACT_ROW_GROUP_BYTES of decoded rows, from the rows
    being written (a key sorted in memory, or a bucket): their decoded bytes a row,
    exactly (decoded_bytes). A reader holds a row group in memory, so it is sized by its
    decoded size, not its Parquet size: claims rows take several times more bytes in
    memory than in uncompressed Parquet. (`nbytes`: another target, as a job is given its
    parent's.)"""
    nbytes = nbytes or COMPACT_ROW_GROUP_BYTES
    return max(1, int(nbytes * t.num_rows / max(decoded_bytes(t), 1)))


# Writing


def _read(sources: list[Path]) -> pa.Table:
    """A key's files read one at a time and concatenated. Not read as one dataset, which
    casts to a unified schema: casting corrupts the nested structs of claims (see
    compact._source_batches)."""
    return pa.concat_tables(_read_file(f) for f in sources)


def _read_file(path: Path) -> pa.Table:
    """A file's rows, read by pq-glue (arrow-rs, its row groups decoded in parallel): the
    values and schema pyarrow reads, in a fifth to a half of the CPU time on claims in a
    fresh worker process, where pyarrow's reads spent most of theirs in the kernel taking
    memory (lmmx/pq-glue#4). The schema's metadata is dropped, as the writes take the
    key's schema without it."""
    return pq_glue.read_parquet(path).replace_schema_metadata()


def _sorted(rows: list[pa.Table], col: str, what: str) -> pa.Table:
    """The table in `rows` (taken out of the list, so the caller holds no reference to
    it) stably sorted by `col`, the sort checked (_check_sort). The same table as
    `t.take` of its sort indices, taken one column at a time: each column combined into
    one array (a take from chunks combines a nested column anyway, inside the take), its
    chunks then freed, and taken. So each sorted column reuses the pages of the column it
    replaces, and the sort holds about one copy of the rows and a column, not two copies
    and a combined column per thread. Each column's freed pages are returned to the OS
    (_release), as are the rows' once taken, so the write that follows does not start on
    top of them. Before 2026-10-25, with a thread a column, a bucket's take spent most of
    its time in the kernel faulting in fresh pages, and held its rows and their sorted
    copy at once. _check_sort reads only the row count and the sort column of the rows,
    so it gets those."""
    t = rows.pop()
    idx = pc.sort_indices(t, [(col, "ascending")])  # a stable sort
    schema, ids = t.schema, t.column(col)
    columns = t.columns
    del t
    # (by index, not `enumerate`, whose result tuple holds the column it last gave until
    # it gives the next: the chunks of the column being taken were held through its take)
    for k in range(len(columns)):
        c, columns[k] = columns[k], None
        a = c.chunk(0) if c.num_chunks == 1 else c.combine_chunks()
        del c
        _release()
        columns[k] = a.take(idx)
        del a
    _release()
    st = pa.Table.from_arrays(columns, schema=schema)
    _check_sort(what, pa.Table.from_arrays([ids], schema=pa.schema([schema.field(col)])), idx, st, col)
    return st


def _sorting(schema: pa.Schema, col: str) -> list:
    return [pq.SortingColumn(schema.get_field_index(col))]


def _entry(dst: Path, rows: int) -> dict:
    return {"name": dst.name, "rows": rows, "bytes": dst.stat().st_size, "sha256": _sha256(dst)}


# A key over this many bytes of Parquet is read on every core (Arrow decoding its
# columns in threads), as a large one is often read once the other workers are done (the
# benchmark's claims key, 70 MB, alone in its table). The rest are read on one thread:
# a small key's columns are too small to share out, and the shared pool's other workers
# have keys of their own (the quick tier's ~2,300 small keys are each under 1 MB)
_THREADED_READ_BYTES = 16 * 1024**2


def _read_key(sources: list[Path], threads: bool) -> tuple[pa.Schema, pa.Table]:
    """A key's schema (as _schema) and rows (as _read), each file opened once, not twice:
    on the benchmark's quick tier, whose ~2,300 small keys have a file each, the schema
    read apart was a share of a key's time, and pq.read_table (a dataset scan) took 1.5
    to 2 times as long as ParquetFile.read. If `threads` (a large key) the files are read
    by pq-glue on every core (_read_file), else by pyarrow on this thread. The same values
    as _read either way, so the same sorted files. (Before 2026-10-25 it also read the
    largest file's first row group, a sample for the rows a row group; they are now taken
    from the rows themselves, row_group_rows.)"""
    files = [pq.ParquetFile(src) for src in sources]
    schema = files[0].schema_arrow.remove_metadata()
    for src, f in zip(sources[1:], files[1:]):
        if not f.schema_arrow.remove_metadata().equals(schema):
            raise RuntimeError(f"[sort] {src} has another schema")
    if threads:
        t = pa.concat_tables(_read_file(src) for src in sources)
    else:
        t = pa.concat_tables(f.read(use_threads=False) for f in files)
    return schema, t


def sort_in_memory(table: Table, key: str, sources: list[Path], plan: KeyPlan | None = None) -> list[dict]:
    """Sort a key read whole into its plan's part files of equal rows (plan_key,
    KeyPlan.part_rows), in row groups of about COMPACT_ROW_GROUP_BYTES of its decoded
    rows (row_group_rows), the sort checked and the files checked against the sorted
    rows."""
    plan = plan or plan_key(table, sources)
    col = sort_column(table)
    what = f"{table}/{key}"
    size = sum(s.stat().st_size for s in sources)
    schema, *rows = _read_key(sources, threads=size > _THREADED_READ_BYTES)  # rows: [the key's rows]
    if rows[0].num_rows != plan.rows:
        raise RuntimeError(f"[sort] {what}: {rows[0].num_rows} rows read, {plan.rows} planned")
    t = _sorted(rows, col, what)
    rg_rows = row_group_rows(t)
    dst_dir = _out_dir(table) / key
    dst_dir.mkdir(parents=True, exist_ok=True)
    files, at = [], 0
    for i, n in enumerate(plan.part_rows()):
        dst = dst_dir / _part_name(i, plan.parts)
        part = t.slice(at, n)
        at += n
        rows = _write_file_parallel(
            dst, schema, part.to_batches(max_chunksize=rg_rows), rg_rows, _sorting(schema, col)
        )
        files.append(_entry(dst, rows))
    _check_written(what, [dst_dir / f["name"] for f in files], t, schema)
    return files


# The bucket path, for a key too large to sort in memory


def _boundaries(sources: list[Path], col: str, n: int) -> tuple[list[str], list[int]]:
    """The first id of each bucket but the first, splitting the key's rows into `n`
    buckets of about equal rows, never splitting an id; and each bucket's rows, the key's
    rows of its ids counted."""
    counts = (
        pl.scan_parquet(sources)
        .group_by(col)
        .len()
        .sort(col)
        .collect(engine="streaming")
    )
    cum = counts["len"].cum_sum().to_numpy()
    total = int(cum[-1])
    idx = np.searchsorted(cum, [total * i / n for i in range(1, n)], side="left") + 1
    idx = sorted(set(int(i) for i in idx if 0 < i < len(counts)))
    ends = [int(cum[i - 1]) for i in idx] + [total]
    rows = [b - a for a, b in zip([0, *ends], ends)]
    return counts[col].gather(idx).to_list(), rows


# A bucket's fragments are Arrow IPC, lz4-compressed: scratch files, written once and
# read once, so not encoded as Parquet (on claims, IPC took under half the CPU time to
# write and a fifth to read back). Each flush is one IPC stream, read back and compared
# with its rows before its bytes are appended to the fragment, and the fragment's bytes
# digested, so the sort reads exactly the rows bucketed. lz4, not zstd: on claims it took
# half the CPU time to compress and two thirds to decompress (each flush is decompressed
# twice, read back and sorted), at 2.2 times the size ("arrows.lz4"; zstd, "arrows",
# before 2026-10-25; IPC files without the checks, "arrow", before 2026-10-24)
FRAGMENT_FORMAT = "arrows.lz4"

# How a key is bucketed: since 2026-10-25 ("direct"), a bucket reads most of its rows
# from the source row groups themselves, and only the rest are spilled to its fragments
# (_direct); before, every row was spilled (the fragments' format alone)
BUCKETS_FORMAT = f"direct.{FRAGMENT_FORMAT}"

# A row spilled to a fragment costs about this many reads of it: it is routed, written,
# read back and compared, digested, and read again by the sort (on the benchmark's claims,
# about 2.0 to 2.2 times reading its row group, with lz4)
_SPILL_READS = 2

# A bucket reads a row group directly only for at least this share of its rows, so no
# bucket reads more than 16 times its rows: a property's few rows (`P…`, all in the first
# bucket) among a chunk's items are spilled, not read with each of the thousands of row
# groups they are in
_DIRECT_SHARE = 1 / 16


def _direct(counts: dict[int, int]) -> list[int]:
    """The buckets that read a row group directly, of the buckets its rows are in
    (`counts`, its rows by bucket): the k holding the most rows (each at least
    _DIRECT_SHARE of them), k chosen for the least reading. Each of the k reads the whole
    row group; if any of its rows are left, they are spilled, one more read and
    _SPILL_READS a row spilled. The dump is nearly in id order, so most row groups' ids
    are of one narrow range, in one or two buckets, and read by each; a few rows far
    from the rest (a property's among items) are spilled; and a row group of the
    earliest ids, or of ids of fewer digits (which interleave in string order with the
    many longer ids of their prefix), whose rows spread over many buckets, is spilled
    whole, as every row was before."""
    order = sorted(counts, key=lambda b: (-counts[b], b))
    rows = sum(counts.values())
    best, k, kept = 1 + _SPILL_READS, 0, 0
    for i, b in enumerate(order, 1):
        if counts[b] < _DIRECT_SHARE * rows:
            break
        kept += counts[b]
        cost = i + (1 + _SPILL_READS * (rows - kept) / rows if kept < rows else 0)
        if cost <= best:
            best, k = cost, i
    return sorted(order[:k])


def _ipc_stream(t: pa.Table, schema: pa.Schema) -> pa.Buffer:
    sink = pa.BufferOutputStream()
    with ipc.new_stream(sink, schema, options=ipc.IpcWriteOptions(compression="lz4")) as w:
        w.write_table(t)
    return sink.getvalue()


def _read_streams(data, schema: pa.Schema) -> pa.Table:
    """The rows of consecutive IPC streams (a fragment, or one flush of it)."""
    reader, size, parts = pa.BufferReader(data), len(data), []
    while reader.tell() < size:
        parts.append(ipc.open_stream(reader).read_all())
    return pa.concat_tables(parts) if parts else schema.empty_table()


def _digest(data) -> str:
    return hashlib.blake2b(data, digest_size=8).hexdigest()


def _fragment_name(i: int, j: int, n_sources: int, ext: str = FRAGMENT_FORMAT) -> str:
    """Bucket i's rows from source file j, the source index padded to the digits of the
    key's last source index."""
    return f"bucket-{i:05d}-{j:0{len(str(n_sources - 1))}d}.{ext}"


def _bucket_source(
    table: Table,
    src: Path,
    j: int,
    n_sources: int,
    bdir: Path,
    bounds: list[str],
    schema: pa.Schema,
    flush_rows: int,
) -> dict:
    """Plan source file j's row groups and spill the rows its buckets do not read
    directly: each row group's ids read and counted by bucket, its buckets that read it
    directly chosen (_direct), and its other rows, if any, read and written to their
    buckets' fragments of file j, in source order, each bucket's rows buffered to record
    batches of about `flush_rows` rows, each checked as written (FRAGMENT_FORMAT). Returns
    its rows per bucket, each row group's rows per bucket and direct buckets, and the
    digest of each fragment written (none for a bucket it spilled no rows to) (run in a
    worker process)."""
    col = sort_column(table)
    lookup = pl.Series(bounds, dtype=pl.String)
    n = len(bounds) + 1
    paths = [bdir / _fragment_name(i, j, n_sources) for i in range(n)]
    outs: dict[int, BinaryIO] = {}
    hashes = {}
    pending: list[list[pa.Table]] = [[] for _ in range(n)]
    pending_rows = [0] * n
    rows = np.zeros(n, dtype=np.int64)
    groups = []

    def flush(i: int) -> None:
        if pending[i]:
            # one batch, not one per slice (a few rows each when there are many buckets)
            batch = pa.concat_tables(pending[i]).combine_chunks()
            data = _ipc_stream(batch, schema)
            if not _read_streams(data, schema).equals(batch):
                raise RuntimeError(f"[sort] {table}: {paths[i].name} reads back other rows than written")
            if i not in outs:
                outs[i], hashes[i] = open(paths[i], "wb"), hashlib.blake2b(digest_size=8)
            outs[i].write(data)
            hashes[i].update(data)
        pending[i], pending_rows[i] = [], 0

    pf = pq.ParquetFile(src)
    try:
        for g in range(pf.metadata.num_row_groups):
            ids = pf.read_row_group(g, columns=[col]).column(col)
            b = lookup.search_sorted(pl.from_arrow(ids).cast(pl.String), side="right").to_numpy()
            counts = np.bincount(b, minlength=n)
            present = np.flatnonzero(counts)
            direct = _direct({int(i): int(counts[i]) for i in present})
            groups.append({"buckets": present.tolist(), "rows": counts[present].tolist(), "direct": direct})
            rows += counts
            if len(direct) == len(present):
                continue
            rg = pf.read_row_group(g)
            if not rg.column(col).equals(ids):
                raise RuntimeError(f"[sort] {table}: {src.name} row group {g} has other ids than counted")
            spill = np.flatnonzero(~np.isin(b, direct))
            order = spill[np.argsort(b[spill], kind="stable")]  # each bucket's rows in source order
            rg, b = rg.take(pa.array(order)), b[order]
            starts = np.flatnonzero(np.r_[True, b[1:] != b[:-1]])
            for s, e in zip(starts, np.r_[starts[1:], len(b)]):
                i = int(b[s])
                pending[i].append(rg.slice(s, e - s))
                pending_rows[i] += int(e - s)
                if pending_rows[i] >= flush_rows:
                    flush(i)
        for i in range(n):
            flush(i)
    finally:
        for f in outs.values():
            f.close()
    if rows.sum() != pf.metadata.num_rows:
        raise RuntimeError(f"[sort] {table}: {src.name} bucketed {rows.sum()} of its {pf.metadata.num_rows} rows")
    digests = [hashes[i].hexdigest() if i in hashes else None for i in range(n)]
    return {
        "source": src.name,
        "rows": rows.tolist(),
        "row_groups": groups,
        "digests": digests,
        "format": BUCKETS_FORMAT,
        "at": time.time(),
    }


def bucket_key(table: Table, key: str, sources: list[Path], plan: KeyPlan | None = None) -> dict:
    """Plan each of the key's row groups and spill the rows its buckets do not read
    directly (_bucket_source), each source file to its own fragments, FINALISE_WORKERS
    files at once, into the plan's buckets (plan_key). The boundaries and each bucket's
    rows (from the ids counted) are in `bounds.json`, each source file planned and spilled
    in `bucketed.jsonl` (reused on a restart), and the buckets and their rows in
    `buckets.json` (buckets of another format, or planned by another rule, PLAN, are made
    again). Each bucket's rows planned are its ids' rows counted, and the buckets' rows
    the key's."""
    plan = plan or plan_key(table, sources)
    bdir = _bucket_dir(table, key)
    record = bdir / "buckets.json"
    if record.exists():
        made = json.loads(record.read_text())
        if made.get("format") == BUCKETS_FORMAT and made.get("plan") == PLAN:
            return made
        shutil.rmtree(bdir)
    col = sort_column(table)
    schema = _schema(sources)
    names = [s.name for s in sources]
    bounds_path = bdir / "bounds.json"
    prior = json.loads(bounds_path.read_text()) if bounds_path.exists() else {}
    if prior.get("sources") == names and "rows" in prior and prior.get("plan") == PLAN:
        bounds, counted = prior["bounds"], prior["rows"]
    else:
        n = plan.buckets
        print(f"[sort] {table}/{key}: finding {n} bucket boundaries", flush=True)
        bounds, counted = _boundaries(sources, col, n)
        shutil.rmtree(bdir, ignore_errors=True)
        bdir.mkdir(parents=True)
        bounds_path.write_text(json.dumps({"bounds": bounds, "rows": counted, "sources": names, "plan": PLAN}))
    log = bdir / "bucketed.jsonl"
    # (a source bucketed in another format is bucketed again)
    done = {
        e["source"]: e
        for e in _read_jsonl(log)
        if e["source"] in names and e.get("format") == BUCKETS_FORMAT
    }
    flush_rows = plan.rows_for(SORT_BUCKET_WRITE_BYTES)
    jobs = {
        j: (table, src, j, len(sources), bdir, bounds, schema, flush_rows)
        for j, src in enumerate(sources)
        if src.name not in done
    }
    desc = f"[sort] {table}/{key}: bucketing"
    # One job per process: a job holds every bucket's buffer
    for _, entry in in_parallel(_bucket_source, jobs, desc, "file", len(sources), tasks_per_child=1):
        _append_jsonl(log, entry)
        done[entry["source"]] = entry
    rows = [sum(r) for r in zip(*(done[n]["rows"] for n in names), strict=True)]
    if rows != counted:
        bad = next((i for i, (a, b) in enumerate(zip(rows, counted)) if a != b), None)
        raise RuntimeError(f"[sort] {table}/{key}: rows bucketed differ from the ids counted (bucket {bad})")
    want = sum(pq.ParquetFile(s).metadata.num_rows for s in sources)
    if sum(rows) != want:
        raise RuntimeError(f"[sort] {table}/{key}: {sum(rows)} rows bucketed, {want} in the sources")
    spilled = sum(
        c
        for e in done.values()
        for g in e["row_groups"]
        for b, c in zip(g["buckets"], g["rows"])
        if b not in g["direct"]
    )
    print(f"[sort] {table}/{key}: {spilled:,} of {want:,} rows spilled to fragments", flush=True)
    buckets = {
        "bounds": bounds,
        "rows": rows,
        "sources": names,
        "fragments": len(sources),
        "format": BUCKETS_FORMAT,
        "plan": PLAN,
    }
    record.write_text(json.dumps(buckets))
    return buckets


def _bucket_parts(bdir: Path, sources: list[Path], buckets: dict, bucketed: dict) -> list[list[tuple]]:
    """Each bucket's parts, in source order: for each source file, its fragment of the
    bucket (or none), the fragment's digest, and each of the file's row groups with rows
    of the bucket, in order: its index, those rows, and whether the bucket reads it
    directly."""
    n = len(buckets["rows"])
    parts: list[list[tuple]] = [[] for _ in range(n)]
    for j, (src, name) in enumerate(zip(sources, buckets["sources"], strict=True)):
        e = bucketed[name]
        groups: list[list[tuple]] = [[] for _ in range(n)]
        for g, rg in enumerate(e["row_groups"]):
            for b, c in zip(rg["buckets"], rg["rows"], strict=True):
                groups[b].append((g, c, b in rg["direct"]))
        for i, digest in enumerate(e["digests"]):
            frag = bdir / _fragment_name(i, j, buckets["fragments"]) if digest else None
            parts[i].append((src, frag, digest, groups[i]))
    return parts


def _read_bucket(
    what: str, parts: list[tuple], schema: pa.Schema, col: str, lo: str | None, hi: str | None
) -> pa.Table:
    """A bucket's rows, in source order: each source file's row groups' rows of the bucket
    in turn, either read directly (the row group read, one at a time, and its rows of
    `col` in [lo, hi) kept, the row group as read if all of them are) or taken in order
    from the file's fragment of the bucket (checked to be the bytes bucketed). Each row
    group's rows kept are as many as it was planned to have in the bucket, and each
    fragment holds exactly the rows spilled."""
    out = []
    for src, frag, digest, groups in parts:
        spilled = schema.empty_table()
        if frag is not None:
            data = frag.read_bytes()
            if _digest(data) != digest:
                raise RuntimeError(f"[sort] {what}: {frag.name} differs from the bytes bucketed")
            spilled = _read_streams(data, schema)
        pf, at = None, 0
        for g, rows, direct in groups:
            if direct:
                pf = pf or pq.ParquetFile(src)
                rg = pf.read_row_group(g).replace_schema_metadata()
                ids = rg.column(col)
                keep = pc.is_valid(ids)
                if lo is not None:
                    keep = pc.and_(keep, pc.greater_equal(ids, lo))
                if hi is not None:
                    keep = pc.and_(keep, pc.less(ids, hi))
                # (a row group wholly the bucket's is kept as read: the filter would copy
                # every column, in the kernel's time taking memory, for the same rows)
                part = rg if (pc.sum(keep).as_py() or 0) == rg.num_rows else rg.filter(keep)
                del rg
            else:
                part, at = spilled.slice(at, rows), at + rows
            if part.num_rows != rows:
                raise RuntimeError(f"[sort] {what}: {src.name} row group {g} gave {part.num_rows} rows, {rows} planned")
            out.append(part)
        if at != spilled.num_rows:
            raise RuntimeError(f"[sort] {what}: {frag.name} holds {spilled.num_rows} rows, {at} spilled")
    return pa.concat_tables(out) if out else schema.empty_table()


def _sort_bucket(
    table: Table,
    key: str,
    parts: list[tuple],
    dst: Path,
    schema: pa.Schema,
    rg_rows: int | None,
    want_rows: int,
    lo: str | None,
    hi: str | None,
    part: bool = False,
    row_group_bytes: int | None = None,
) -> dict:
    """Sort one bucket, its rows of `col` in [lo, hi), to `dst` with the Hub's settings
    (zstd 3, a page index, content-defined chunking, sorted by `col`): a part file of the
    key's, if `part` (its entry then with its sha256), or else a scratch file to be glued
    into one as it is. Its rows read as planned (_read_bucket), as many as its ids
    counted, the sort checked (_check_sort), its ids within [lo, hi), and the file checked
    against the sorted rows (_check_written) (run in a worker process). Its row groups are
    of `rg_rows` rows, or if None of about `row_group_bytes` (COMPACT_ROW_GROUP_BYTES if
    None) of its own decoded rows (row_group_rows)."""
    col = sort_column(table)
    what = f"{table}/{key} {dst.name}"
    rows = [_read_bucket(what, parts, schema, col, lo, hi)]
    if rows[0].num_rows != want_rows:
        raise RuntimeError(f"[sort] {what}: {rows[0].num_rows} rows, {want_rows} bucketed")
    t = _sorted(rows, col, what)
    ids = t.column(col)
    if want_rows and ((lo is not None and ids[0].as_py() < lo) or (hi is not None and ids[-1].as_py() >= hi)):
        raise RuntimeError(f"[sort] {what}: {col} outside the bucket's range")
    rg_rows = rg_rows or row_group_rows(t, row_group_bytes)
    rows = _write_file_parallel(
        dst, schema, t.to_batches(max_chunksize=rg_rows), rg_rows, _sorting(schema, col)
    )
    _check_written(what, [dst], t, schema, threads=False)
    return _entry(dst, rows) if part else {"name": dst.name, "rows": rows, "bytes": dst.stat().st_size}


# A bucket sort's peak RSS, modelled: a worker's floor, and a multiple of its rows'
# decoded bytes as the key's plan estimates them (KeyPlan.decoded / rows). On the bench's
# 3x claims key (881,334 rows, 1,684 B a row so estimated, 1.48 GB), measured 2.32 GB
# locally and 2.55 GB on CI (bench 8.0.0), so estimated 2.69 GB, as before 2026-10-25,
# when the bytes a row were the largest source's first row group's (846 B, and 3.2 a
# byte) (docs/journal/2026-10-25-finalise-memory.md)
_BUCKET_FLOOR_BYTES = 300 * 1024**2
_BUCKET_PEAK_PER_BYTE = 1.6


def _bucket_memory(rows: int, plan: KeyPlan) -> int:
    """A bucket sort's estimated peak bytes, for the jobs' memory budget (in_parallel)."""
    return int(_BUCKET_FLOOR_BYTES + _BUCKET_PEAK_PER_BYTE * rows * plan.decoded / max(plan.rows, 1))


def _spill_again(
    table: Table, key: str, sources: list[Path], buckets: dict, bucketed: dict, lost: set[int], needed: set[int]
) -> dict:
    """Spill again the fragments of buckets to sort again whose fragments were removed once
    they were sorted (`lost`: a part file of theirs not listed, or other than listed,
    once its buckets' scratch files were removed; or a plan of other part files): each
    source file with a fragment of one of them bucketed again (_bucket_source), its row
    groups planned as before, and its fragments of buckets not to be sorted (not in
    `needed`) removed. Each source's entry is appended to `bucketed.jsonl`, its digests
    those of the fragments now written. Returns the sources' entries."""
    bdir = _bucket_dir(table, key)
    schema = _schema(sources)
    flush_rows = plan_key(table, sources).rows_for(SORT_BUCKET_WRITE_BYTES)
    n_sources = len(sources)
    jobs = {
        j: (table, src, j, n_sources, bdir, buckets["bounds"], schema, flush_rows)
        for j, src in enumerate(sources)
        if any(bucketed[src.name]["digests"][i] for i in lost)
    }
    print(f"[sort] {table}/{key}: {len(lost)} buckets to sort again, spilled again from {len(jobs)} files", flush=True)
    out = dict(bucketed)
    desc = f"[sort] {table}/{key}: spilling again"
    for j, entry in in_parallel(_bucket_source, jobs, desc, "file", tasks_per_child=1):
        prior = bucketed[entry["source"]]
        if (
            entry["rows"] != prior["rows"]
            or entry["row_groups"] != prior["row_groups"]
            or [d is None for d in entry["digests"]] != [d is None for d in prior["digests"]]
        ):
            raise RuntimeError(f"[sort] {table}/{key}: {entry['source']} bucketed again otherwise than before")
        _append_jsonl(bdir / "bucketed.jsonl", entry)
        out[entry["source"]] = entry
        for i, digest in enumerate(entry["digests"]):
            if digest is not None and i not in needed:
                (bdir / _fragment_name(i, j, n_sources)).unlink()
    return out


def sort_buckets(
    table: Table,
    key: str,
    sources: list[Path],
    buckets: dict,
    plan: list[list[int]],
    key_plan: KeyPlan | None = None,
) -> tuple[list[dict], list[int]]:
    """Sort each bucket into the key's part files (`plan`: each part file's buckets,
    consecutive, every bucket once in order), FINALISE_WORKERS at once, or fewer while
    their estimated peaks (_bucket_memory) would pass the memory available
    (parallel.memory_budget). A part file of one bucket is written by that bucket's sort
    itself, and checked against its rows there (_sort_bucket). A part file of several is
    glued from its buckets (_pack_file) as soon as they are all sorted, each sorted to a
    scratch file (`final-`) and checked against its rows, on a thread of this process
    while the other buckets are sorted (on this thread once none is left), and its
    buckets' scratch files are then removed, so they are read back as just written and
    the key's scratch holds only the buckets of part files not yet glued. Each sorted
    scratch file is listed in `sorted.jsonl`, and each part file, once made and checked,
    in `files.jsonl`, with the buckets it holds; both are reused on a restart (a part
    file's buckets then not sorted again). Each bucket's fragments are removed once it is
    sorted (so a bucket to sort again once its part file is listed, such as one of a part
    file changed after it was listed, has its fragments spilled again, _spill_again).
    Each bucket's row groups are of about COMPACT_ROW_GROUP_BYTES of its own decoded rows
    (row_group_rows), and its estimated peak is from the key's plan (`key_plan`, plan_key).
    Returns the part files and each bucket's rows sorted."""
    key_plan = key_plan or plan_key(table, sources)
    schema = _schema(sources)
    rg_rows = None  # (each bucket's own: row_group_rows)
    bdir = _bucket_dir(table, key)
    sorted_log = bdir / "sorted.jsonl"
    files_log = _out_dir(table) / "files.jsonl"
    sorted_done = {e["name"]: e for e in _read_jsonl(sorted_log)}
    files_done = {(e["key"], e["name"]): e for e in _read_jsonl(files_log)}
    bucketed = {e["source"]: e for e in _read_jsonl(bdir / "bucketed.jsonl") if e.get("format") == BUCKETS_FORMAT}
    bounds = buckets["bounds"]
    n = len(buckets["rows"])
    if [j for idx in plan for j in idx] != list(range(n)) or not all(plan):
        raise RuntimeError(f"[sort] {table}/{key}: the part files' buckets are not each bucket once, in order")
    dst_dir = _out_dir(table) / key
    dst_dir.mkdir(parents=True, exist_ok=True)
    # (`final-`: written with the Hub's settings; a restart does not glue the `sorted-`
    # scratch files of before, zstd 1 without chunking, but sorts again)
    scratch = [bdir / f"final-{j:05d}.parquet" for j in range(n)]
    parts = [dst_dir / _part_name(i, len(plan)) for i in range(len(plan))]
    part_of = {j: i for i, idx in enumerate(plan) for j in idx}
    work = _bucket_parts(bdir, sources, buckets, bucketed)
    files: dict[int, dict] = {}
    rows: dict[int, int] = {}  # each bucket's rows sorted
    waiting: dict[int, set[int]] = {}  # each part file of several not yet made: its buckets not yet sorted
    jobs = {}
    for i, idx in enumerate(plan):
        glued = len(idx) > 1
        names = [scratch[j].name for j in idx]
        prior = files_done.get((key, parts[i].name))
        if (
            prior
            and prior.get("plan") == PLAN  # (planned by the same rule)
            and prior["buckets"] == names
            and prior.get("glued", True) == glued  # (glued before 2026-10-25, every part file)
            and (not glued or all(name in sorted_done for name in names))
            and parts[i].exists()
            and parts[i].stat().st_size == prior["bytes"]
            and _sha256(parts[i]) == prior["sha256"]
        ):
            files[i] = {k: prior[k] for k in ("name", "rows", "bytes", "sha256")}
            rows.update({j: sorted_done[scratch[j].name]["rows"] for j in idx} if glued else {idx[0]: prior["rows"]})
            for j in idx:
                scratch[j].unlink(missing_ok=True)  # (left by a run stopped once the part was listed)
            continue
        if glued:
            waiting[i] = set()
        for j in idx:
            done = sorted_done.get(scratch[j].name)
            if glued and done and scratch[j].exists() and scratch[j].stat().st_size == done["bytes"]:
                rows[j] = done["rows"]
                continue
            lo, hi = bounds[j - 1] if j else None, bounds[j] if j < len(bounds) else None
            dst = scratch[j] if glued else parts[i]
            jobs[j] = (table, key, work[j], dst, schema, rg_rows, buckets["rows"][j], lo, hi, not glued,
                       COMPACT_ROW_GROUP_BYTES)
            if glued:
                waiting[i].add(j)
    lost = {j for j in jobs for _, frag, _, _ in work[j] if frag is not None and not frag.exists()}
    if lost:
        bucketed = _spill_again(table, key, sources, buckets, bucketed, lost, set(jobs))
        work = _bucket_parts(bdir, sources, buckets, bucketed)
        jobs = {j: (*args[:2], work[j], *args[3:]) for j, args in jobs.items()}

    def made(i: int, entry: dict) -> None:
        idx = plan[i]
        if len(idx) > 1 and entry["rows"] != sum(rows[j] for j in idx):
            raise RuntimeError(f"[sort] {table}/{key}: {entry['name']} holds {entry['rows']} rows, its buckets"
                               f" {sum(rows[j] for j in idx)}")
        _append_jsonl(files_log, {"key": key, "buckets": [scratch[j].name for j in idx], "glued": len(idx) > 1,
                                  "plan": PLAN, **entry})
        files[i] = entry
        for j in idx:
            scratch[j].unlink(missing_ok=True)  # no longer needed once glued and checked

    def glue(i: int) -> dict:
        return _pack_file(table, key, parts[i], [scratch[j] for j in plan[i]], schema)

    desc = f"[sort] {table}/{key}: sorting"
    memory = {j: _bucket_memory(jobs[j][6], key_plan) for j in jobs}
    left = len(jobs)
    with ThreadPoolExecutor(1) as pool:
        gluing: dict = {}  # future -> part
        ready = [i for i, w in waiting.items() if not w]  # (every bucket sorted before a restart)
        runs = in_parallel(_sort_bucket, jobs, desc, "bucket", n, tasks_per_child=1, memory=memory)
        for j, entry in runs:
            left -= 1
            i = part_of[j]
            rows[j] = entry["rows"]
            if len(plan[i]) == 1:
                made(i, entry)
            else:
                _append_jsonl(sorted_log, entry)
                waiting[i].discard(j)
                if not waiting[i]:
                    ready.append(i)
            for _, frag, _, _ in jobs[j][2]:
                if frag is not None:
                    frag.unlink()  # no longer needed once checked
            if left:  # (glued beside the sorts still running)
                gluing.update({pool.submit(glue, i): i for i in ready})
                ready = []
            for fut in [f for f in gluing if f.done()]:
                made(gluing.pop(fut), fut.result())
        for fut in list(gluing):
            made(gluing.pop(fut), fut.result())
        for i in ready:  # (no sort left: glued on this thread)
            made(i, glue(i))
    outputs = [dst_dir / files[i]["name"] for i in range(len(plan))]
    what = f"{table}/{key}"
    _check_sorted(what, tqdm(outputs, desc=f"[sort] {what}: checking order", unit="file"), sort_column(table), schema)
    return [files[i] for i in range(len(plan))], [rows[j] for j in range(n)]


def _pack_file(table: Table, key: str, dst: Path, parts: list[Path], schema: pa.Schema) -> dict:
    """Glue one part file from its sorted buckets, already written with the Hub's
    settings: their encoded row groups copied in order, not decoded and written again
    (pq_glue.concat), and the file checked against them byte for byte (run on a thread of
    sort_buckets' process, as the glue and the check wait on the disk, mostly reading
    pages just written). Its row groups are the buckets', each bucket's last one shorter."""
    tmp = dst.with_suffix(".tmp")
    rows = pq_glue.concat([str(p) for p in parts], str(tmp))
    tmp.replace(dst)
    _check_glued(table, key, parts, dst, schema)
    return _entry(dst, rows)


def _check_whole(what: str, sources: list[Path], buckets: dict, rows: list[int]) -> None:
    """The key's rows are its sources' rows: each source's rows counted into one bucket as
    bucketed (each bucket's rows its ids' rows counted, bucket_key), each bucket's rows
    read as planned (each row group's rows of the bucket as many as counted, each
    fragment the bytes bucketed, _read_bucket) and its sorted file checked against them
    row for row (_sort_bucket), each part file of several buckets checked against its
    sorted buckets byte for byte (_pack_file), a part file of one bucket being that
    bucket's sorted file. So the sorted buckets' rows (`rows`), bucket by bucket, and in
    all the sources' rows."""
    if rows != buckets["rows"]:
        bad = next(i for i, (a, b) in enumerate(zip(rows, buckets["rows"])) if a != b) if len(rows) == len(buckets["rows"]) else None
        raise RuntimeError(f"[sort] {what}: rows sorted differ from those bucketed (bucket {bad})")
    want = sum(pq.ParquetFile(s).metadata.num_rows for s in sources)
    if sum(rows) != want:
        raise RuntimeError(f"[sort] {what}: {sum(rows)} rows sorted, {want} in the sources")
    print(f"[sort] {what}: rows match the sources", flush=True)


def sort_bucketed(table: Table, key: str, sources: list[Path], plan: KeyPlan | None = None) -> list[dict]:
    plan = plan or plan_key(table, sources)
    buckets = bucket_key(table, key, sources, plan)
    if buckets["sources"] != [s.name for s in sources]:
        raise RuntimeError(f"[sort] {table}/{key}: buckets were made from other files")
    # (each part file's buckets, by their rows counted, before any is sorted: KeyPlan.pack)
    files, rows = sort_buckets(table, key, sources, buckets, plan.pack(buckets["rows"]), plan)
    _check_whole(f"{table}/{key}", sources, buckets, rows)
    return files


# Stages


def read_manifest(table: Table) -> dict[str, dict]:
    return {e["key"]: e for e in _read_jsonl(_manifest_path(table))}


def write_table(table: Table) -> None:
    """Sort every key not yet in the manifest."""
    done = read_manifest(table)
    keys = _local_keys(table)
    todo = {
        key: files
        for key, files in keys.items()
        if done.get(key, {}).get("sources") != [f.name for f in files]
    }
    print(f"[sort] {table}: sorting {len(todo)} of {len(keys)} keys", flush=True)

    def record(key: str, sources: list[Path], files: list[dict]) -> None:
        entry = {"key": key, "sources": [s.name for s in sources], "files": files}
        _append_jsonl(_manifest_path(table), entry)

    plans = {k: plan_key(table, s) for k, s in todo.items()}  # (from the footers alone)
    for key in [k for k in todo if not plans[k].in_memory]:
        print(f"[sort] {table}/{key}: {len(todo[key])} files, sorting through buckets", flush=True)
        record(key, todo[key], sort_bucketed(table, key, todo[key], plans[key]))
    # Keys sorted in memory, a job each, largest first: the larger ones (KeyPlan.larger)
    # FINALISE_LARGE_WORKERS at once, as each holds its rows twice
    small = sorted((k for k in todo if plans[k].in_memory), key=lambda k: -plans[k].decoded)
    larger = [k for k in small if plans[k].larger]
    # (a larger key's worker replaced after each, the rest kept)
    for keys, desc, workers, per_child in (
        (larger, f"[sort] {table}: larger keys", FINALISE_LARGE_WORKERS, 1),
        ([k for k in small if k not in larger], f"[sort] {table}: keys", None, None),
    ):
        jobs = {key: (table, key, todo[key], plans[key]) for key in keys}
        runs = in_parallel(sort_in_memory, jobs, desc, "key", workers=workers, tasks_per_child=per_child)
        for key, files in runs:
            record(key, todo[key], files)
    print(f"[sort] {table}: written", flush=True)


def _key_done(table: Table, entry: dict, remote: dict[str, object]) -> bool:
    """The key's files on the Hub are exactly its sorted files."""
    names = {f["name"] for f in entry["files"]}
    if set(remote) != names:
        return False
    local = _out_dir(table) / entry["key"]
    return all(_same(remote[f["name"]], local / f["name"], f["sha256"]) for f in entry["files"])


def _key_operations(
    table: Table, entry: dict, remote: dict[str, object], replaced: set[str]
) -> list:
    """The key's sorted files added, and its other files on the Hub deleted: its sources,
    or the group files they were compacted from (`replaced`)."""
    key = entry["key"]
    new = {f["name"] for f in entry["files"]}
    unexpected = set(remote) - new - set(entry["sources"]) - replaced
    if unexpected:
        raise RuntimeError(f"[sort] {table}/{key}: unknown files on the Hub: {unexpected}")
    adds = [
        CommitOperationAdd(
            path_in_repo=f"{key}/{name}", path_or_fileobj=str(_out_dir(table) / key / name)
        )
        for name in sorted(new)
    ]
    deletes = [CommitOperationDelete(path_in_repo=f"{key}/{n}") for n in sorted(set(remote) - new)]
    return adds + deletes


def commit_table(table: Table, repo_id: str, api: HfApi) -> None:
    """Replace each key's files on the Hub with its sorted files, one commit per batch of
    keys, a key's additions and deletions always in the same commit. The Hub has each
    key's group files (or none: claims_labels' are never uploaded), or its compacted
    files when compaction uploaded them (before 2026-10-05)."""
    print(f"[sort] {table}: committing to {repo_id}", flush=True)
    manifest = read_manifest(table)
    compacted = compact_manifest(table)
    remote = _remote_files(repo_id, api)
    if extra := set(remote) - set(manifest):
        raise RuntimeError(f"[sort] {table}: keys on the Hub not sorted: {sorted(extra)}")
    todo = [e for k, e in manifest.items() if not _key_done(table, e, remote.get(k, {}))]
    batch: list = []
    batch_keys: list[str] = []

    def commit():
        nonlocal batch, batch_keys
        if batch:
            api.create_commit(
                repo_id,
                repo_type="dataset",
                revision=HUB_REVISION,
                operations=batch,
                commit_message=f"Sort {len(batch_keys)} keys by id ({batch_keys[0]} to {batch_keys[-1]})",
            )
            print(f"[sort] {table}: committed {len(batch_keys)} keys", flush=True)
        batch, batch_keys = [], []

    for entry in todo:
        replaced = set(compacted.get(entry["key"], {}).get("sources", []))
        ops = _key_operations(table, entry, remote.get(entry["key"], {}), replaced)
        n_adds = sum(isinstance(op, CommitOperationAdd) for op in batch + ops)
        if batch and (n_adds > COMPACT_COMMIT_MAX_ADDS or len(batch) + len(ops) > COMPACT_COMMIT_MAX_OPS):
            commit()
        batch += ops
        batch_keys.append(entry["key"])
    commit()
    print(f"[sort] {table}: {len(todo)} keys committed, {len(manifest) - len(todo)} already", flush=True)


def verify_table(table: Table, repo_id: str, api: HfApi) -> None:
    """The Hub has exactly the sorted files of every key, and no other key."""
    print(f"[sort] {table}: verifying {repo_id}", flush=True)
    manifest = read_manifest(table)
    remote = _remote_files(repo_id, api)
    if set(remote) != set(manifest):
        raise RuntimeError(f"[sort] {table}: keys differ from the manifest on the Hub")
    bad = [k for k, e in manifest.items() if not _key_done(table, e, remote[k])]
    if bad:
        raise RuntimeError(f"[sort] {table}: keys not sorted on the Hub: {bad}")
    print(f"[sort] {table}: verified {len(manifest)} keys", flush=True)


def write_metadata(table: Table) -> None:
    """Record the table's files, bytes and rows per key in DATASET_CARDS_METADATA."""
    path = DATASET_CARDS_METADATA
    metadata = json.loads(path.read_text()) if path.exists() else {}
    metadata[str(table)] = {
        key: {
            "files": len(e["files"]),
            "bytes": sum(f["bytes"] for f in e["files"]),
            "rows": sum(f["rows"] for f in e["files"]),
        }
        for key, e in sorted(read_manifest(table).items())
    }
    ordered = {str(t): metadata[str(t)] for t in Table if str(t) in metadata}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ordered, indent=2) + "\n")
    print(f"[sort] {table}: metadata written", flush=True)


def remove_compact_src(table: Table) -> None:
    """Remove the table's compaction source directory, emptied when compaction finished
    and superseded by the checked local copy, and `compact/src` once empty."""
    shutil.rmtree(_compact_src_dir(table), ignore_errors=True)
    root = _compact_src_dir(table).parent
    if root.is_dir() and not any(root.iterdir()):
        root.rmdir()


def replace_local_copy(table: Table) -> None:
    """Move each key's sorted files into the local copy in place of its old files, so the
    copy matches the Hub again."""
    for key, entry in read_manifest(table).items():
        key_dir = _src_dir(table) / key
        for name in entry["sources"]:
            (key_dir / name).unlink(missing_ok=True)
            cached = _src_dir(table) / ".cache" / "huggingface" / "download" / key / f"{name}.metadata"
            cached.unlink(missing_ok=True)
        for f in entry["files"]:
            src = _out_dir(table) / key / f["name"]
            if src.exists():
                shutil.move(src, key_dir / f["name"])
    print(f"[sort] {table}: local copy replaced with the sorted files", flush=True)


def sort_table(
    table: Table, repo_id: str, state_dir: Path, api: HfApi | None = None
) -> None:
    """Run the table's remaining stages, once it is compacted."""
    api = api or HfApi()
    if compact_stage(state_dir, table) != "done":
        raise RuntimeError(f"[sort] {table}: not compacted yet")
    stage = last_stage(state_dir, table)
    done = STAGES.index(stage) if stage else -1
    if stage == "done":
        print(f"[sort] {table}: already sorted ({repo_id})", flush=True)
        return
    resuming = f", resuming after stage {stage!r}" if stage else ""
    print(f"[sort] Sorting {table} ({repo_id}){resuming}", flush=True)
    if done < STAGES.index("sourced"):
        if not _src_dir(table).is_dir():
            raise RuntimeError(f"[sort] {table}: no local copy in {_src_dir(table)}: run download-wikidata")
        if _uploaded_by_compaction(state_dir, table):
            source_table(table, repo_id, api)
        else:
            check_compacted_copy(table)
        record_stage(state_dir, table, "sourced")
    remove_compact_src(table)
    if done < STAGES.index("written"):
        write_table(table)
        record_stage(state_dir, table, "written")
    if done < STAGES.index("committed"):
        commit_table(table, repo_id, api)
        record_stage(state_dir, table, "committed")
    if done < STAGES.index("verified"):
        verify_table(table, repo_id, api)
        record_stage(state_dir, table, "verified")
    if done < STAGES.index("done"):
        write_metadata(table)
        replace_local_copy(table)
        if CLEAN_UP_LOCAL:
            shutil.rmtree(SORT_DIR / "buckets" / table, ignore_errors=True)
            for key_dir in _out_dir(table).iterdir():
                if key_dir.is_dir():
                    shutil.rmtree(key_dir)
        record_stage(state_dir, table, "done")
    print(f"[sort] {table}: complete", flush=True)
