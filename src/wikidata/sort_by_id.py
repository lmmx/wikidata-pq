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
  A key over SORT_IN_MEMORY_BYTES (claims) goes through id-range buckets: each source
  file bucketed, each bucket sorted to a file of the Hub's settings, and consecutive
  sorted buckets glued into the key's files (their encoded row groups copied, not
  decoded), each step FINALISE_WORKERS at once and resumable
  (docs/journal/2026-10-05-sort-speed.md)
- committed: each key's part files added and its old files (group files, or compacted
  files uploaded before 2026-10-05) deleted in one commit
- verified: the Hub has exactly the part files of every key
- done: the metadata JSON rewritten, and HUB_COPY_DIR/{table} holds the sorted files
"""

from __future__ import annotations

import json
import math
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

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
    _row_group_rows,
    _same,
    _sample_row_group_rows,
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
    SORT_IN_MEMORY_BYTES,
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


_HALF = 2**32


def _row_hashes(lf: pl.LazyFrame, *keep: str) -> pl.DataFrame:
    """Each row's hash under seeds 0 and 1 (`h0`, `h1`), with the `keep` columns, in the
    rows' order. Streaming. The hashes are columns, summed after: on claims a struct
    hash summed in the same expression took about twice the CPU time."""
    row = pl.struct(pl.all())
    hashes = [row.hash(seed=0).alias("h0"), row.hash(seed=1).alias("h1")]
    return lf.select(*keep, *hashes).collect(engine="streaming")


def _sums_of(h: pl.DataFrame) -> list[int]:
    """_additive of the rows hashed in `h` (_row_hashes)."""
    out = [h.height]
    for seed in (0, 1):
        s = h[f"h{seed}"]
        out += [int((s % _HALF).sum()), int((s // _HALF).sum())]
    return out


def _additive(lf: pl.LazyFrame) -> list[int]:
    """Row count and, for two seeds, the sums of the low and of the high 32 bits of each
    row's hash: exact (no overflow below 2**32 rows), so the sums of the parts of any
    split of the rows add up to the whole's (see _add). Equal for the same rows in any
    order. Streaming."""
    return _sums_of(_row_hashes(lf))


def _add(parts) -> list[int]:
    """The element-wise sum of _additive results."""
    return [sum(xs) for xs in zip(*parts, strict=True)]


def _check_equal(what: str, got, want) -> None:
    if got != want:
        raise RuntimeError(f"[sort] {what}: rows differ (rows, hash sums: {got}, expected {want})")


def _check_stable_sort(
    what: str, outputs: list[Path], sources: list[Path], col: str, schema: pa.Schema
) -> None:
    """The outputs have the schema, and their rows, in order, are the sources' rows
    stably sorted by `col`: the sort column ascends through the outputs, and each row's
    hash, as Polars reads the outputs, is the hash of the row Polars' own stable sort of
    the sources puts there. Exact order, not sums, each side read and hashed once."""
    _check_schemas(what, outputs, schema)
    row = pl.struct(pl.all()).hash(seed=0).alias("h0")
    got, want = pl.collect_all([
        pl.scan_parquet(outputs).select(pl.col(col), row),
        pl.scan_parquet(sources)
        .select(pl.col(col), row)
        .sort(col, maintain_order=True, nulls_last=True)
        .select("h0"),
    ])
    _check_order(what, got, want["h0"], col)


def _check_schemas(what: str, outputs: list[Path], schema: pa.Schema) -> None:
    for out in outputs:
        if not pq.read_schema(out).remove_metadata().equals(schema):
            raise RuntimeError(f"[sort] {what}: {out.name} has another schema")


def _check_order(what: str, got: pl.DataFrame, want: pl.Series, col: str) -> None:
    """`got`, the outputs' sort column and row hashes (`h0`) in order, ascends by `col`
    and has the hashes `want`, in order."""
    if not got[col].is_sorted():
        raise RuntimeError(f"[sort] {what}: not sorted by {col}")
    if got.height != len(want):
        raise RuntimeError(f"[sort] {what}: {got.height} rows, expected {len(want)}")
    differ = (got["h0"] != want).arg_true()
    if len(differ):
        raise RuntimeError(f"[sort] {what}: {len(differ)} rows differ, the first at row {differ[0]}")


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


# Writing


def _read(sources: list[Path]) -> pa.Table:
    """A key's files read one at a time and concatenated. Not read as one dataset, which
    casts to a unified schema: casting corrupts the nested structs of claims (see
    compact._source_batches)."""
    return pa.concat_tables(pq.read_table(f) for f in sources)


_THREADED_TAKE_BYTES = 64 * 1024**2


def _sorted(t: pa.Table, col: str) -> pa.Table:
    """`t` stably sorted by `col`. A table over _THREADED_TAKE_BYTES has each column
    taken in a thread of its own (Arrow releases the GIL): a large key, often sorted
    once the other workers are done, uses the cores they leave. The same table as
    `t.take` either way."""
    idx = pc.sort_indices(t, [(col, "ascending")])  # a stable sort
    if t.nbytes <= _THREADED_TAKE_BYTES:
        return t.take(idx)
    with ThreadPoolExecutor(min(t.num_columns, pa.cpu_count())) as pool:
        return pa.Table.from_arrays(list(pool.map(lambda c: c.take(idx), t.columns)), schema=t.schema)


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


def _read_key(sources: list[Path], threads: bool) -> tuple[pa.Schema, int, pa.Table]:
    """A key's schema (as _schema), rows per row group (as compact._row_group_rows) and
    rows (as _read), each file opened once, not three times: on the benchmark's quick
    tier, whose ~2,300 small keys have a file each, the schema and the row group sample
    read apart (the sample a second read of the whole file) were a tenth of a key's
    time, and pq.read_table (a dataset scan) took 1.5 to 2 times as long as
    ParquetFile.read. The sample is the largest file's first row group, read as
    _row_group_rows reads it, and is that file's rows if it has one row group (the same
    values as `read`, but not the same `nbytes`, so the sample is not taken from
    `read`). Read on Arrow's threads if `threads`, else on this one. The same values as
    _read, so the same sorted files."""
    files = [pq.ParquetFile(src) for src in sources]
    schema = files[0].schema_arrow.remove_metadata()
    for src, f in zip(sources[1:], files[1:]):
        if not f.schema_arrow.remove_metadata().equals(schema):
            raise RuntimeError(f"[sort] {src} has another schema")
    largest = max(range(len(sources)), key=lambda i: sources[i].stat().st_size)
    sample = files[largest].read_row_group(0, use_threads=threads)
    whole = files[largest].metadata.num_row_groups == 1
    t = pa.concat_tables(
        sample if i == largest and whole else f.read(use_threads=threads)
        for i, f in enumerate(files)
    )
    return schema, _sample_row_group_rows(sample), t


def sort_in_memory(table: Table, key: str, sources: list[Path]) -> list[dict]:
    """Sort a key read whole into `ceil(bytes / COMPACT_FILE_BYTES)` files of equal rows,
    and check them against the key's files."""
    col = sort_column(table)
    size = sum(s.stat().st_size for s in sources)
    schema, rg_rows, t = _read_key(sources, threads=size > _THREADED_READ_BYTES)
    t = _sorted(t, col)
    n = max(1, math.ceil(size / COMPACT_FILE_BYTES))
    per_file = math.ceil(t.num_rows / n)
    dst_dir = _out_dir(table) / key
    dst_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for i in range(n):
        dst = dst_dir / _part_name(i, n)
        part = t.slice(i * per_file, per_file)
        rows = _write_file_parallel(
            dst, schema, part.to_batches(max_chunksize=rg_rows), rg_rows, _sorting(schema, col)
        )
        files.append(_entry(dst, rows))
    del t
    outputs = [dst_dir / f["name"] for f in files]
    _check_stable_sort(f"{table}/{key}", outputs, sources, col, schema)
    return files


# The bucket path, for a key too large to sort in memory


def _boundaries(sources: list[Path], col: str, n: int) -> list[str]:
    """The first id of each bucket but the first, splitting the key's rows into `n`
    buckets of about equal rows, never splitting an id."""
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
    return counts[col].gather(idx).to_list()


# A bucket's fragments are Arrow IPC files, zstd-compressed: scratch files, written once
# and read once, so not encoded as Parquet (on claims, IPC took under half the CPU time
# to write and a fifth to read back, at 1.4 times the size)
FRAGMENT_FORMAT = "arrow"


def _fragment_name(i: int, j: int, n_sources: int, ext: str = FRAGMENT_FORMAT) -> str:
    """Bucket i's rows from source file j, the source index padded to the digits of the
    key's last source index."""
    return f"bucket-{i:05d}-{j:0{len(str(n_sources - 1))}d}.{ext}"


def _bucket_files(bdir: Path, i: int, buckets: dict) -> list[Path]:
    """Bucket i's files, in source order: one per source file, or one in all (buckets
    made before 2026-10-05, without "fragments"); Parquet if made before 2026-10-24,
    without "format"."""
    n = buckets.get("fragments")
    if n is None:
        return [bdir / f"bucket-{i:05d}.parquet"]
    ext = buckets.get("format", "parquet")
    return [bdir / _fragment_name(i, j, n, ext) for j in range(n)]


def _read_fragments(paths: list[Path]) -> pa.Table:
    """A bucket's fragments, Arrow IPC or (made before 2026-10-24) Parquet, concatenated."""
    if paths and paths[0].suffix == ".arrow":
        return pa.concat_tables(ipc.open_file(p).read_all() for p in paths)
    return _read(paths)


def _scan_fragments(paths: list[Path]) -> pl.LazyFrame:
    if paths and paths[0].suffix == ".arrow":
        return pl.scan_ipc(paths)
    return pl.scan_parquet(paths)


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
    """Write source file j's rows to its fragment of every bucket, in source order, each
    bucket's rows buffered to record batches of about `flush_rows` rows; return its rows
    per bucket, and its _additive sums as Polars reads it (run in a worker process)."""
    col = sort_column(table)
    lookup = pl.Series(bounds, dtype=pl.String)
    n = len(bounds) + 1
    paths = [bdir / _fragment_name(i, j, n_sources) for i in range(n)]
    options = ipc.IpcWriteOptions(compression="zstd")
    writers = [ipc.new_file(p, schema, options=options) for p in paths]
    pending: list[list[pa.Table]] = [[] for _ in range(n)]
    pending_rows = [0] * n
    rows = [0] * n

    def flush(i: int) -> None:
        if pending[i]:
            # one batch, not one per slice (a few rows each when there are many buckets)
            writers[i].write_table(pa.concat_tables(pending[i]).combine_chunks())
        pending[i], pending_rows[i] = [], 0

    pf = pq.ParquetFile(src)
    try:
        for g in range(pf.metadata.num_row_groups):
            rg = pf.read_row_group(g)
            ids = pl.from_arrow(rg.column(col)).cast(pl.String)
            b = lookup.search_sorted(ids, side="right").to_numpy()
            order = np.argsort(b, kind="stable")  # each bucket's rows in source order
            rg, b = rg.take(pa.array(order)), b[order]
            starts = np.flatnonzero(np.r_[True, b[1:] != b[:-1]])
            for s, e in zip(starts, np.r_[starts[1:], len(b)]):
                i = int(b[s])
                pending[i].append(rg.slice(s, e - s))
                pending_rows[i] += int(e - s)
                rows[i] += int(e - s)
                if pending_rows[i] >= flush_rows:
                    flush(i)
        for i in range(n):
            flush(i)
    finally:
        for w in writers:
            w.close()
    sums = _additive(pl.scan_parquet(src))
    return {"source": src.name, "rows": rows, "sums": sums, "format": FRAGMENT_FORMAT, "at": time.time()}


def bucket_key(table: Table, key: str, sources: list[Path]) -> dict:
    """Write each of the key's rows to its bucket, in source order: each source file to
    its own fragment of every bucket, FINALISE_WORKERS files at once. The boundaries are in
    `bounds.json`, each source file bucketed in `bucketed.jsonl` (reused on a restart),
    and the buckets, their rows and the sources' _additive sums in `buckets.json`."""
    bdir = _bucket_dir(table, key)
    record = bdir / "buckets.json"
    if record.exists():
        return json.loads(record.read_text())
    col = sort_column(table)
    schema = _schema(sources)
    names = [s.name for s in sources]
    bounds_path = bdir / "bounds.json"
    if bounds_path.exists() and json.loads(bounds_path.read_text())["sources"] == names:
        bounds = json.loads(bounds_path.read_text())["bounds"]
    else:
        n = max(1, math.ceil(sum(s.stat().st_size for s in sources) / SORT_BUCKET_BYTES))
        print(f"[sort] {table}/{key}: finding {n} bucket boundaries", flush=True)
        bounds = _boundaries(sources, col, n)
        shutil.rmtree(bdir, ignore_errors=True)
        bdir.mkdir(parents=True)
        bounds_path.write_text(json.dumps({"bounds": bounds, "sources": names}))
    log = bdir / "bucketed.jsonl"
    # (a source bucketed to Parquet fragments, before 2026-10-24, is bucketed again)
    done = {
        e["source"]: e
        for e in _read_jsonl(log)
        if e["source"] in names and e.get("format") == FRAGMENT_FORMAT
    }
    rg_rows = _row_group_rows(sources)
    flush_rows = max(1, rg_rows * SORT_BUCKET_WRITE_BYTES // COMPACT_ROW_GROUP_BYTES)
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
    buckets = {
        "bounds": bounds,
        "rows": [sum(r) for r in zip(*(done[n]["rows"] for n in names), strict=True)],
        "sources": names,
        "fragments": len(sources),
        "format": FRAGMENT_FORMAT,
        "sums": _add(done[n]["sums"] for n in names),
    }
    record.write_text(json.dumps(buckets))
    return buckets


def _sort_bucket(
    table: Table,
    key: str,
    srcs: list[Path],
    dst: Path,
    schema: pa.Schema,
    rg_rows: int,
    want_rows: int,
) -> dict:
    """Sort one bucket to `dst` with the Hub's settings (zstd 3, a page index,
    content-defined chunking, sorted by `col`), to be glued into the key's files as it
    is, checked against the bucket's files; with the bucket's _additive sums (run in a
    worker process)."""
    col = sort_column(table)
    t = _sorted(_read_fragments(srcs), col)
    rows = _write_file_parallel(
        dst, schema, t.to_batches(max_chunksize=rg_rows), rg_rows, _sorting(schema, col)
    )
    del t
    if rows != want_rows:
        raise RuntimeError(f"[sort] {table}/{key}: {dst.name} has {rows} rows")
    # The bucket's rows hashed once: their sums for the key's end check (_check_whole),
    # and in stable id order, the sorted file's rows exactly
    what = f"{table}/{key} {dst.name}"
    _check_schemas(what, [dst], schema)
    h = _row_hashes(_scan_fragments(srcs), col)
    want = h.sort(col, maintain_order=True, nulls_last=True)["h0"]
    got = (
        pl.scan_parquet(dst)
        .select(pl.col(col), pl.struct(pl.all()).hash(seed=0).alias("h0"))
        .collect(engine="streaming")
    )
    _check_order(what, got, want, col)
    return {"name": dst.name, "rows": rows, "bytes": dst.stat().st_size, "sums": _sums_of(h)}


def sort_buckets(table: Table, key: str, sources: list[Path], buckets: dict) -> list[dict]:
    """Sort each bucket to a scratch file, checked against the bucket's files,
    FINALISE_WORKERS at once; each is listed in `sorted.jsonl` and reused on a restart."""
    schema = _schema(sources)
    rg_rows = _row_group_rows(sources)
    bdir = _bucket_dir(table, key)
    log = bdir / "sorted.jsonl"
    done = {e["name"]: e for e in _read_jsonl(log)}
    n = len(buckets["rows"])
    out: dict[int, dict] = {}
    jobs = {}
    for i in range(n):
        # (`final-`: written with the Hub's settings; a restart does not glue the
        # `sorted-` scratch files of before, zstd 1 without chunking, but sorts again)
        srcs, dst = _bucket_files(bdir, i, buckets), bdir / f"final-{i:05d}.parquet"
        prior = done.get(dst.name)
        if prior and dst.exists() and dst.stat().st_size == prior["bytes"]:
            out[i] = prior
        else:
            jobs[i] = (table, key, srcs, dst, schema, rg_rows, buckets["rows"][i])
    desc = f"[sort] {table}/{key}: sorting"
    for i, entry in in_parallel(_sort_bucket, jobs, desc, "bucket", n, tasks_per_child=1):
        _append_jsonl(log, entry)
        for src in jobs[i][2]:
            src.unlink()  # no longer needed once checked
        out[i] = entry
    return [out[i] for i in range(n)]


def _pack(sizes: list[int]) -> list[list[int]]:
    """Consecutive bucket indices for each of `ceil(total / COMPACT_FILE_BYTES)` files,
    cut where the cumulative size is nearest each multiple of total / n."""
    total = sum(sizes)
    n = max(1, math.ceil(total / COMPACT_FILE_BYTES))
    cum = np.cumsum(sizes)
    cuts = [0]
    for k in range(1, n):
        c = int(np.argmin(np.abs(cum - total * k / n))) + 1
        if cuts[-1] < c < len(sizes):
            cuts.append(c)
    cuts.append(len(sizes))
    return [list(range(a, b)) for a, b in zip(cuts, cuts[1:])]


def _pack_file(table: Table, key: str, dst: Path, parts: list[Path], schema: pa.Schema) -> dict:
    """Glue one part file from its sorted buckets, already written with the Hub's
    settings: their encoded row groups copied in order, not decoded and written again
    (pq_glue.concat), and the file checked against them byte for byte (run in a worker
    process). Its row groups are the buckets', each bucket's last one shorter."""
    tmp = dst.with_suffix(".tmp")
    rows = pq_glue.concat([str(p) for p in parts], str(tmp))
    tmp.replace(dst)
    _check_glued(table, key, parts, dst, schema)
    return _entry(dst, rows)


def pack_key(
    table: Table, key: str, sources: list[Path], buckets: dict, sorted_buckets: list[dict]
) -> list[dict]:
    """Glue consecutive sorted buckets into the key's part files of about even size; each
    checked against its buckets and listed in `files.jsonl`, and reused on a restart.
    FINALISE_WORKERS files at once."""
    col = sort_column(table)
    schema = _schema(sources)
    bdir = _bucket_dir(table, key)
    log = _out_dir(table) / "files.jsonl"
    done = {(e["key"], e["name"]): e for e in _read_jsonl(log)}
    plan = _pack([b["bytes"] for b in sorted_buckets])
    dst_dir = _out_dir(table) / key
    dst_dir.mkdir(parents=True, exist_ok=True)
    files: dict[int, dict] = {}
    jobs = {}
    for i, idx in enumerate(plan):
        dst = dst_dir / _part_name(i, len(plan))
        parts = [bdir / sorted_buckets[j]["name"] for j in idx]
        prior = done.get((key, dst.name))
        if (
            prior
            and prior["buckets"] == [p.name for p in parts]
            and dst.exists()
            and dst.stat().st_size == prior["bytes"]
            and _sha256(dst) == prior["sha256"]
        ):
            files[i] = {k: prior[k] for k in ("name", "rows", "bytes", "sha256")}
        else:
            jobs[i] = (table, key, dst, parts, schema)
    desc = f"[sort] {table}/{key}: packing"
    for i, entry in in_parallel(_pack_file, jobs, desc, "file", len(plan), tasks_per_child=1):
        parts = jobs[i][3]
        _append_jsonl(log, {"key": key, "buckets": [p.name for p in parts], **entry})
        files[i] = entry
    outputs = [dst_dir / files[i]["name"] for i in range(len(plan))]
    what = f"{table}/{key}"
    _check_sorted(what, tqdm(outputs, desc=f"[sort] {what}: checking order", unit="file"), col, schema)
    _check_whole(what, sources, buckets, sorted_buckets, outputs)
    return [files[i] for i in range(len(plan))]


def _check_whole(
    what: str, sources: list[Path], buckets: dict, sorted_buckets: list[dict], outputs: list[Path]
) -> None:
    """The key's rows are its sources' rows. Each part file is checked against its sorted
    buckets and each sorted bucket against its bucket, so this checks bucketing: the
    sources' _additive sums (taken as they were bucketed) against the sum of the buckets'
    (taken as they were sorted). Without them (buckets made before 2026-10-05), the
    sources and part files are read again, a file at a time."""
    if "sums" in buckets and all("sums" in b for b in sorted_buckets):
        _check_equal(what, _add(b["sums"] for b in sorted_buckets), buckets["sums"])
        print(f"[sort] {what}: rows match the sources", flush=True)
        return

    def sums(files: list[Path], side: str) -> list[int]:
        bar = tqdm(files, desc=f"[sort] {what}: checking {side} rows", unit="file")
        return _add(_additive(pl.scan_parquet(f)) for f in bar)

    _check_equal(what, sums(outputs, "sorted"), sums(sources, "source"))


def sort_bucketed(table: Table, key: str, sources: list[Path]) -> list[dict]:
    buckets = bucket_key(table, key, sources)
    if buckets["sources"] != [s.name for s in sources]:
        raise RuntimeError(f"[sort] {table}/{key}: buckets were made from other files")
    return pack_key(table, key, sources, buckets, sort_buckets(table, key, sources, buckets))


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

    size = {k: sum(f.stat().st_size for f in s) for k, s in todo.items()}
    for key in [k for k in todo if size[k] > SORT_IN_MEMORY_BYTES]:
        print(f"[sort] {table}/{key}: {len(todo[key])} files, sorting through buckets", flush=True)
        record(key, todo[key], sort_bucketed(table, key, todo[key]))
    # Keys sorted in memory, a job each: the larger ones (over an eighth of
    # SORT_IN_MEMORY_BYTES) FINALISE_LARGE_WORKERS at once, as each holds its rows twice
    small = sorted((k for k in todo if size[k] <= SORT_IN_MEMORY_BYTES), key=lambda k: -size[k])
    larger = [k for k in small if size[k] > SORT_IN_MEMORY_BYTES // 8]
    # (a larger key's worker replaced after each, the rest kept)
    for keys, desc, workers, per_child in (
        (larger, f"[sort] {table}: larger keys", FINALISE_LARGE_WORKERS, 1),
        ([k for k in small if k not in larger], f"[sort] {table}: keys", None, None),
    ):
        jobs = {key: (table, key, todo[key]) for key in keys}
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
