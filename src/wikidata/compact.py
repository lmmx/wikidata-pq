"""Compact each table's group files, once every group is uploaded.

The grouped upload leaves one file per partition key per group, so a key has up to one
file per group, most of them small. Compaction glues each key's group files into files of
about COMPACT_FILE_BYTES, copying their encoded row groups (pq_glue.concat; a deduplicated
table's keys are rewritten, keeping each distinct row once), split only between groups and named by the chunks they cover, and
hands them to the sort (sort_by_id.py), which replaces the group files on the Hub with
the sorted files: the compacted files are not uploaded (docs/journal/2026-10-05-sort-speed.md).

A table goes through these stages, recorded in `compact.jsonl` in the state dir so an
interrupted run resumes at the stage it had not finished. Every stage is safe to repeat.

- downloaded: the group files are in COMPACT_DIR/src/{table} (claims_labels' are put
  there by its build, and nothing is downloaded)
- written: each key is rewritten to COMPACT_DIR/out/{table}, checked against its group
  files, and listed in the table's manifest. Each output file glued from several group
  files is a job, FINALISE_WORKERS at once (a deduplicated table's key is one job); one
  of a single group file is a hard link to it, made in the parent (_link_run); each file
  checked is listed in `files.jsonl` and not rewritten on a restart. A key the sort will
  sort in memory (sort_by_id.sorted_in_memory, planned from its group files' footers) is
  not glued: each of its group files is linked as a file of its own, which the sort reads
  whole as it would have read the glued files
- done: the table's files, bytes and rows per key are in DATASET_CARDS_METADATA, and
  the new files are moved to HUB_COPY_DIR/{table}, the sort's input

Before 2026-10-05 compaction also uploaded its files (stages `committed`, `verified`
between `written` and `done`), as for release 20260928's claims.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

import polars as pl
import pq_glue
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from tqdm import tqdm

from .config import (
    HUB_REVISION,
    CLEAN_UP_LOCAL,
    COMPACT_DIR,
    COMPACT_DOWNLOAD_WORKERS,
    COMPACT_FILE_BYTES,
    COMPACT_ROW_GROUP_BYTES,
    DATASET_CARDS_METADATA,
    HUB_COPY_DIR,
    RELEASE,
    Table,
)
from .parallel import in_parallel
from .push.core import DEDUPLICATE, _git_blob_sha1, _sha256
from .push.groups import chunk_range_name
from .state import last_chunk

STAGES = ["downloaded", "written", "committed", "verified", "done"]

# A key's file on the Hub, before and after compaction: {key}/chunks-{first}-{last}.parquet
# (see chunk_range_name). Any width is read: release 20260928 uploaded its groups with
# 4-digit names up to chunk 9999 and 5-digit names after
FILE_RE = re.compile(r"^([^/]+)/chunks-(\d+)-(\d+)\.parquet$")


def _ledger(state_dir: Path) -> Path:
    return state_dir / "compact.jsonl"


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
    return COMPACT_DIR / "src" / table


def _out_dir(table: Table) -> Path:
    return COMPACT_DIR / "out" / table


def _manifest_path(table: Table) -> Path:
    return _out_dir(table) / "manifest.jsonl"


def _chunk_range(name: str) -> tuple[int, int]:
    m = FILE_RE.match(f"key/{name}")
    assert m, name
    return int(m.group(2)), int(m.group(3))


def download(table: Table, repo_id: str) -> None:
    """Download the table's group files (resumable: files already there are skipped)."""
    print(f"[compact] {table}: downloading {repo_id} to {_src_dir(table)}", flush=True)
    snapshot_download(
        repo_id,
        repo_type="dataset",
        revision=HUB_REVISION,
        local_dir=_src_dir(table),
        allow_patterns="*/chunks-*.parquet",
        max_workers=COMPACT_DOWNLOAD_WORKERS,
    )
    print(f"[compact] {table}: downloaded", flush=True)


def _local_keys(table: Table) -> dict[str, list[Path]]:
    """Each key's group files, in chunk order."""
    keys: dict[str, list[Path]] = {}
    for p in sorted(_src_dir(table).glob("*/chunks-*.parquet"), key=lambda p: _chunk_range(p.name)):
        keys.setdefault(p.parent.name, []).append(p)
    return keys


def _runs(sources: list[Path], sizes: dict[str, float]) -> list[list[Path]]:
    """Consecutive group files making up each output file: a file takes group files until
    the next would take it past COMPACT_FILE_BYTES, by their `sizes` (bytes by name)."""
    runs: list[list[Path]] = [[]]
    size = 0.0
    for src in sources:
        if runs[-1] and size + sizes[src.name] > COMPACT_FILE_BYTES:
            runs.append([])
            size = 0.0
        runs[-1].append(src)
        size += sizes[src.name]
    return runs


def _run_name(run: list[Path], last_chunk: int) -> str:
    first, last = _chunk_range(run[0].name)[0], _chunk_range(run[-1].name)[1]
    return f"{chunk_range_name(first, last, last_chunk)}.parquet"


def _row_group_rows(sources: list[Path]) -> int:
    """Rows per row group for about COMPACT_ROW_GROUP_BYTES of rows in memory, from the
    Arrow size per row of the key's largest group file's first row group. A reader holds
    a row group in memory, so it is sized by its in-memory size, not its Parquet size:
    claims rows take several times more bytes in memory than in uncompressed Parquet (a
    claims row group of ~950k rows, 65 MB of uncompressed Parquet, took 3.4 GB to read
    with Polars on one thread)."""
    f = pq.ParquetFile(max(sources, key=lambda s: s.stat().st_size))
    return _sample_row_group_rows(f.read_row_group(0))


def _sample_row_group_rows(sample: pa.Table) -> int:
    """_row_group_rows from its sample, the largest group file's first row group."""
    return max(1, int(COMPACT_ROW_GROUP_BYTES * sample.num_rows / max(sample.nbytes, 1)))


def _write_file(
    dst: Path,
    schema: pa.Schema,
    batches,
    row_group_rows: int,
    sorting_columns=None,
    scratch: bool = False,
) -> int:
    """Write record batches to `dst` in row groups of `row_group_rows` rows, declaring
    `sorting_columns` (pq.SortingColumn list) in each row group if given. A `scratch`
    file, read once by a later step, is written cheaply: zstd 1, no content-defined
    chunking or page index (the settings for the Hub's files)."""
    tmp = dst.with_suffix(".tmp")
    writer = pq.ParquetWriter(
        tmp,
        schema,
        compression="zstd",
        compression_level=1 if scratch else 3,
        write_page_index=not scratch,
        use_content_defined_chunking=not scratch,
        sorting_columns=sorting_columns,
    )
    rows = 0
    pending: list[pa.RecordBatch] = []
    pending_rows = 0

    def flush():
        nonlocal pending, pending_rows, rows
        if pending:
            t = pa.Table.from_batches(pending, schema)
            writer.write_table(t, row_group_size=t.num_rows)
            rows += t.num_rows
        pending, pending_rows = [], 0

    for batch in batches:
        pending.append(batch)
        pending_rows += batch.num_rows
        if pending_rows >= row_group_rows:
            flush()
    flush()
    writer.close()
    tmp.replace(dst)
    return rows


def _write_file_parallel(
    dst: Path,
    schema: pa.Schema,
    batches,
    row_group_rows: int,
    sorting_columns=None,
    scratch: bool = False,
) -> int:
    """A file for the Hub as _write_file writes one (zstd 3, a page index,
    content-defined chunking, `sorting_columns`), written by pq-glue's writer (arrow-rs)
    on every core: each row group encoded by a writer of its own, in parallel, and glued
    in order. Row groups of `row_group_rows` rows (the last the rest). A `scratch` file,
    read once by a later step, is written cheaply: zstd 1, no content-defined chunking,
    each row group's leaf columns encoded in parallel."""
    tmp = dst.with_suffix(".tmp")
    sorting = [(c.column_index, c.descending, c.nulls_first) for c in sorting_columns or []]
    rows = pq_glue.write_parquet(
        pa.Table.from_batches(list(batches), schema),
        tmp,
        engine="parallel",
        row_group_size=row_group_rows,
        compression="zstd",
        compression_level=1 if scratch else 3,
        content_defined_chunking=not scratch,
        sorting_columns=sorting or None,
    )
    tmp.replace(dst)
    return rows


def _source_batches(run: list[Path]):
    """The group files' record batches, a row group at a time. Not cast to the key's
    schema (_key_schema checks every group file has it): RecordBatch.cast corrupts the
    nested structs of claims ("Struct child array has length smaller than expected")."""
    for src in run:
        f = pq.ParquetFile(src)
        for i in range(f.metadata.num_row_groups):
            yield from f.read_row_group(i).to_batches()


def _deduplicated(sources: list[Path], schema: pa.Schema) -> dict[str, pa.Table]:
    """Each group file's rows not already in an earlier group file of the key, by name."""
    df = (
        pl.scan_parquet(sources, include_file_paths="_source")
        .unique(subset=schema.names, keep="first", maintain_order=True)
        .collect()
    )
    return {
        Path(src).name: part.drop("_source").to_arrow().cast(schema)
        for (src,), part in df.partition_by("_source", as_dict=True).items()
    }


def _files_path(table: Table) -> Path:
    return _out_dir(table) / "files.jsonl"


def read_files(table: Table) -> dict[tuple[str, str], dict]:
    """The output files written and checked so far, by (key, name)."""
    path = _files_path(table)
    if not path.exists():
        return {}
    entries = [json.loads(line) for line in path.read_text().splitlines() if line]
    return {(e["key"], e["name"]): e for e in entries}  # rewritten again: keep the latest


def _file_entry(dst: Path, rows: int) -> dict:
    return {"name": dst.name, "rows": rows, "bytes": dst.stat().st_size, "sha256": _sha256(dst)}


def _reusable(prior: dict | None, run: list[Path], dst: Path) -> bool:
    """A file written and checked before a restart, from the same group files, and
    unchanged since (same size and hash)."""
    return (
        prior is not None
        and prior["sources"] == [s.name for s in run]
        and dst.exists()
        and dst.stat().st_size == prior["bytes"]
        and _sha256(dst) == prior["sha256"]
    )


def _key_schema(table: Table, key: str, sources: list[Path]) -> pa.Schema:
    schema = pq.read_schema(sources[0]).remove_metadata()
    for src in sources[1:]:
        if not pq.read_schema(src).remove_metadata().equals(schema):
            raise RuntimeError(f"[compact] {table}/{key}: {src.name} has another schema")
    return schema


def _write_run(table: Table, key: str, run: list[Path], dst: Path, schema: pa.Schema) -> dict:
    """Glue one output file from its run of group files, copying their encoded row groups
    without decoding them (pq_glue.concat), and check it against them (run in a worker
    process). The file is the sort's input alone, never uploaded, so nothing needs it
    re-encoded: the sort reads it a row group at a time and writes the Hub's files."""
    tmp = dst.with_suffix(".tmp")
    rows = pq_glue.concat([str(p) for p in run], str(tmp))
    tmp.replace(dst)
    _check_glued(table, key, run, dst, schema)
    return _file_entry(dst, rows)


def _link_run(table: Table, key: str, src: Path, dst: Path) -> dict:
    """An output file of one group file is that file: hard linked, not glued (run in the
    parent, a few stats and a hash, not a job). Gluing one file copies all of it, every
    row group and the footer as they are (a group file is itself glued by pq_glue.concat,
    so its copy came out the same bytes: all 2,324 keys of the benchmark's quick tier),
    then checks the copy against it byte for byte; a link is the same bytes by
    construction, nothing copied to go wrong, and saves the copy's disk. Where the file
    system cannot link, it is copied and checked by sha256 against the group file. A
    link shares the group file's inode, but nothing writes a group file or a compacted
    file in place: each is only ever replaced (a download, the sort's files) or removed
    (compact/src once the local copy is checked, the local copy's old files)."""
    dst.unlink(missing_ok=True)  # a link left by an interrupted run, not yet listed
    try:
        os.link(src, dst)
        linked = True
    except OSError:  # another file system, or one without hard links
        shutil.copyfile(src, dst.with_suffix(".tmp"))
        dst.with_suffix(".tmp").replace(dst)
        linked = False
    entry = _file_entry(dst, pq.read_metadata(dst).num_rows)
    if not linked and entry["sha256"] != _sha256(src):
        raise RuntimeError(f"[compact] {table}/{key} ({dst.name}): copy is not {src.name}'s bytes")
    return entry


def _write_deduplicated_key(table: Table, key: str, sources: list[Path], last_chunk: int) -> dict:
    """Rewrite a deduplicated table's key, keeping each distinct row's first occurrence,
    check its files together, and return its manifest entry (run in a worker process)."""
    schema = _key_schema(table, key, sources)
    dedup = _deduplicated(sources, schema)
    # A group file's share of the output: its size scaled by the rows it keeps
    sizes = {}
    for src in sources:
        kept = dedup[src.name].num_rows if src.name in dedup else 0
        sizes[src.name] = src.stat().st_size * kept / pq.ParquetFile(src).metadata.num_rows
    row_group_rows = _row_group_rows(sources)
    dst_dir = _out_dir(table) / key
    dst_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for run in _runs(sources, sizes):
        tables = [dedup[src.name] for src in run if src.name in dedup]
        if not tables:
            continue  # every row of these group files is in an earlier one
        dst = dst_dir / _run_name(run, last_chunk)
        batches = (b for t in tables for b in t.to_batches())
        files.append(_file_entry(dst, _write_file(dst, schema, batches, row_group_rows)))
    _check_deduplicated(table, key, sources, [dst_dir / f["name"] for f in files], schema)
    return {"key": key, "sources": [s.name for s in sources], "files": files}


def _fingerprint(lf: pl.LazyFrame) -> tuple[int, int, int]:
    """Row count and two sums of row hashes, each row hashed with its position under two
    seeds, so any changed, missing, extra or reordered row changes a sum. Computed by
    Polars' streaming engine: a key is checked without holding its rows in memory."""
    row = pl.struct(pl.all())
    return (
        lf.with_row_index("_row")
        .select(
            pl.len().alias("rows"),
            row.hash(seed=0).sum().alias("h0"),
            row.hash(seed=1).sum().alias("h1"),
        )
        .collect(engine="streaming")
        .row(0)
    )


def _compare(
    table: Table,
    key: str,
    name: str,
    outputs: list[Path],
    want: pl.LazyFrame,
    schema: pa.Schema,
) -> None:
    """The output files have the key's schema, and read back with Polars (a reader
    independent of pyarrow, which wrote them) the rows of `want`, in the same order."""
    for out in outputs:
        if not pq.read_schema(out).remove_metadata().equals(schema):
            raise RuntimeError(f"[compact] {table}/{key}: {out.name} has another schema")
    got_fp, want_fp = _fingerprint(pl.scan_parquet(outputs)), _fingerprint(want)
    if got_fp != want_fp:
        raise RuntimeError(
            f"[compact] {table}/{key} ({name}): rows differ from the group files"
            f" (rows, hash sums: wrote {got_fp}, expected {want_fp})"
        )


def _chunk_span(column) -> tuple[int, int]:
    """A column chunk's encoded bytes in its file: from its first page (the dictionary
    page, when it has one) for its compressed length."""
    start = column.data_page_offset
    if column.has_dictionary_page and column.dictionary_page_offset is not None:
        start = min(start, column.dictionary_page_offset)
    return start, column.total_compressed_size


def _check_glued(table: Table, key: str, run: list[Path], out: Path, schema: pa.Schema) -> None:
    """The glued file is its run of group files' row groups, in order: the key's schema,
    each row group's row count, and every column chunk's encoded bytes the same as its
    source's, compared byte for byte without decoding them (the file is glued by copying
    them; decoding both sides to compare their rows could find nothing this does not)."""
    if not pq.read_schema(out).remove_metadata().equals(schema):
        raise RuntimeError(f"[compact] {table}/{key}: {out.name} has another schema")
    got = pq.ParquetFile(out).metadata
    sources = [(src, pq.ParquetFile(src).metadata) for src in run]
    pairs = [(src, meta.row_group(i)) for src, meta in sources for i in range(meta.num_row_groups)]
    if got.num_row_groups != len(pairs):
        raise RuntimeError(f"[compact] {table}/{key} ({out.name}): {got.num_row_groups} row groups,"
                           f" not the group files' {len(pairs)}")
    with open(out, "rb") as fo:
        for g, (src, want) in enumerate(pairs):
            have = got.row_group(g)
            if have.num_rows != want.num_rows or have.num_columns != want.num_columns:
                raise RuntimeError(f"[compact] {table}/{key} ({out.name}): row group {g} differs from {src.name}'s")
            with open(src, "rb") as fs:
                for c in range(have.num_columns):
                    (o, n), (s, m) = _chunk_span(have.column(c)), _chunk_span(want.column(c))
                    fo.seek(o)
                    fs.seek(s)
                    if n != m or fo.read(n) != fs.read(m):
                        raise RuntimeError(f"[compact] {table}/{key} ({out.name}): row group {g}, column {c}"
                                           f" is not {src.name}'s bytes")


def _check_deduplicated(
    table: Table, key: str, sources: list[Path], outputs: list[Path], schema: pa.Schema
) -> None:
    """The key's files have each distinct row of its group files once, in order of first
    occurrence."""
    want = pl.scan_parquet(sources).unique(keep="first", maintain_order=True)
    _compare(table, key, "all files", outputs, want, schema)


def read_manifest(table: Table) -> dict[str, dict]:
    path = _manifest_path(table)
    if not path.exists():
        return {}
    entries = [json.loads(line) for line in path.read_text().splitlines() if line]
    return {e["key"]: e for e in entries}  # a key rewritten again: keep the latest


def _append_manifest(table: Table, entry: dict) -> None:
    with _manifest_path(table).open("a") as f:
        f.write(json.dumps(entry) + "\n")


def rewrite_table(table: Table, last_chunk: int) -> None:
    """Rewrite every key not yet in the manifest, FINALISE_WORKERS jobs at once: each
    output file a job, or each key for a deduplicated table. Resumable per key, and per
    file within a key (files.jsonl)."""
    done = read_manifest(table)
    keys = _local_keys(table)
    todo = {
        key: sources
        for key, sources in keys.items()
        if not (key in done and done[key]["sources"] == [s.name for s in sources])
    }
    print(f"[compact] {table}: rewriting {len(todo)} of {len(keys)} keys", flush=True)
    if table in DEDUPLICATE:
        jobs = {key: (table, key, sources, last_chunk) for key, sources in todo.items()}
        for _, entry in in_parallel(_write_deduplicated_key, jobs, f"[compact] {table}", "key"):
            _append_manifest(table, entry)
        print(f"[compact] {table}: written", flush=True)
        return
    checked = read_files(table)
    files: dict[str, list[dict | None]] = {}
    runs: dict[str, list[list[Path]]] = {}
    jobs = {}
    links = {}
    from .sort_by_id import sorted_in_memory  # (sort_by_id imports this module)

    for key, sources in tqdm(todo.items(), desc=f"[compact] {table}: planning", unit="key"):
        if len(sources) > 1 and sorted_in_memory(table, sources):
            # Read whole by the sort, which plans it from its rows alone: its group files
            # linked one by one, not glued into files that would only be read once
            # (their schemas are checked as the sort reads them, _read_key)
            schema = None
            runs[key] = [[src] for src in sources]
        else:
            # The key's group files must share a schema; one alone has nothing to differ from
            schema = _key_schema(table, key, sources) if len(sources) > 1 else None
            sizes = {src.name: float(src.stat().st_size) for src in sources}
            runs[key] = _runs(sources, sizes)
        dst_dir = _out_dir(table) / key
        dst_dir.mkdir(parents=True, exist_ok=True)
        files[key] = [None] * len(runs[key])
        for j, run in enumerate(runs[key]):
            dst = dst_dir / _run_name(run, last_chunk)
            prior = checked.get((key, dst.name))
            if _reusable(prior, run, dst):
                files[key][j] = {k: prior[k] for k in ("name", "rows", "bytes", "sha256")}
            elif len(run) == 1:
                links[(key, j)] = (table, key, run[0], dst)
            else:
                jobs[(key, j)] = (table, key, run, dst, schema)

    def finish(key: str) -> None:
        if all(f is not None for f in files[key]):
            _append_manifest(table, {"key": key, "sources": [s.name for s in todo[key]], "files": files[key]})

    def record(key: str, j: int, entry: dict) -> None:
        line = {"key": key, "sources": [s.name for s in runs[key][j]], **entry}
        with _files_path(table).open("a") as f:
            f.write(json.dumps(line) + "\n")
        files[key][j] = entry
        finish(key)

    for key in todo:
        finish(key)  # every file reused
    n_files = sum(len(r) for r in runs.values())
    # Most keys of a run of few groups have one group file: linked here, no worker
    # started for them (the quick tier's 2,324 keys took 2.9 s as a job each, its one
    # claims key alone spawning a worker of its own: the import of Polars and pyarrow)
    for (key, j), args in tqdm(links.items(), desc=f"[compact] {table}: linking", unit="file"):
        record(key, j, _link_run(*args))
    desc = f"[compact] {table}: writing"
    # A claims file is heavy (a worker replaced after each); other tables' are light
    per_child = 1 if table == Table.CLAIMS else None
    for (key, j), entry in in_parallel(_write_run, jobs, desc, "file", n_files, tasks_per_child=per_child):
        record(key, j, entry)
    print(f"[compact] {table}: written", flush=True)


def _same(info, local: Path, sha256: str) -> bool:
    if info.size != local.stat().st_size:
        return False
    if info.lfs is not None:
        return info.lfs.sha256 == sha256
    return info.blob_id == _git_blob_sha1(local)


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
    print(f"[compact] {table}: metadata written", flush=True)


def keep_as_hub_copy(table: Table) -> None:
    """Move each key's new files to HUB_COPY_DIR/{table}/{key}, the sort's input,
    removing any other file there and its download record. Safe to repeat."""
    dst_root = HUB_COPY_DIR / table
    dst_root.mkdir(parents=True, exist_ok=True)
    for key, entry in read_manifest(table).items():
        names = {f["name"] for f in entry["files"]}
        src, dst = _out_dir(table) / key, dst_root / key
        # A file's record from an earlier download would make it be downloaded again
        shutil.rmtree(dst_root / ".cache" / "huggingface" / "download" / key, ignore_errors=True)
        # A key not in the copy yet whose directory holds its new files alone (every key,
        # unless interrupted) is moved whole, one rename: per file, a move into a new
        # directory and the emptied one's removal were most of this stage's ~2,300 keys
        if not dst.exists() and src.is_dir() and set(os.listdir(src)) == names:
            src.rename(dst)
            continue
        dst.mkdir(parents=True, exist_ok=True)
        for p in dst.glob("*.parquet"):
            if p.name not in names:
                p.unlink()
        for name in names:
            if (src / name).exists():
                (src / name).replace(dst / name)
        shutil.rmtree(src, ignore_errors=True)
    print(f"[compact] {table}: new files moved to {dst_root}", flush=True)


def compact_table(table: Table, repo_id: str, state_dir: Path) -> None:
    """Run the table's remaining stages."""
    stage = last_stage(state_dir, table)
    done = STAGES.index(stage) if stage else -1
    if stage == "done":
        print(f"[compact] {table}: already compacted ({repo_id})", flush=True)
        return
    resuming = f", resuming after stage {stage!r}" if stage else ""
    print(f"[compact] Compacting {table} ({repo_id}){resuming}", flush=True)
    if done < STAGES.index("downloaded"):
        if not (RELEASE and table == Table.CLAIMS_LABELS):  # built locally, never uploaded
            download(table, repo_id)
        record_stage(state_dir, table, "downloaded")
    if done < STAGES.index("written"):
        rewrite_table(table, last_chunk(state_dir))
        record_stage(state_dir, table, "written")
    if done < STAGES.index("done"):
        write_metadata(table)
        keep_as_hub_copy(table)
        if CLEAN_UP_LOCAL:
            shutil.rmtree(_src_dir(table), ignore_errors=True)
        record_stage(state_dir, table, "done")
    print(f"[compact] {table}: complete", flush=True)
