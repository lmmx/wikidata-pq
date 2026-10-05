"""Compact each table's repo on the Hub, once every group is uploaded.

The grouped upload leaves one file per partition key per group, so a key has up to one
file per group, most of them small. Compaction rewrites each key into files of about
COMPACT_FILE_BYTES, split only between groups and named by the chunks they cover, and
replaces the group files on the Hub with them.

A table goes through these stages, recorded in `compact.jsonl` in the state dir so an
interrupted run resumes at the stage it had not finished. Every stage is safe to repeat.

- downloaded: the group files are in COMPACT_DIR/src/{table}
- written: each key is rewritten to COMPACT_DIR/out/{table}, checked against its group
  files, and listed in the table's manifest; within a key, each file checked is listed
  in `files.jsonl` and not rewritten on a restart (see write_key)
- committed: each key's new files are added and its group files deleted in one commit,
  keys batched into commits; a key whose files on the Hub are already its new files is
  skipped
- verified: the Hub has exactly the new files of every key, with the same size and hash
- done: the table's files, bytes and rows per key are in DATASET_CARDS_METADATA
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import (
    CommitOperationAdd,
    CommitOperationDelete,
    HfApi,
    snapshot_download,
)

from .config import (
    HUB_REVISION,
    CLEAN_UP_LOCAL,
    COMPACT_COMMIT_MAX_ADDS,
    COMPACT_COMMIT_MAX_OPS,
    COMPACT_DIR,
    COMPACT_DOWNLOAD_WORKERS,
    COMPACT_FILE_BYTES,
    COMPACT_ROW_GROUP_BYTES,
    DATASET_CARDS_METADATA,
    Table,
)
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
    sample = f.read_row_group(0)
    return max(1, int(COMPACT_ROW_GROUP_BYTES * sample.num_rows / max(sample.nbytes, 1)))


def _write_file(
    dst: Path, schema: pa.Schema, batches, row_group_rows: int, sorting_columns=None
) -> int:
    """Write record batches to `dst` in row groups of `row_group_rows` rows, declaring
    `sorting_columns` (pq.SortingColumn list) in each row group if given."""
    tmp = dst.with_suffix(".tmp")
    writer = pq.ParquetWriter(
        tmp,
        schema,
        compression="zstd",
        compression_level=3,
        write_page_index=True,
        use_content_defined_chunking=True,
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


def _source_batches(run: list[Path]):
    """The group files' record batches, a row group at a time. Not cast to the key's
    schema (write_key checks every group file has it): RecordBatch.cast corrupts the
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


def write_key(
    table: Table,
    key: str,
    sources: list[Path],
    checked: dict[tuple[str, str], dict],
    last_chunk: int,
) -> dict:
    """Rewrite a key's group files, check the result, and return its manifest entry.

    A table not deduplicated has each output file checked as soon as it is written, and
    recorded in `files.jsonl`: a file in `checked` (read from it) is reused, not
    rewritten, so a restart in the middle of a large key (claims has a single key) redoes
    only its unfinished file. A deduplicated table is checked, and so resumed, per key.
    """
    schema = pq.read_schema(sources[0]).remove_metadata()
    for src in sources[1:]:
        if not pq.read_schema(src).remove_metadata().equals(schema):
            raise RuntimeError(f"[compact] {table}/{key}: {src.name} has another schema")
    sizes = {src.name: float(src.stat().st_size) for src in sources}
    dedup = None
    if table in DEDUPLICATE:
        dedup = _deduplicated(sources, schema)
        # A group file's share of the output: its size scaled by the rows it keeps
        for src in sources:
            kept = dedup[src.name].num_rows if src.name in dedup else 0
            sizes[src.name] *= kept / pq.ParquetFile(src).metadata.num_rows
    row_group_rows = _row_group_rows(sources)
    dst_dir = _out_dir(table) / key
    dst_dir.mkdir(parents=True, exist_ok=True)
    files = []
    runs = _runs(sources, sizes)
    for j, run in enumerate(runs, 1):
        dst = dst_dir / _run_name(run, last_chunk)
        if dedup is None:
            prior = checked.get((key, dst.name))
            if _reusable(prior, run, dst):
                files.append({k: prior[k] for k in ("name", "rows", "bytes", "sha256")})
                print(f"[compact] {table}/{key}: {dst.name} already written", flush=True)
                continue
            progress = f"[compact] {table}/{key}: {dst.name} ({j}/{len(runs)})"
            print(f"{progress}: writing", flush=True)
            rows = _write_file(dst, schema, _source_batches(run), row_group_rows)
            print(f"{progress}: checking {rows:,} rows", flush=True)
            _check_file(table, key, run, dst, schema)
            entry = _file_entry(dst, rows)
            with _files_path(table).open("a") as f:
                f.write(json.dumps({"key": key, "sources": [s.name for s in run], **entry}) + "\n")
            files.append(entry)
        else:
            tables = [dedup[src.name] for src in run if src.name in dedup]
            if not tables:
                continue  # every row of these group files is in an earlier one
            batches = (b for t in tables for b in t.to_batches())
            files.append(_file_entry(dst, _write_file(dst, schema, batches, row_group_rows)))
    if dedup is not None:
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


def _check_file(table: Table, key: str, run: list[Path], out: Path, schema: pa.Schema) -> None:
    """The file has the rows of its run of group files."""
    _compare(table, key, out.name, [out], pl.scan_parquet(run), schema)


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


def rewrite_table(table: Table, last_chunk: int) -> None:
    """Rewrite every key not yet in the manifest (resumable per key, and per file within
    a key: see write_key)."""
    done = read_manifest(table)
    checked = read_files(table)
    keys = _local_keys(table)
    todo = sum(k not in done for k in keys)
    print(f"[compact] {table}: rewriting {todo} of {len(keys)} keys", flush=True)
    for i, (key, sources) in enumerate(keys.items(), 1):
        if key in done and done[key]["sources"] == [s.name for s in sources]:
            continue
        entry = write_key(table, key, sources, checked, last_chunk)
        with _manifest_path(table).open("a") as f:
            f.write(json.dumps(entry) + "\n")
        n_in, n_out = len(sources), len(entry["files"])
        print(f"[compact] {table}/{key} ({i}/{len(keys)}): {n_in} -> {n_out} files", flush=True)
    print(f"[compact] {table}: written", flush=True)


def _remote_files(repo_id: str, api: HfApi) -> dict[str, dict[str, object]]:
    """Each key's files on the Hub, by name."""
    keys: dict[str, dict[str, object]] = {}
    for info in api.list_repo_tree(repo_id, repo_type="dataset", recursive=True, revision=HUB_REVISION):
        if m := FILE_RE.match(info.path):
            keys.setdefault(m.group(1), {})[info.path.split("/", 1)[1]] = info
    return keys


def _same(info, local: Path, sha256: str) -> bool:
    if info.size != local.stat().st_size:
        return False
    if info.lfs is not None:
        return info.lfs.sha256 == sha256
    return info.blob_id == _git_blob_sha1(local)


def _key_done(table: Table, entry: dict, remote: dict[str, object]) -> bool:
    """The key's files on the Hub are exactly its new files."""
    names = {f["name"] for f in entry["files"]}
    if set(remote) != names:
        return False
    local = _out_dir(table) / entry["key"]
    return all(_same(remote[f["name"]], local / f["name"], f["sha256"]) for f in entry["files"])


def _key_operations(table: Table, entry: dict, remote: dict[str, object]) -> list:
    key = entry["key"]
    new = {f["name"] for f in entry["files"]}
    unexpected = set(remote) - new - set(entry["sources"])
    if unexpected:
        raise RuntimeError(f"[compact] {table}/{key}: unknown files on the Hub: {unexpected}")
    adds = [
        CommitOperationAdd(
            path_in_repo=f"{key}/{name}", path_or_fileobj=str(_out_dir(table) / key / name)
        )
        for name in sorted(new)
    ]
    deletes = [CommitOperationDelete(path_in_repo=f"{key}/{n}") for n in sorted(set(remote) - new)]
    return adds + deletes


def commit_table(table: Table, repo_id: str, api: HfApi) -> None:
    """Replace each key's group files on the Hub with its new files, one commit per batch
    of keys, a key's additions and deletions always in the same commit."""
    print(f"[compact] {table}: committing to {repo_id}", flush=True)
    manifest = read_manifest(table)
    remote = _remote_files(repo_id, api)
    missing = set(remote) - set(manifest)
    if missing:
        raise RuntimeError(f"[compact] {table}: keys on the Hub not rewritten: {sorted(missing)}")
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
                commit_message=f"Compact {len(batch_keys)} keys ({batch_keys[0]} to {batch_keys[-1]})",
            )
            print(f"[compact] {table}: committed {len(batch_keys)} keys", flush=True)
        batch, batch_keys = [], []

    for entry in todo:
        ops = _key_operations(table, entry, remote.get(entry["key"], {}))
        n_adds = sum(isinstance(op, CommitOperationAdd) for op in batch + ops)
        if batch and (n_adds > COMPACT_COMMIT_MAX_ADDS or len(batch) + len(ops) > COMPACT_COMMIT_MAX_OPS):
            commit()
        batch += ops
        batch_keys.append(entry["key"])
    commit()
    print(f"[compact] {table}: {len(todo)} keys committed, {len(manifest) - len(todo)} already", flush=True)


def verify_table(table: Table, repo_id: str, api: HfApi) -> None:
    """The Hub has exactly the new files of every key, and no other key."""
    print(f"[compact] {table}: verifying {repo_id}", flush=True)
    manifest = read_manifest(table)
    remote = _remote_files(repo_id, api)
    if set(remote) != set(manifest):
        raise RuntimeError(f"[compact] {table}: keys differ from the manifest on the Hub")
    bad = [k for k, e in manifest.items() if not _key_done(table, e, remote[k])]
    if bad:
        raise RuntimeError(f"[compact] {table}: keys not compacted on the Hub: {bad}")
    print(f"[compact] {table}: verified {len(manifest)} keys", flush=True)


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
    path.write_text(json.dumps(ordered, indent=2) + "\n")
    print(f"[compact] {table}: metadata written", flush=True)


def compact_table(
    table: Table, repo_id: str, state_dir: Path, api: HfApi | None = None
) -> None:
    """Run the table's remaining stages."""
    api = api or HfApi()
    stage = last_stage(state_dir, table)
    done = STAGES.index(stage) if stage else -1
    if stage == "done":
        print(f"[compact] {table}: already compacted ({repo_id})", flush=True)
        return
    resuming = f", resuming after stage {stage!r}" if stage else ""
    print(f"[compact] Compacting {table} ({repo_id}){resuming}", flush=True)
    if done < STAGES.index("downloaded"):
        download(table, repo_id)
        record_stage(state_dir, table, "downloaded")
    if done < STAGES.index("written"):
        rewrite_table(table, last_chunk(state_dir))
        record_stage(state_dir, table, "written")
    if done < STAGES.index("committed"):
        commit_table(table, repo_id, api)
        record_stage(state_dir, table, "committed")
    if done < STAGES.index("verified"):
        verify_table(table, repo_id, api)
        record_stage(state_dir, table, "verified")
    if done < STAGES.index("done"):
        write_metadata(table)
        if CLEAN_UP_LOCAL:
            shutil.rmtree(_src_dir(table), ignore_errors=True)
            for key_dir in _out_dir(table).iterdir():
                if key_dir.is_dir():
                    shutil.rmtree(key_dir)
        record_stage(state_dir, table, "done")
    print(f"[compact] {table}: complete", flush=True)
