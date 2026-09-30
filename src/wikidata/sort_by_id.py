"""Sort each table's repo on the Hub by id, once it is compacted.

Compaction leaves each key's rows in source chunk order, which runs through the id space
several times, so a reader looking up an id can skip no file and few row groups. This
stage sorts each key's rows by its sort column (`id`, or `ref` for claims_labels) in
string order, stably so each id's rows keep their order, across all the key's files, and
declares the order in each row group (`sorting_columns`). Files are renamed
`part-{i}-of-{n}.parquet` (docs/journal/2026-09-30-sort-by-id.md).

A table goes through these stages, recorded in `sort.jsonl` in the state dir:

- sourced: HUB_COPY_DIR/{table} (from download-wikidata) has exactly the Hub's files,
  by size and sha256; the emptied compaction source directory is then removed
- written: each key sorted to SORT_DIR/out/{table}, checked, and listed in the manifest.
  A key over SORT_IN_MEMORY_BYTES (claims) goes through id-range buckets: bucketed in one
  pass, each bucket sorted, and the sorted buckets packed into files, each step resumable
- committed: each key's part files added and its old files deleted in one commit
- verified: the Hub has exactly the part files of every key
- done: the metadata JSON rewritten, and HUB_COPY_DIR/{table} holds the sorted files
"""

from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import (
    CommitOperationAdd,
    CommitOperationDelete,
    HfApi,
    snapshot_download,
)
from tqdm import tqdm

from .compact import _fingerprint, _row_group_rows, _same, _write_file
from .compact import _src_dir as _compact_src_dir
from .compact import last_stage as compact_stage
from .config import (
    CLEAN_UP_LOCAL,
    COMPACT_COMMIT_MAX_ADDS,
    COMPACT_COMMIT_MAX_OPS,
    COMPACT_DOWNLOAD_WORKERS,
    COMPACT_FILE_BYTES,
    DATASET_CARDS_METADATA,
    HUB_COPY_DIR,
    SORT_BUCKET_BYTES,
    SORT_DIR,
    SORT_IN_MEMORY_BYTES,
    Table,
)
from .push.core import _sha256

STAGES = ["sourced", "written", "committed", "verified", "done"]

SORT_COLUMN = {Table.CLAIMS_LABELS: "ref"}

# A key's file on the Hub, compacted or sorted
FILE_RE = re.compile(r"^([^/]+)/((?:chunks-\d{4}-\d{4}|part-\d+-of-\d+)\.parquet)$")


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


def _local_keys(table: Table) -> dict[str, list[Path]]:
    """Each key's files in the local copy, in name order (chunk order for compacted
    files)."""
    keys: dict[str, list[Path]] = {}
    for p in sorted(_src_dir(table).glob("*/*.parquet")):
        keys.setdefault(p.parent.name, []).append(p)
    return keys


def _remote_files(repo_id: str, api: HfApi) -> dict[str, dict[str, object]]:
    """Each key's Parquet files on the Hub, by name."""
    keys: dict[str, dict[str, object]] = {}
    for info in api.list_repo_tree(repo_id, repo_type="dataset", recursive=True):
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


# Checks


def _schema(sources: list[Path]) -> pa.Schema:
    schema = pq.read_schema(sources[0]).remove_metadata()
    for src in sources[1:]:
        if not pq.read_schema(src).remove_metadata().equals(schema):
            raise RuntimeError(f"[sort] {src} has another schema")
    return schema


def _sums(row: pl.Expr) -> list[pl.Expr]:
    return [
        pl.len().alias("rows"),
        row.hash(seed=0).sum().alias("h0"),
        row.hash(seed=1).sum().alias("h1"),
    ]


def _multiset(lf: pl.LazyFrame) -> tuple[int, int, int]:
    """Row count and two sums of row hashes, not position: equal for the same rows in
    any order. Streaming."""
    row = pl.struct(pl.all())
    return (
        lf.select(_sums(row))
        .collect(engine="streaming")
        .row(0)
    )


def _ranked(lf: pl.LazyFrame, col: str) -> tuple[int, int, int]:
    """As _multiset, each row hashed with its rank among its id's rows: equal for the
    same rows with each id's rows in the same order, whatever the order of the ids."""
    lf = lf.with_columns(pl.int_range(pl.len()).over(col).alias("_rank"))
    row = pl.struct(pl.all())
    return lf.select(_sums(row)).collect().row(0)


def _check_equal(what: str, got, want) -> None:
    if got != want:
        raise RuntimeError(f"[sort] {what}: rows differ (rows, hash sums: {got}, expected {want})")


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


def _sorted(t: pa.Table, col: str) -> pa.Table:
    return t.take(pc.sort_indices(t, [(col, "ascending")]))  # a stable sort


def _sorting(schema: pa.Schema, col: str) -> list:
    return [pq.SortingColumn(schema.get_field_index(col))]


def _rebatched(batches, schema: pa.Schema, rows: int):
    """The batches re-cut into batches of exactly `rows` rows (the last may be fewer),
    across the input batches' boundaries, so row groups are the same size."""
    pending: list[pa.RecordBatch] = []
    n = 0
    for b in batches:
        pending.append(b)
        n += b.num_rows
        while n >= rows:
            t = pa.Table.from_batches(pending, schema)
            yield from t.slice(0, rows).to_batches()
            pending = t.slice(rows).to_batches()
            n -= rows
    yield from pending


def _file_batches(files: list[Path]):
    for f in files:
        pf = pq.ParquetFile(f)
        for i in range(pf.metadata.num_row_groups):
            yield from pf.read_row_group(i).to_batches()


def _entry(dst: Path, rows: int) -> dict:
    return {"name": dst.name, "rows": rows, "bytes": dst.stat().st_size, "sha256": _sha256(dst)}


def sort_in_memory(table: Table, key: str, sources: list[Path]) -> list[dict]:
    """Sort a key read whole into `ceil(bytes / COMPACT_FILE_BYTES)` files of equal rows,
    and check them against the key's files."""
    col = sort_column(table)
    schema = _schema(sources)
    rg_rows = _row_group_rows(sources)
    t = _sorted(_read(sources), col)
    n = max(1, math.ceil(sum(s.stat().st_size for s in sources) / COMPACT_FILE_BYTES))
    per_file = math.ceil(t.num_rows / n)
    dst_dir = _out_dir(table) / key
    dst_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for i in range(n):
        dst = dst_dir / _part_name(i, n)
        part = t.slice(i * per_file, per_file)
        rows = _write_file(
            dst, schema, part.to_batches(max_chunksize=rg_rows), rg_rows, _sorting(schema, col)
        )
        files.append(_entry(dst, rows))
    del t
    outputs = [dst_dir / f["name"] for f in files]
    _check_sorted(f"{table}/{key}", outputs, col, schema)
    _check_equal(
        f"{table}/{key}",
        _ranked(pl.scan_parquet(outputs), col),
        _ranked(pl.scan_parquet(sources), col),
    )
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


def bucket_key(table: Table, key: str, sources: list[Path]) -> dict:
    """Write each of the key's rows to its bucket file, in source order, once: the
    buckets and their rows are recorded in `buckets.json`, and a restart before that
    redoes the whole pass."""
    bdir = _bucket_dir(table, key)
    record = bdir / "buckets.json"
    if record.exists():
        return json.loads(record.read_text())
    col = sort_column(table)
    schema = _schema(sources)
    n = max(1, math.ceil(sum(s.stat().st_size for s in sources) / SORT_BUCKET_BYTES))
    print(f"[sort] {table}/{key}: finding {n} bucket boundaries", flush=True)
    bounds = _boundaries(sources, col, n)
    lookup = pl.Series(bounds, dtype=pl.String)
    shutil.rmtree(bdir, ignore_errors=True)
    bdir.mkdir(parents=True)
    paths = [bdir / f"bucket-{i:05d}.parquet" for i in range(len(bounds) + 1)]
    writers = [pq.ParquetWriter(p, schema, compression="zstd", compression_level=1) for p in paths]
    rows = [0] * len(paths)
    groups = [(s, i) for s in sources for i in range(pq.ParquetFile(s).metadata.num_row_groups)]
    try:
        for src, i in tqdm(groups, desc=f"[sort] {table}/{key}: bucketing", unit="row group"):
            rg = pq.ParquetFile(src).read_row_group(i)
            ids = pl.from_arrow(rg.column(col)).cast(pl.String)
            b = lookup.search_sorted(ids, side="right").to_numpy()
            order = np.argsort(b, kind="stable")  # each bucket's rows in source order
            rg, b = rg.take(pa.array(order)), b[order]
            starts = np.flatnonzero(np.r_[True, b[1:] != b[:-1]])
            for s, e in zip(starts, np.r_[starts[1:], len(b)]):
                writers[b[s]].write_table(rg.slice(s, e - s))
                rows[b[s]] += int(e - s)
    finally:
        for w in writers:
            w.close()
    buckets = {"bounds": bounds, "rows": rows, "sources": [s.name for s in sources]}
    record.write_text(json.dumps(buckets))
    return buckets


def sort_buckets(table: Table, key: str, sources: list[Path], buckets: dict) -> list[dict]:
    """Sort each bucket and write it with the final settings, checked against its bucket
    file; each is listed in `sorted.jsonl` and reused on a restart."""
    col = sort_column(table)
    schema = _schema(sources)
    rg_rows = _row_group_rows(sources)
    bdir = _bucket_dir(table, key)
    log = bdir / "sorted.jsonl"
    done = {e["name"]: e for e in _read_jsonl(log)}
    out = []
    for i in tqdm(range(len(buckets["rows"])), desc=f"[sort] {table}/{key}: sorting", unit="bucket"):
        src, dst = bdir / f"bucket-{i:05d}.parquet", bdir / f"sorted-{i:05d}.parquet"
        prior = done.get(dst.name)
        if prior and dst.exists() and dst.stat().st_size == prior["bytes"]:
            out.append(prior)
            continue
        t = _sorted(pq.read_table(src), col)
        rows = _write_file(
            dst, schema, t.to_batches(max_chunksize=rg_rows), rg_rows, _sorting(schema, col)
        )
        del t
        if rows != buckets["rows"][i]:
            raise RuntimeError(f"[sort] {table}/{key}: {dst.name} has {rows} rows")
        _check_sorted(f"{table}/{key}", [dst], col, schema)
        _check_equal(
            f"{table}/{key} {dst.name}",
            _ranked(pl.scan_parquet(dst), col),
            _ranked(pl.scan_parquet(src), col),
        )
        entry = {"name": dst.name, "rows": rows, "bytes": dst.stat().st_size}
        _append_jsonl(log, entry)
        src.unlink()  # no longer needed once its sorted bucket is checked
        out.append(entry)
    return out


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


def pack_key(
    table: Table, key: str, sources: list[Path], sorted_buckets: list[dict]
) -> list[dict]:
    """Rewrite consecutive sorted buckets into the key's part files of about even size,
    row groups re-cut across bucket boundaries; each checked against its buckets and
    listed in `files.jsonl`, and reused on a restart."""
    col = sort_column(table)
    schema = _schema(sources)
    rg_rows = _row_group_rows(sources)
    bdir = _bucket_dir(table, key)
    log = _out_dir(table) / "files.jsonl"
    done = {(e["key"], e["name"]): e for e in _read_jsonl(log)}
    plan = _pack([b["bytes"] for b in sorted_buckets])
    dst_dir = _out_dir(table) / key
    dst_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for i, idx in enumerate(tqdm(plan, desc=f"[sort] {table}/{key}: packing", unit="file")):
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
            files.append({k: prior[k] for k in ("name", "rows", "bytes", "sha256")})
            continue
        rows = _write_file(
            dst,
            schema,
            _rebatched(_file_batches(parts), schema, rg_rows),
            rg_rows,
            _sorting(schema, col),
        )
        _check_equal(
            f"{table}/{key} {dst.name}",
            _fingerprint(pl.scan_parquet(dst)),
            _fingerprint(pl.scan_parquet(parts)),
        )
        entry = _entry(dst, rows)
        _append_jsonl(log, {"key": key, "buckets": [p.name for p in parts], **entry})
        files.append(entry)
    outputs = [dst_dir / f["name"] for f in files]
    print(f"[sort] {table}/{key}: checking {len(outputs)} files against the source", flush=True)
    _check_sorted(f"{table}/{key}", outputs, col, schema)
    _check_equal(
        f"{table}/{key}",
        _multiset(pl.scan_parquet(outputs)),
        _multiset(pl.scan_parquet(sources)),
    )
    return files


def sort_bucketed(table: Table, key: str, sources: list[Path]) -> list[dict]:
    buckets = bucket_key(table, key, sources)
    if buckets["sources"] != [s.name for s in sources]:
        raise RuntimeError(f"[sort] {table}/{key}: buckets were made from other files")
    return pack_key(table, key, sources, sort_buckets(table, key, sources, buckets))


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

    large = {k: s for k, s in todo.items() if sum(f.stat().st_size for f in s) > SORT_IN_MEMORY_BYTES}
    for key, sources in large.items():
        print(f"[sort] {table}/{key}: {len(sources)} files, sorting through buckets", flush=True)
        record(key, sources, sort_bucketed(table, key, sources))
    small = {k: s for k, s in todo.items() if k not in large}
    for key, sources in tqdm(small.items(), desc=f"[sort] {table}", unit="key", disable=not small):
        record(key, sources, sort_in_memory(table, key, sources))
    print(f"[sort] {table}: written", flush=True)


def _key_done(table: Table, entry: dict, remote: dict[str, object]) -> bool:
    """The key's files on the Hub are exactly its sorted files."""
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
    keys, a key's additions and deletions always in the same commit."""
    print(f"[sort] {table}: committing to {repo_id}", flush=True)
    manifest = read_manifest(table)
    remote = _remote_files(repo_id, api)
    if set(remote) != set(manifest):
        raise RuntimeError(f"[sort] {table}: keys on the Hub differ from the manifest")
    todo = [e for k, e in manifest.items() if not _key_done(table, e, remote[k])]
    batch: list = []
    batch_keys: list[str] = []

    def commit():
        nonlocal batch, batch_keys
        if batch:
            api.create_commit(
                repo_id,
                repo_type="dataset",
                operations=batch,
                commit_message=f"Sort {len(batch_keys)} keys by id ({batch_keys[0]} to {batch_keys[-1]})",
            )
            print(f"[sort] {table}: committed {len(batch_keys)} keys", flush=True)
        batch, batch_keys = [], []

    for entry in todo:
        ops = _key_operations(table, entry, remote[entry["key"]])
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
        source_table(table, repo_id, api)
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
