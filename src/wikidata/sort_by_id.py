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
  A key over SORT_IN_MEMORY_BYTES (claims) goes through id-range buckets: each source
  file bucketed, each bucket sorted, and the sorted buckets packed into files, each step
  SORT_WORKERS at once and resumable (docs/journal/2026-10-05-sort-speed.md)
- committed: each key's part files added and its old files deleted in one commit
- verified: the Hub has exactly the part files of every key
- done: the metadata JSON rewritten, and HUB_COPY_DIR/{table} holds the sorted files
"""

from __future__ import annotations

import json
import math
import multiprocessing
import re
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    SORT_IN_MEMORY_BYTES,
    SORT_WORKERS,
    Table,
)
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


def _in_parallel(
    fn, jobs: dict[int, tuple], desc: str, total: int, unit: str, tasks_per_child: int = 8
):
    """Run `fn(*jobs[i])` for each i, SORT_WORKERS at a time in spawned processes, each
    process replaced after `tasks_per_child` jobs (its memory then freed; spawned, not
    forked, as the parent may have threads), yielding `(i, result)` as each finishes. A
    failure cancels the jobs not yet started and raises once those running finish."""
    if not jobs:
        return
    ctx = multiprocessing.get_context("spawn")
    with (
        ProcessPoolExecutor(
            SORT_WORKERS, mp_context=ctx, max_tasks_per_child=tasks_per_child
        ) as pool,
        tqdm(total=total, initial=total - len(jobs), desc=desc, unit=unit) as bar,
    ):
        futures = {pool.submit(fn, *args): i for i, args in jobs.items()}
        try:
            for fut in as_completed(futures):
                yield futures[fut], fut.result()
                bar.update()
        except BaseException:
            pool.shutdown(cancel_futures=True)
            raise


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


def _ranked(lf: pl.LazyFrame, col: str) -> tuple[int, int, int]:
    """Row count and two sums of row hashes, each row hashed with its rank among its id's
    rows: equal for the same rows with each id's rows in the same order, whatever the
    order of the ids."""
    lf = lf.with_columns(pl.int_range(pl.len()).over(col).alias("_rank"))
    row = pl.struct(pl.all())
    return lf.select(_sums(row)).collect().row(0)


_HALF = pl.lit(2**32, dtype=pl.UInt64)


def _additive_exprs(row: pl.Expr) -> list[pl.Expr]:
    exprs = [pl.len().alias("n")]
    for seed in (0, 1):
        h = row.hash(seed=seed)
        exprs += [(h % _HALF).sum().alias(f"lo{seed}"), (h // _HALF).sum().alias(f"hi{seed}")]
    return exprs


def _additive(lf: pl.LazyFrame) -> list[int]:
    """Row count and, for two seeds, the sums of the low and of the high 32 bits of each
    row's hash: exact (no overflow below 2**32 rows), so the sums of the parts of any
    split of the rows add up to the whole's (see _add). Equal for the same rows in any
    order. Streaming."""
    return list(lf.select(_additive_exprs(pl.struct(pl.all()))).collect(engine="streaming").row(0))


def _ranked_and_additive(lf: pl.LazyFrame, col: str) -> tuple[tuple[int, int, int], list[int]]:
    """_ranked and _additive of the same rows, in one read."""
    lf = lf.with_columns(pl.int_range(pl.len()).over(col).alias("_rank"))
    exprs = _sums(pl.struct(pl.all())) + _additive_exprs(pl.struct(pl.exclude("_rank")))
    row = lf.select(exprs).collect().row(0)
    return row[:3], list(row[3:])


def _add(parts) -> list[int]:
    """The element-wise sum of _additive results."""
    return [sum(xs) for xs in zip(*parts, strict=True)]


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


def _fragment_name(i: int, j: int, n_sources: int) -> str:
    """Bucket i's rows from source file j, the source index padded to the digits of the
    key's last source index."""
    return f"bucket-{i:05d}-{j:0{len(str(n_sources - 1))}d}.parquet"


def _bucket_files(bdir: Path, i: int, buckets: dict) -> list[Path]:
    """Bucket i's files, in source order: one per source file, or one in all (buckets
    made before 2026-10-05, without "fragments")."""
    n = buckets.get("fragments")
    if n is None:
        return [bdir / f"bucket-{i:05d}.parquet"]
    return [bdir / _fragment_name(i, j, n) for j in range(n)]


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
    bucket's rows buffered to row groups of about `flush_rows` rows; return its rows per
    bucket, and its _additive sums as Polars reads it (run in a worker process)."""
    col = sort_column(table)
    lookup = pl.Series(bounds, dtype=pl.String)
    n = len(bounds) + 1
    paths = [bdir / _fragment_name(i, j, n_sources) for i in range(n)]
    writers = [pq.ParquetWriter(p, schema, compression="zstd", compression_level=1) for p in paths]
    pending: list[list[pa.Table]] = [[] for _ in range(n)]
    pending_rows = [0] * n
    rows = [0] * n

    def flush(i: int) -> None:
        if pending[i]:
            t = pa.concat_tables(pending[i])
            writers[i].write_table(t, row_group_size=t.num_rows)
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
    return {"source": src.name, "rows": rows, "sums": _additive(pl.scan_parquet(src)), "at": time.time()}


def bucket_key(table: Table, key: str, sources: list[Path]) -> dict:
    """Write each of the key's rows to its bucket, in source order: each source file to
    its own fragment of every bucket, SORT_WORKERS files at once. The boundaries are in
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
    done = {e["source"]: e for e in _read_jsonl(log) if e["source"] in names}
    rg_rows = _row_group_rows(sources)
    flush_rows = max(1, rg_rows * SORT_BUCKET_WRITE_BYTES // COMPACT_ROW_GROUP_BYTES)
    jobs = {
        j: (table, src, j, len(sources), bdir, bounds, schema, flush_rows)
        for j, src in enumerate(sources)
        if src.name not in done
    }
    desc = f"[sort] {table}/{key}: bucketing"
    # One job per process: a job holds every bucket's buffer
    for _, entry in _in_parallel(_bucket_source, jobs, desc, len(sources), "file", tasks_per_child=1):
        _append_jsonl(log, entry)
        done[entry["source"]] = entry
    buckets = {
        "bounds": bounds,
        "rows": [sum(r) for r in zip(*(done[n]["rows"] for n in names), strict=True)],
        "sources": names,
        "fragments": len(sources),
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
    """Sort one bucket to `dst`, a scratch file read once by packing, checked against
    the bucket's files; with the bucket's _additive sums (run in a worker process)."""
    col = sort_column(table)
    t = _sorted(_read(srcs), col)
    rows = _write_file(
        dst,
        schema,
        t.to_batches(max_chunksize=rg_rows),
        rg_rows,
        _sorting(schema, col),
        scratch=True,
    )
    del t
    if rows != want_rows:
        raise RuntimeError(f"[sort] {table}/{key}: {dst.name} has {rows} rows")
    _check_sorted(f"{table}/{key}", [dst], col, schema)
    want, sums = _ranked_and_additive(pl.scan_parquet(srcs), col)
    _check_equal(f"{table}/{key} {dst.name}", _ranked(pl.scan_parquet(dst), col), want)
    return {"name": dst.name, "rows": rows, "bytes": dst.stat().st_size, "sums": sums}


def sort_buckets(table: Table, key: str, sources: list[Path], buckets: dict) -> list[dict]:
    """Sort each bucket to a scratch file, checked against the bucket's files,
    SORT_WORKERS at once; each is listed in `sorted.jsonl` and reused on a restart."""
    schema = _schema(sources)
    rg_rows = _row_group_rows(sources)
    bdir = _bucket_dir(table, key)
    log = bdir / "sorted.jsonl"
    done = {e["name"]: e for e in _read_jsonl(log)}
    n = len(buckets["rows"])
    out: dict[int, dict] = {}
    jobs = {}
    for i in range(n):
        srcs, dst = _bucket_files(bdir, i, buckets), bdir / f"sorted-{i:05d}.parquet"
        prior = done.get(dst.name)
        if prior and dst.exists() and dst.stat().st_size == prior["bytes"]:
            out[i] = prior
        else:
            jobs[i] = (table, key, srcs, dst, schema, rg_rows, buckets["rows"][i])
    desc = f"[sort] {table}/{key}: sorting"
    for i, entry in _in_parallel(_sort_bucket, jobs, desc, n, "bucket"):
        _append_jsonl(log, entry)
        for src in jobs[i][2]:
            src.unlink()  # no longer needed once checked
        out[i] = entry
    return [out[i] for i in range(n)]


def _pack(sizes: list[int], final_bytes: int) -> list[list[int]]:
    """Consecutive bucket indices for each of `ceil(final_bytes / COMPACT_FILE_BYTES)`
    files, cut where the cumulative size is nearest each multiple of total / n. The
    sorted buckets are scratch files, larger than the files they become: `final_bytes`
    is the key's size with the final settings (its source files')."""
    total = sum(sizes)
    n = max(1, math.ceil(final_bytes / COMPACT_FILE_BYTES))
    cum = np.cumsum(sizes)
    cuts = [0]
    for k in range(1, n):
        c = int(np.argmin(np.abs(cum - total * k / n))) + 1
        if cuts[-1] < c < len(sizes):
            cuts.append(c)
    cuts.append(len(sizes))
    return [list(range(a, b)) for a, b in zip(cuts, cuts[1:])]


def _pack_file(
    table: Table, key: str, dst: Path, parts: list[Path], schema: pa.Schema, rg_rows: int
) -> dict:
    """Write one part file from its sorted buckets, checked against them (run in a worker
    process)."""
    rows = _write_file(
        dst,
        schema,
        _rebatched(_file_batches(parts), schema, rg_rows),
        rg_rows,
        _sorting(schema, sort_column(table)),
    )
    _check_equal(
        f"{table}/{key} {dst.name}",
        _fingerprint(pl.scan_parquet(dst)),
        _fingerprint(pl.scan_parquet(parts)),
    )
    return _entry(dst, rows)


def pack_key(
    table: Table, key: str, sources: list[Path], buckets: dict, sorted_buckets: list[dict]
) -> list[dict]:
    """Rewrite consecutive sorted buckets into the key's part files of about even size,
    row groups re-cut across bucket boundaries; each checked against its buckets and
    listed in `files.jsonl`, and reused on a restart. SORT_WORKERS files at once."""
    col = sort_column(table)
    schema = _schema(sources)
    rg_rows = _row_group_rows(sources)
    bdir = _bucket_dir(table, key)
    log = _out_dir(table) / "files.jsonl"
    done = {(e["key"], e["name"]): e for e in _read_jsonl(log)}
    plan = _pack([b["bytes"] for b in sorted_buckets], sum(s.stat().st_size for s in sources))
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
            jobs[i] = (table, key, dst, parts, schema, rg_rows)
    desc = f"[sort] {table}/{key}: packing"
    for i, entry in _in_parallel(_pack_file, jobs, desc, len(plan), "file"):
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
                revision=HUB_REVISION,
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
