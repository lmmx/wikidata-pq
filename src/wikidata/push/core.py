"""Close a group of partitioned chunks: merge, upload, verify, clean up.

Each stage is recorded in the group ledger once complete, so an interrupted group resumes
at the stage it had not finished. Every stage is safe to repeat.
"""

from __future__ import annotations

import hashlib
import json
import os
import resource
import shutil
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from threading import Lock

import polars as pl
import pq_glue
import pyarrow.parquet as pq
from huggingface_hub import HfApi

from ..cards import render_card
from ..hub import ensure_build_branch
from ..config import (
    HUB_REVISION,
    AUDIT_DIR,
    CLEAN_UP_LOCAL,
    CARD_TEMPLATES_DIR,
    COMPACT_DIR,
    GROUP_SINK_ROWS,
    HF_REPO_PRIVATE,
    KEEP_GROUP_FILES,
    KEEP_GROUP_FILES_MIN_FREE_GB,
    OUTPUT_DIR,
    PARTITION_COLS,
    RELEASE,
    STAGING_DIR,
    UNSPLIT_COL,
    Table,
)
from ..partitioning import fused, prepare_for_partition, rowgroups
from ..partitioning.core import ARROW_ROW_GROUP_ROWS, custom_file_path
from ..state import Step, update_state
from .groups import STAGES, Group, record_stage
from .turn import wait_turn

# Rows repeated across the chunks of a group are dropped when merging these tables
DEDUPLICATE = {Table.CLAIMS_LABELS}
# The tables split by key, partitioned once for the whole group at its close, from its
# chunks' processed files, rather than per chunk into a file per key each (~11-12M files
# at a release, about 1 ms of sink a file) that were then glued here: the group's file of
# a key is written by the one sink, with the same rows in the same order
# (docs/journal/2026-10-25-group-partition.md)
GROUP_PARTITIONED = {t for t in Table if PARTITION_COLS[t] != UNSPLIT_COL}
# The unsplit tables (claims, entities): each chunk's one partition file is merged by
# copying its encoded row groups (pq_glue.concat), not decoded and re-written by Polars'
# sink, which spent 4 cores for a minute on a 66-chunk group's claims
# (docs/journal/2026-10-23-merge-glue.md)
GLUE = set(Table) - GROUP_PARTITIONED


def _sidecars(table: Table, group: Group, audit_dir: Path) -> pl.DataFrame:
    """Audit rows (path, num_rows, partition key) of the group's partition files."""
    key = PARTITION_COLS[table]
    return pl.concat(
        [
            pl.read_parquet(audit_dir / table / f"chunk_{c}.parquet").select(
                "path", "num_rows", pl.col(key).alias("key")
            )
            for c in group.chunks
        ]
    )


def staged_path(table: Table, key: str, group: Group, staging_dir: Path) -> Path:
    return staging_dir / table / key / f"{group.name}.parquet"


def _missing(paths: list[str]) -> list[str]:
    return sorted(p for p in paths if not os.path.exists(p))


def _glue_table(table: Table, group: Group, audit_dir: Path, staging_dir: Path) -> None:
    """Glue an unsplit table's partition files into its staged file by pq_glue.concat_many
    (see merge_group), check its rows against the audit sidecars, move it into place, and
    delete the partition files."""
    keys = _sidecars(table, group, audit_dir).group_by("key", maintain_order=True).agg("path", pl.col("num_rows").sum())
    table_dir = staging_dir / table
    table_dir.mkdir(parents=True, exist_ok=True)
    # Listed once, so only the languages new to the staging dir cost a mkdir
    made = {e.name for e in os.scandir(table_dir)}
    jobs = []  # (key, partition files, rows the sidecars say, staged file)
    for key, paths, expected in keys.iter_rows():
        dst = staged_path(table, key, group, staging_dir)
        # Only the first partition file is looked for: a merged language has none left,
        # and a missing one besides fails its glue (checked below)
        if not os.path.exists(paths[0]):
            missing = _missing(paths)
            if len(missing) == len(paths) and dst.exists():
                continue
            raise RuntimeError(f"[push] {table}/{key}: partition files missing: {missing}")
        if key not in made:
            dst.parent.mkdir()
        jobs.append((key, paths, expected, dst))
    try:
        # The rows each copied, counted from the row groups of the footer it wrote;
        # raises for the first that failed, once all have run
        rows = pq_glue.concat_many([(paths, dst.with_suffix(".tmp")) for _, paths, _, dst in jobs])
    except OSError:
        for key, paths, _, _ in jobs:
            if missing := _missing(paths):
                raise RuntimeError(f"[push] {table}/{key}: partition files missing: {missing}") from None
        raise
    for (key, paths, expected, dst), n in zip(jobs, rows):
        tmp = dst.with_suffix(".tmp")
        if n != expected:
            tmp.unlink()
            raise RuntimeError(f"[push] {table}/{key}: merged {n} rows, audit sidecars say {expected}")
        tmp.replace(dst)
        if CLEAN_UP_LOCAL:
            # The key's dir stays (a chunk being partitioned may be writing into it),
            # removed once the run ends (main._remove_empty_dirs)
            for p in paths:
                os.unlink(p)
    print(f"[push] {group.name}: merged {table}", flush=True)


def _marker(table: Table, group: Group, staging_dir: Path) -> Path:
    """Written once the table's group files are staged and checked, before its processed
    files are deleted: a table with it is not partitioned again."""
    return staging_dir / ".merged" / group.name / table


def _work_dir(table: Table, group: Group, staging_dir: Path) -> Path:
    return staging_dir / ".tmp" / group.name / table


# The threads copying the built files' row groups into a group's key files (_copy_built)
GROUP_COPY_THREADS = min(os.cpu_count() or 1, 8)
# A group file's row groups are its key's consecutive row groups of the built files merged
# while one has at most this many rows (the split's writer's own, and Polars' sink's)
GROUP_ROW_GROUP_ROWS = ARROW_ROW_GROUP_ROWS


def _chunk_batches(rows: list[int], limit: int) -> list[list[int]]:
    """The group's chunks (by their positions in it) in consecutive batches of up to
    `limit` rows (by their sidecars), each at least one chunk."""
    batches: list[list[int]] = []
    n = 0
    for i, r in enumerate(rows):
        if not batches or (n + r > limit and batches[-1]):
            batches.append([])
            n = 0
        batches[-1].append(i)
        n += r
    return batches


def _rows(table: Table, f: Path) -> pl.LazyFrame | None:
    """A chunk's rows of a table split by key: its processed file's, flattened
    (prepare_for_partition), or the file the split built, in those rows already
    (partitioning.fused; None if it has no rows)."""
    if (keys := fused.built_index(f.parent, f.name)) is not None:
        return pl.scan_parquet(f) if keys else None
    return prepare_for_partition(f, table)


def _sink_batch(table: Table, files: list[Path], dst: Path, name: str, unique: bool) -> None:
    """Partition the processed files' rows, in order, into one file per key at
    dst/{key}/{name}, by one Polars sink (`unique`: each distinct row's first only)."""
    frames = [lf for f in files if (lf := _rows(table, f)) is not None]
    if not frames:
        return
    lf = pl.concat(frames)
    if unique:  # every column, the key's among them: each key's distinct rows
        lf = lf.unique(maintain_order=True)
    by = PARTITION_COLS[table]
    lf.sink_parquet(
        pl.PartitionBy(
            dst,
            key=by,
            include_key=True,
            file_path_provider=partial(custom_file_path, source=Path(name)),
            approximate_bytes_per_file=None,
        ),
        mkdir=True,
    )


def _partition_table(
    table: Table, group: Group, audit_dir: Path, staging_dir: Path, results_dir: Path
) -> None:
    """Partition the group's processed files of a table split by key into its group
    files, one a key (see merge_group), checked against the audit sidecars, then delete
    the processed files. Resumable: a table already staged (its marker) is not
    partitioned again, and a partial one is redone from its processed files, which are
    kept until its marker is written."""
    processed = [results_dir / table / f"chunk_{c}.parquet" for c in group.chunks]
    marker = _marker(table, group, staging_dir)
    if not marker.exists():
        # (a table the split built with no rows has its index, and no file)
        if missing := [str(p) for p in processed if not p.exists() and fused.built_index(p.parent, p.name) is None]:
            raise RuntimeError(f"[push] {table}: processed files missing: {missing}")
        sidecars = [pl.read_parquet(audit_dir / table / f"chunk_{c}.parquet") for c in group.chunks]
        key = PARTITION_COLS[table]
        # By key dir name, as the sink names them (custom_file_path: str of the key)
        expected = {
            str(k): n
            for k, n in pl.concat([s.select(pl.col(key).alias("key"), "num_rows") for s in sidecars])
            .group_by("key")
            .agg(pl.col("num_rows").sum())
            .iter_rows()
        }
        work = _work_dir(table, group, staging_dir)
        shutil.rmtree(work, ignore_errors=True)
        name = f"{group.name}.parquet"
        indexes = [fused.built_index(p.parent, p.name) for p in processed]
        if table not in DEDUPLICATE and all(i is not None for i in indexes):
            merged = _copy_built(processed, indexes, work, name)
        else:
            merged = _sink_group(table, processed, sidecars, work, name)
        if set(merged) != set(expected):
            raise RuntimeError(f"[push] {table}: keys partitioned {sorted(set(merged) ^ set(expected))} "
                               "differ from the audit sidecars'")
        dedup = table in DEDUPLICATE
        for k, f in merged.items():
            n = pq.read_metadata(f).num_rows
            if not (0 < n <= expected[k] if dedup else n == expected[k]):
                raise RuntimeError(f"[push] {table}/{k}: partitioned {n} rows, audit sidecars say {expected[k]}")
        table_dir = staging_dir / table
        table_dir.mkdir(parents=True, exist_ok=True)
        # Listed once, so only the keys new to the staging dir cost a mkdir
        made = {e.name for e in os.scandir(table_dir)}
        for k, f in merged.items():
            if k not in made:
                (table_dir / k).mkdir()
            os.replace(f, staged_path(table, k, group, staging_dir))
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
        shutil.rmtree(work, ignore_errors=True)
    if CLEAN_UP_LOCAL:
        for p in processed:
            p.unlink(missing_ok=True)
            fused.index_path(p.parent, p.name).unlink(missing_ok=True)
    print(f"[push] {group.name}: partitioned {table}", flush=True)


def _copy_built(processed: list[Path], indexes: list[list], work: Path, name: str) -> dict[str, Path]:
    """The group's file of each key, at work/{key}/{name}, from the files the split built
    (every chunk's, partitioning.fused), their key's row groups copied as bytes in chunk
    order (partitioning.rowgroups): the rows the sink would write, in its order.

    A key's consecutive row groups are merged into one of up to GROUP_ROW_GROUP_ROWS rows
    (rowgroups.merge_row_groups: their pages copied, under one dictionary page), not kept
    a chunk each: at a release a group has hundreds of chunks, so a key's group file would
    otherwise have hundreds of row groups of a few rows, each a footer entry (a quarter of
    the four tables' group bytes) and a read of its own in finalise
    (docs/journal/2026-10-25-group-row-groups.md). A built file whose footer has fields
    the split's writer does not write is copied a row group at a time
    (rowgroups.copy_row_groups), as is every file of its table; so is a key whose row
    groups do not merge (_write_key)."""
    try:
        footers = [rowgroups.ChunkFooter(p) if keys else None for p, keys in zip(processed, indexes)]
        write, merging = partial(rowgroups.merge_row_groups, max_rows=GROUP_ROW_GROUP_ROWS), True
    except rowgroups.Unsupported:
        footers = [rowgroups.Footer(p) if keys else None for p, keys in zip(processed, indexes)]
        write, merging = rowgroups.copy_row_groups, False
    parts: dict[str, list] = {}
    for footer, keys in zip(footers, indexes):
        for key, _, _, _, _, first, n in keys:
            parts.setdefault(key, []).append((footer, first, n))
    merged = {k: work / k / name for k in parts}
    for k, f in merged.items():
        f.parent.mkdir(parents=True)
    # Each built file opened once for all its keys' reads (pread), within the open-file
    # limit; else each read opens it
    opened = [f for f in footers if f is not None] if merging else []
    if len(opened) > resource.getrlimit(resource.RLIMIT_NOFILE)[0] // 4:
        opened = []
    try:
        for f in opened:
            f.fd = os.open(f.path, os.O_RDONLY)
        # (the reads and writes release the GIL; the footers are small)
        with ThreadPoolExecutor(GROUP_COPY_THREADS) as pool:
            list(pool.map(partial(_write_key, write, merging, merged, parts, {}, Lock()), parts))
    finally:
        for f in opened:
            if f.fd is not None:
                os.close(f.fd)
                f.fd = None
    return merged


def _write_key(
    write, merging: bool, merged: dict[str, Path], parts: dict[str, list], footers: dict, lock: Lock, key: str
) -> None:
    """A key's group file by `write`; if merging its row groups raises (a dictionary page
    or column not its first row group's), its row groups copied one by one instead, from
    the files' footers read as trees (`footers`, by path, each read once)."""
    try:
        write(merged[key], parts[key])
    except ValueError:
        if not merging:
            raise
        copied = []
        for f, first, n in parts[key]:
            with lock:
                if f.path not in footers:
                    footers[f.path] = rowgroups.Footer(f.path)
            copied.append((footers[f.path], first, n))
        rowgroups.copy_row_groups(merged[key], copied)


def _sink_group(table: Table, processed: list[Path], sidecars: list, work: Path, name: str) -> dict[str, Path]:
    """The group's file of each key, at its path under `work`, by Polars' sink of the
    processed files' rows in batches of up to GROUP_SINK_ROWS, a key's batch files glued
    (a deduplicated table's deduplicated)."""
    batches = _chunk_batches([int(s["num_rows"].sum()) for s in sidecars], GROUP_SINK_ROWS)
    dedup = table in DEDUPLICATE
    for b, batch in enumerate(batches):
        # A deduplicated table of more batches is deduplicated per key once glued
        _sink_batch(table, [processed[i] for i in batch], work / f"b{b}", name, dedup and len(batches) == 1)
    parts: dict[str, list[Path]] = {}
    for b in range(len(batches)):
        if (work / f"b{b}").is_dir():
            for e in sorted(os.scandir(work / f"b{b}"), key=lambda e: e.name):
                parts.setdefault(e.name, []).append(Path(e.path) / name)
    merged = {k: p[0] for k, p in parts.items() if len(p) == 1}
    several = {k: p for k, p in parts.items() if len(p) > 1}
    for k, paths in several.items():
        merged[k] = work / "glued" / k / name
        merged[k].parent.mkdir(parents=True, exist_ok=True)
    if dedup:
        for k, paths in several.items():
            pl.scan_parquet(paths).unique(maintain_order=True).sink_parquet(merged[k])
    elif several:
        pq_glue.concat_many([(list(map(str, p)), str(merged[k])) for k, p in several.items()])
    return merged


def merge_group(group: Group, audit_dir: Path, staging_dir: Path, results_dir: Path = OUTPUT_DIR) -> None:
    """Make the group's files, one per table per key, in `staging_dir`.

    - A table split by key (GROUP_PARTITIONED) is partitioned once for the group from its
      chunks' processed files in `results_dir` (_partition_table), its tables one at a
      time. Where the split built every chunk's file of the table (partitioning.fused),
      in its rows already, a row group's run a key, each key's runs are copied as bytes
      into its group file, merged into row groups of up to GROUP_ROW_GROUP_ROWS rows
      (_copy_built). Otherwise it is by Polars' sink on all cores, a
      group's chunks taken in batches of up to GROUP_SINK_ROWS rows, to bound the sink's
      memory, and a key's batch files glued.
    - An unsplit table (GLUE: claims, entities) has a partition file a chunk, glued by
      pq_glue.concat_many in a thread of its own while the others are partitioned. A key
      is skipped if its staged file exists and its partition files are gone (merged
      before an interruption).

    Row counts are checked against the audit sidecars before any input is deleted.
    """
    glued = [table for table in Table if table in GLUE]
    with ThreadPoolExecutor(max(len(glued), 1)) as pool:
        futures = [pool.submit(_glue_table, t, group, audit_dir, staging_dir) for t in glued]
        for table in Table:
            if table in GROUP_PARTITIONED:
                _partition_table(table, group, audit_dir, staging_dir, results_dir)
        for f in futures:  # the first failure raises here
            f.result()


def _staged_files(table: Table, group: Group, staging_dir: Path) -> list[Path]:
    return sorted((staging_dir / table).glob(f"*/{group.name}.parquet"))


def _ensure_dataset_card(repo_id: str, table: Table, api: HfApi) -> None:
    """Push the table's rendered card as the repo's README.md, if it has none yet (see
    cards.py; `finalise` pushes it again once the table's metadata is written)."""
    if not (CARD_TEMPLATES_DIR / f"{table}.md").exists():
        return
    if api.file_exists(repo_id, "README.md", repo_type="dataset", revision=HUB_REVISION):
        return
    api.upload_file(
        path_or_fileobj=render_card(table, metadata={}, stats={}).encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        revision=HUB_REVISION,
    )
    print(f"[push] {repo_id}: added dataset card", flush=True)


def push_group(
    group: Group, target_repos: dict[Table, str], staging_dir: Path, api: HfApi
) -> None:
    """Upload each table's staged files (resumable: upload_large_folder skips what is done)."""
    for table in Table:
        if not _staged_files(table, group, staging_dir):
            continue
        repo_id = target_repos[table]
        api.create_repo(
            repo_id, repo_type="dataset", private=HF_REPO_PRIVATE, exist_ok=True
        )
        ensure_build_branch(repo_id, api)
        _ensure_dataset_card(repo_id, table, api)
        api.upload_folder(
            repo_id=repo_id,
            folder_path=staging_dir / table,
            repo_type="dataset",
            allow_patterns=f"*/{group.name}.parquet",
            revision=HUB_REVISION,
        )
        print(f"[push] {group.name}: uploaded {table} to {repo_id}", flush=True)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def _git_blob_sha1(path: Path) -> str:
    h = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    h.update(path.read_bytes())
    return h.hexdigest()


def verify_group(
    group: Group, target_repos: dict[Table, str], staging_dir: Path, api: HfApi
) -> dict[Table, dict[str, list]]:
    """Check every staged file is on the Hub with the same size and hash. Returns each
    table's files checked, by path in its repo: [size, sha256]."""
    checked: dict[Table, dict[str, list]] = {}
    for table in Table:
        staged = _staged_files(table, group, staging_dir)
        remote_paths = [p.relative_to(staging_dir / table).as_posix() for p in staged]
        remote = {}
        for i in range(0, len(remote_paths), 100):
            infos = api.get_paths_info(
                target_repos[table], remote_paths[i : i + 100], repo_type="dataset",
                revision=HUB_REVISION,
            )
            remote.update({info.path: info for info in infos})
        checked[table] = {}
        for local, path in zip(staged, remote_paths):
            info = remote.get(path)
            if info is None:
                raise RuntimeError(f"[push] {table}/{path} is not on the Hub")
            sha256 = _sha256(local)
            if info.lfs is not None:
                same = info.lfs.sha256 == sha256
            else:
                same = info.blob_id == _git_blob_sha1(local)
            if info.size != local.stat().st_size or not same:
                raise RuntimeError(f"[push] {table}/{path} differs on the Hub")
            checked[table][path] = [info.size, sha256]
        print(f"[push] {group.name}: verified {len(staged)} {table} files", flush=True)
    return checked


VERIFIED = "verified.jsonl"


def record_verified(state_dir: Path, group: Group, checked: dict[Table, dict[str, list]]) -> None:
    """Append each table's files verified on the Hub for the group (a line a table, every
    table's, with no files if it has none): what finalise may read locally instead of the
    Hub's copy, as the bytes the Hub has by its own sha256."""
    with (state_dir / VERIFIED).open("a") as f:
        for table, files in checked.items():
            f.write(json.dumps({"first": group.first, "last": group.last, "table": str(table), "files": files}) + "\n")


def read_verified(state_dir: Path) -> dict[tuple[int, int, str], dict[str, list]]:
    """The files verified for each (group's first chunk, last chunk, table): the latest."""
    path = state_dir / VERIFIED
    if not path.exists():
        return {}
    lines = (json.loads(line) for line in path.read_text().splitlines() if line)
    return {(r["first"], r["last"], r["table"]): r["files"] for r in lines}


def keeps(table: Table) -> bool:
    """A table whose verified group files may be kept for finalise: not a release's
    claims_labels, which finalise builds into the compaction source itself
    (claims_labels.py) in place of the pipeline's."""
    return not (RELEASE and table == Table.CLAIMS_LABELS)


def keep_group_files(staging_dir: Path) -> bool:
    """Whether a verified group's files are kept (KEEP_GROUP_FILES): never (the default),
    always, or, set to auto, while the disk has KEEP_GROUP_FILES_MIN_FREE_GB free with them."""
    if KEEP_GROUP_FILES is not None:
        return KEEP_GROUP_FILES
    return shutil.disk_usage(staging_dir).free >= KEEP_GROUP_FILES_MIN_FREE_GB * 1024**3


def _keep(table: Table, group: Group, staging_dir: Path, kept_dir: Path) -> int:
    """Move the table's staged files of the group to kept_dir/{table}/{key}/, where
    compaction would download them (a rename on the same file system)."""
    moved = 0
    for p in _staged_files(table, group, staging_dir):
        dst = kept_dir / table / p.relative_to(staging_dir / table)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(p, dst)
        moved += 1
    return moved


def _set_step(group: Group, step: Step, state_dir: Path) -> None:
    for c in group.chunks:
        update_state(Path(f"chunk_{c}.parquet"), step, state_dir)


def close_group(
    group: Group,
    *,
    target_repos: dict[Table, str],
    state_dir: Path,
    stage: str | None = None,
    audit_dir: Path = AUDIT_DIR,
    staging_dir: Path = STAGING_DIR,
    kept_dir: Path = COMPACT_DIR / "src",
    api: HfApi | None = None,
) -> None:
    """Run the group's remaining stages after `stage` (the last one completed, if any).

    Once verified, the group's files are recorded with their sha256 (VERIFIED) and, where
    kept (keep_group_files), moved to `kept_dir`, the compaction source, so finalise
    reads them there instead of downloading them again; otherwise deleted."""
    api = api or HfApi()
    done = STAGES.index(stage) if stage else -1
    if done < STAGES.index("closed"):
        record_stage(state_dir, group, "closed")
    if done < STAGES.index("merged"):
        merge_group(group, audit_dir, staging_dir)
        record_stage(state_dir, group, "merged")
    if done < STAGES.index("pushed"):
        wait_turn(group.name)  # (the scholarly set's, behind the main set's: turn.py)
        push_group(group, target_repos, staging_dir, api)
        _set_step(group, Step.PUSH, state_dir)
        record_stage(state_dir, group, "pushed")
    if done < STAGES.index("verified"):
        record_verified(state_dir, group, verify_group(group, target_repos, staging_dir, api))
        _set_step(group, Step.POST_CHECK, state_dir)
        record_stage(state_dir, group, "verified")
    if CLEAN_UP_LOCAL:
        keep = keep_group_files(staging_dir)
        # Staging holds only this group (and upload_large_folder's .cache/ progress)
        for table in Table:
            if keep and keeps(table) and (n := _keep(table, group, staging_dir, kept_dir)):
                print(f"[push] {group.name}: kept {n} {table} files for finalise", flush=True)
            shutil.rmtree(staging_dir / table, ignore_errors=True)
        for d in (".merged", ".tmp"):
            shutil.rmtree(staging_dir / d / group.name, ignore_errors=True)
    _set_step(group, Step.COMPLETE, state_dir)
    record_stage(state_dir, group, "done")
    print(f"[push] {group.name}: complete", flush=True)
