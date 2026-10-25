"""Close a group of partitioned chunks: merge, upload, verify, clean up.

Each stage is recorded in the group ledger once complete, so an interrupted group resumes
at the stage it had not finished. Every stage is safe to repeat.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

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
    GROUP_SINK_ROWS,
    HF_REPO_PRIVATE,
    OUTPUT_DIR,
    PARTITION_COLS,
    STAGING_DIR,
    UNSPLIT_COL,
    Table,
)
from ..partitioning import prepare_for_partition
from ..partitioning.core import custom_file_path
from ..state import Step, update_state
from .groups import STAGES, Group, record_stage

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


def _sink_batch(table: Table, files: list[Path], dst: Path, name: str, unique: bool) -> None:
    """Partition the processed files' rows, in order, into one file per key at
    dst/{key}/{name}, by one Polars sink (`unique`: each distinct row's first only)."""
    lf = pl.concat([prepare_for_partition(f, table) for f in files])
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
        if missing := [str(p) for p in processed if not p.exists()]:
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
        if set(parts) != set(expected):
            raise RuntimeError(f"[push] {table}: keys partitioned {sorted(set(parts) ^ set(expected))} "
                               "differ from the audit sidecars'")
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
    print(f"[push] {group.name}: partitioned {table}", flush=True)


def merge_group(group: Group, audit_dir: Path, staging_dir: Path, results_dir: Path = OUTPUT_DIR) -> None:
    """Make the group's files, one per table per key, in `staging_dir`.

    - A table split by key (GROUP_PARTITIONED) is partitioned once for the group from its
      chunks' processed files in `results_dir` (_partition_table), its tables one at a
      time, each by Polars' sink on all cores. A group's chunks are taken in batches of up
      to GROUP_SINK_ROWS rows, to bound the sink's memory, and a key's batch files glued.
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
) -> None:
    """Check every staged file is on the Hub with the same size and hash."""
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
        for local, path in zip(staged, remote_paths):
            info = remote.get(path)
            if info is None:
                raise RuntimeError(f"[push] {table}/{path} is not on the Hub")
            if info.lfs is not None:
                same = info.lfs.sha256 == _sha256(local)
            else:
                same = info.blob_id == _git_blob_sha1(local)
            if info.size != local.stat().st_size or not same:
                raise RuntimeError(f"[push] {table}/{path} differs on the Hub")
        print(f"[push] {group.name}: verified {len(staged)} {table} files", flush=True)


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
    api: HfApi | None = None,
) -> None:
    """Run the group's remaining stages after `stage` (the last one completed, if any)."""
    api = api or HfApi()
    done = STAGES.index(stage) if stage else -1
    if done < STAGES.index("closed"):
        record_stage(state_dir, group, "closed")
    if done < STAGES.index("merged"):
        merge_group(group, audit_dir, staging_dir)
        record_stage(state_dir, group, "merged")
    if done < STAGES.index("pushed"):
        push_group(group, target_repos, staging_dir, api)
        _set_step(group, Step.PUSH, state_dir)
        record_stage(state_dir, group, "pushed")
    if done < STAGES.index("verified"):
        verify_group(group, target_repos, staging_dir, api)
        _set_step(group, Step.POST_CHECK, state_dir)
        record_stage(state_dir, group, "verified")
    if CLEAN_UP_LOCAL:
        # Staging holds only this group (and upload_large_folder's .cache/ progress)
        for table in Table:
            shutil.rmtree(staging_dir / table, ignore_errors=True)
        for d in (".merged", ".tmp"):
            shutil.rmtree(staging_dir / d / group.name, ignore_errors=True)
    _set_step(group, Step.COMPLETE, state_dir)
    record_stage(state_dir, group, "done")
    print(f"[push] {group.name}: complete", flush=True)
