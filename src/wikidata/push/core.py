"""Close a group of partitioned chunks: merge, upload, verify, clean up.

Each stage is recorded in the group ledger once complete, so an interrupted group resumes
at the stage it had not finished. Every stage is safe to repeat.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl
import pq_glue
from huggingface_hub import HfApi

from ..cards import render_card
from ..hub import ensure_build_branch
from ..config import (
    HUB_REVISION,
    AUDIT_DIR,
    CLEAN_UP_LOCAL,
    CARD_TEMPLATES_DIR,
    HF_REPO_PRIVATE,
    PARTITION_COLS,
    STAGING_DIR,
    Table,
)
from ..state import Step, update_state
from .groups import STAGES, Group, record_stage

# Rows repeated across the chunks of a group are dropped when merging these tables
DEDUPLICATE = {Table.CLAIMS_LABELS}
# Merged by copying the partition files' encoded row groups (pq_glue.concat), not
# decoded and re-written by Polars' sink, which spent 4 cores for a minute on a 66-chunk
# group's claims (docs/journal/2026-10-23-merge-glue.md). A deduplicated table
# must be decoded to find its repeated rows, so it stays with Polars
GLUE = set(Table) - DEDUPLICATE


def _sidecars(table: Table, group: Group, audit_dir: Path) -> pl.DataFrame:
    """Audit rows (path, num_rows, partition key, row groups) of the group's partition
    files: `row_groups` the key's row groups in `path` where a chunk's table is one file
    (partitioning.core.write_one_file), null where `path` is the key's own file."""
    key = PARTITION_COLS[table]
    frames = []
    for c in group.chunks:
        d = pl.read_parquet(audit_dir / table / f"chunk_{c}.parquet")
        if "row_groups" not in d.columns:
            d = d.with_columns(pl.lit(None, pl.List(pl.UInt32)).alias("row_groups"))
        frames.append(d.select("path", "num_rows", pl.col(key).alias("key"), "row_groups"))
    return pl.concat(frames)


def staged_path(table: Table, key: str, group: Group, staging_dir: Path) -> Path:
    return staging_dir / table / key / f"{group.name}.parquet"


def _merge_key(table: Table, key: str, files: pl.DataFrame, group: Group, staging_dir: Path) -> None:
    """Merge one language's partition files into its staged file with Polars (see
    merge_group)."""
    paths = files["path"].to_list()
    dst = staged_path(table, key, group, staging_dir)
    present = [p for p in paths if Path(p).exists()]
    if not present and dst.exists():
        return
    if len(present) != len(paths):
        missing = sorted(set(paths) - set(present))
        raise RuntimeError(f"[push] {table}/{key}: partition files missing: {missing}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp")
    lf = pl.scan_parquet(paths)
    if table in DEDUPLICATE:
        lf = lf.unique(maintain_order=True)
    lf.sink_parquet(tmp)
    n = pl.scan_parquet(tmp).select(pl.len()).collect().item()
    expected = files["num_rows"].sum()
    ok = 0 < n <= expected if table in DEDUPLICATE else n == expected
    if not ok:
        tmp.unlink()
        raise RuntimeError(f"[push] {table}/{key}: merged {n} rows, audit sidecars say {expected}")
    tmp.replace(dst)
    if CLEAN_UP_LOCAL:
        # The language dir stays (a chunk being partitioned may be writing into it),
        # removed once the run ends (main._remove_empty_dirs)
        for p in paths:
            Path(p).unlink()


def _missing(paths: list[str]) -> list[str]:
    return sorted(p for p in paths if not os.path.exists(p))


def _paths(sources: list) -> list[str]:
    return [s if isinstance(s, str) else s[0] for s in sources]


def _glue_table(table: Table, group: Group, audit_dir: Path, staging_dir: Path) -> None:
    """Glue the table's languages into their staged files in one pq_glue.concat_many call
    (see merge_group), then check, move and delete as _merge_key does.

    A key's sources are its partition files, or, where a chunk's table is one file, the
    key's row groups in it (`row_groups`): each chunk's file is then a source of every
    key's job, opened and its footer read once for the call. The partition files are
    deleted once every key is staged and checked, so a key whose partition files are
    not all there was merged before an interruption if its staged file exists (a file
    shared by many keys is deleted only after all of them were staged)."""
    keys = (
        _sidecars(table, group, audit_dir)
        .group_by("key", maintain_order=True)
        .agg("path", "row_groups", pl.col("num_rows").sum())
    )
    table_dir = staging_dir / table
    table_dir.mkdir(parents=True, exist_ok=True)
    # Listed once, so only the languages new to the staging dir cost a mkdir
    made = {e.name for e in os.scandir(table_dir)}
    present: dict[str, bool] = {}  # each partition file, looked for once

    def exists(p: str) -> bool:
        if p not in present:
            present[p] = os.path.exists(p)
        return present[p]

    jobs = []  # (key, sources, rows the sidecars say, staged file)
    paths = set()  # every partition file of the table, to delete once merged
    for key, files, row_groups, expected in keys.iter_rows():
        dst = staged_path(table, key, group, staging_dir)
        if not all(exists(p) for p in dict.fromkeys(files)):
            if dst.exists():
                # Merged: its partition files left are deleted with the others'
                paths.update(p for p in files if exists(p))
                continue
            missing = sorted({p for p in files if not exists(p)})
            raise RuntimeError(f"[push] {table}/{key}: partition files missing: {missing}")
        if key not in made:
            dst.parent.mkdir()
        sources = [p if rgs is None else (p, rgs) for p, rgs in zip(files, row_groups)]
        jobs.append((key, sources, expected, dst))
        paths.update(files)
    try:
        # The rows each copied, counted from the row groups of the footer it wrote;
        # raises for the first that failed, once all have run
        rows = pq_glue.concat_many([(sources, dst.with_suffix(".tmp")) for _, sources, _, dst in jobs])
    except OSError:
        if missing := _missing(sorted({p for _, sources, _, _ in jobs for p in _paths(sources)})):
            raise RuntimeError(f"[push] {table}: partition files missing: {missing}") from None
        raise
    for (key, _, expected, dst), n in zip(jobs, rows):
        tmp = dst.with_suffix(".tmp")
        if n != expected:
            tmp.unlink()
            raise RuntimeError(f"[push] {table}/{key}: merged {n} rows, audit sidecars say {expected}")
        tmp.replace(dst)
    if CLEAN_UP_LOCAL:
        # As in _merge_key, once every key is staged
        for p in sorted(paths):
            os.unlink(p)
    print(f"[push] {group.name}: merged {table}", flush=True)


def merge_group(group: Group, audit_dir: Path, staging_dir: Path) -> None:
    """Merge each table's partition files into one file per language for the group.

    A language is skipped if its staged file exists and its partition files are gone,
    or some of them (merged before an interruption: they are deleted once every language
    of the table is staged). Merged row counts are checked against the audit
    sidecars before the partition files are deleted. A glued table's languages are
    glued in one pq_glue.concat_many call (in parallel, the GIL released: each is a few
    small files, so a call per language spent the close's time in Python and on the
    GIL, not on bytes copied), the glued tables in threads of their own; a deduplicated
    table's one at a time, each by Polars on all cores.
    """
    glued = [table for table in Table if table in GLUE]
    # A thread per table: one table's languages are gathered, and their staging dirs made
    # (a dir's entries are made one at a time), while another's are glued
    with ThreadPoolExecutor(max(len(glued), 1)) as pool:
        # list() so the first failure raises here
        list(pool.map(lambda t: _glue_table(t, group, audit_dir, staging_dir), glued))
    for table in Table:
        if table not in GLUE:
            for (key,), files in _sidecars(table, group, audit_dir).group_by("key", maintain_order=True):
                _merge_key(table, key, files, group, staging_dir)
            print(f"[push] {group.name}: merged {table}", flush=True)


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
    _set_step(group, Step.COMPLETE, state_dir)
    record_stage(state_dir, group, "done")
    print(f"[push] {group.name}: complete", flush=True)
