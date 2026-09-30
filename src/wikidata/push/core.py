"""Close a group of partitioned chunks: merge, upload, verify, clean up.

Each stage is recorded in the group ledger once complete, so an interrupted group resumes
at the stage it had not finished. Every stage is safe to repeat.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import polars as pl
from huggingface_hub import HfApi

from ..cards import render_card
from ..config import (
    AUDIT_DIR,
    CLEAN_UP_LOCAL,
    DATASET_CARDS_DIR,
    HF_REPO_PRIVATE,
    PARTITION_COLS,
    STAGING_DIR,
    Table,
)
from ..state import Step, update_state
from .groups import STAGES, Group, record_stage

# Rows repeated across the chunks of a group are dropped when merging these tables
DEDUPLICATE = {Table.CLAIMS_LABELS}


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


def merge_group(group: Group, audit_dir: Path, staging_dir: Path) -> None:
    """Merge each table's partition files into one file per language for the group.

    A language is skipped if its staged file exists and its partition files are gone
    (merged before an interruption). Merged row counts are checked against the audit
    sidecars before the partition files are deleted.
    """
    for table in Table:
        sidecars = _sidecars(table, group, audit_dir)
        for (key,), files in sidecars.group_by("key", maintain_order=True):
            paths = files["path"].to_list()
            dst = staged_path(table, key, group, staging_dir)
            present = [p for p in paths if Path(p).exists()]
            if not present and dst.exists():
                continue
            if len(present) != len(paths):
                missing = sorted(set(paths) - set(present))
                raise RuntimeError(f"[push] {table}/{key}: partition files missing: {missing}")
            lf = pl.scan_parquet(paths)
            if table in DEDUPLICATE:
                lf = lf.unique(maintain_order=True)
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_suffix(".tmp")
            lf.sink_parquet(tmp)
            n = pl.scan_parquet(tmp).select(pl.len()).collect().item()
            expected = files["num_rows"].sum()
            ok = 0 < n <= expected if table in DEDUPLICATE else n == expected
            if not ok:
                tmp.unlink()
                raise RuntimeError(
                    f"[push] {table}/{key}: merged {n} rows, audit sidecars say {expected}"
                )
            tmp.replace(dst)
            if CLEAN_UP_LOCAL:
                for p in paths:
                    Path(p).unlink()
                try:
                    Path(paths[0]).parent.rmdir()  # the language dir, once empty
                except OSError:
                    pass
        print(f"[push] {group.name}: merged {table}", flush=True)


def _staged_files(table: Table, group: Group, staging_dir: Path) -> list[Path]:
    return sorted((staging_dir / table).glob(f"*/{group.name}.parquet"))


def _ensure_dataset_card(repo_id: str, table: Table, api: HfApi) -> None:
    """Push the table's rendered card as the repo's README.md, if it has none yet (see
    cards.py; `finalise` pushes it again once the table's metadata is written)."""
    if not (DATASET_CARDS_DIR / f"{table}.md").exists():
        return
    if api.file_exists(repo_id, "README.md", repo_type="dataset"):
        return
    api.upload_file(
        path_or_fileobj=render_card(table).encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
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
        _ensure_dataset_card(repo_id, table, api)
        api.upload_folder(
            repo_id=repo_id,
            folder_path=staging_dir / table,
            repo_type="dataset",
            allow_patterns=f"*/{group.name}.parquet",
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
                target_repos[table], remote_paths[i : i + 100], repo_type="dataset"
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
