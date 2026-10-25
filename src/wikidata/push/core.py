"""Close a group of partitioned chunks: merge, upload, verify, clean up.

Each stage is recorded in the group ledger once complete, so an interrupted group resumes
at the stage it had not finished. Every stage is safe to repeat.
"""

from __future__ import annotations

import hashlib
import json
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
    COMPACT_DIR,
    HF_REPO_PRIVATE,
    KEEP_GROUP_FILES,
    KEEP_GROUP_FILES_MIN_FREE_GB,
    PARTITION_COLS,
    RELEASE,
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


def _glue_table(table: Table, group: Group, audit_dir: Path, staging_dir: Path) -> None:
    """Glue the table's languages into their staged files in one pq_glue.concat_many call
    (see merge_group), then check, move and delete as _merge_key does."""
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
            # As in _merge_key
            for p in paths:
                os.unlink(p)
    print(f"[push] {group.name}: merged {table}", flush=True)


def merge_group(group: Group, audit_dir: Path, staging_dir: Path) -> None:
    """Merge each table's partition files into one file per language for the group.

    A language is skipped if its staged file exists and its partition files are gone
    (merged before an interruption). Merged row counts are checked against the audit
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
    _set_step(group, Step.COMPLETE, state_dir)
    record_stage(state_dir, group, "done")
    print(f"[push] {group.name}: complete", flush=True)
