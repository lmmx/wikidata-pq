# src/wikidata/pull.py
"""Reliable, resumable pull step for a single chunk.

Design goals:
- Only download what's needed for this chunk.
- State always reflects *current* activity:
  INIT(0) -> set to PULL(1) when we *start* downloading each file.
- Idempotent on re-runs:
  - If a local file already exists and exactly matches the source size, we keep it.
  - If a file is already present in all target repos (>=1 language subset per table),
    we bump state straight to POST_CHECK(5) so the audit step can verify and finish it.
- Size verification:
  Compare on-disk size to the authoritative bytes from the source repo's tree listing.
- Large-batch efficiency:
  Use a single `snapshot_download` with allow_patterns listing the specific files still needed.

Assumptions:
- Source repo structure puts parquet files under `data/`.
- `state.init_files(...)` used only the filename (no subdir) in state.
- Downstream code (process.py) expects pulled files in `LOCAL_DATA_DIR/*.parquet`
  (i.e., *not* nested under 'data/'), so we relocate after download.

Optional:
- Pass a mapping of target repos, one per table (labels, descriptions, aliases, links, claims).
  If provided, we will skip download and set state to POST_CHECK(5) for files that are
  already visible remotely in *every* target (≥1 language subset exists for that filename).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import polars as pl
from huggingface_hub import HfFileSystem

from ..config import Table
from ..state import Step, update_state
from .download import download_files
from .file_management import _move_into_root
from .remote_check import _already_pushed_in_all_targets
from .selection import _select_chunk_files
from .size_verification import _expected_sizes, _verify_local_file


def pull_chunk(
    chunk_idx: int,
    state_dir: Path,
    local_data_dir: Path,
    repo_id: str,
    target_repos: dict[Table, str] | None = None,
) -> None:
    """Pull all needed files for a chunk, with remote-skip and size verification.

    - Set per-file state to PULL(1) *before* downloading each file.
    - Skip files already present in *all* target repos (if `target_repos` provided)
      and set them to POST_CHECK(5) for the audit stage.
    - Only download files not already present locally with exact source size.
    """
    hf_fs = HfFileSystem()
    local_data_dir.mkdir(parents=True, exist_ok=True)

    rows = _select_chunk_files(state_dir, chunk_idx)
    if rows.is_empty():
        print(f"[pull] Chunk {chunk_idx}: no files in INIT/PULL.")
        return

    # Build expected size index once
    expected = _expected_sizes(repo_id)
    # Derive filenames (state stores only the filename, not 'data/<filename>')
    filenames: list[str] = rows.get_column("file").to_list()

    # First pass: apply remote-skip rule and local-size check
    to_mark_postcheck: list[str] = []
    already_ok_local: list[str] = []
    need_download: list[str] = []

    if target_repos:
        # Remote presence check: if *every* table has ≥1 language subset with this file
        # we consider it already Pushed and hand it off to the post-check later.
        for fname in filenames:
            try:
                if _already_pushed_in_all_targets(fname, target_repos, hf_fs):
                    to_mark_postcheck.append(fname)
            except Exception as e:
                # Be conservative: if check fails for any reason, do not skip.
                print(f"[pull] Remote check failed for {fname}: {e!r}")

    # Files not marked for post-check remain candidates for local check / download
    remaining: Iterable[str] = (
        [f for f in filenames if f not in to_mark_postcheck]
        if to_mark_postcheck
        else filenames
    )

    for fname in remaining:
        exp = expected.get(fname)
        if exp is None:
            # Should not happen if state is in sync with source listing; fail fast.
            raise RuntimeError(
                f"[pull] No expected size found in source repo for {fname!r}. "
                "Has the source listing changed?"
            )
        local_path = local_data_dir / fname
        if _verify_local_file(local_path, exp):
            already_ok_local.append(fname)
            # If this file was stuck at INIT (e.g., resumed run), advance to PULL to reflect readiness.
            row_step = int(rows.filter(pl.col("file") == fname)["step"][0])
            if row_step == int(Step.INIT):
                update_state(Path(fname), Step.PULL, state_dir)
        else:
            need_download.append(fname)

    # Mark POST_CHECK for remote-complete files
    if to_mark_postcheck:
        for fname in to_mark_postcheck:
            update_state(Path(fname), Step.POST_CHECK, state_dir)
        print(
            f"[pull] Chunk {chunk_idx}: {len(to_mark_postcheck)} files already pushed remotely "
            "→ advanced to POST_CHECK."
        )

    if not need_download:
        print(
            f"[pull] Chunk {chunk_idx}: nothing to download "
            f"({len(already_ok_local)} present locally; {len(to_mark_postcheck)} remote-complete)."
        )
        return

    # Before downloading, flip those files to PULL to reflect current activity
    for fname in need_download:
        update_state(Path(fname), Step.PULL, state_dir)

    # Batch download only the needed files; mirror repo structure under local_data_dir
    allow_patterns = [f"data/{fname}" for fname in need_download]
    print(
        f"[pull] Chunk {chunk_idx}: downloading {len(need_download)} files "
        f"(batched) from {repo_id}…"
    )
    download_files(
        repo_id=repo_id,
        local_data_dir=local_data_dir,
        allow_patterns=allow_patterns,
        chunk_idx=chunk_idx,
    )
    # Post-download: verify size, then relocate from 'data/<fname>' to root '<fname>'
    failures: list[str] = []
    for fname in need_download:
        exp = expected[fname]
        rel_under_data = Path("data") / fname
        dst = _move_into_root(local_data_dir, rel_under_data)
        if not _verify_local_file(dst, exp):
            failures.append(fname)

    if failures:
        # Fail loudly; safer to stop than to progress corrupt/incomplete files.
        details = ", ".join(failures[:5])
        more = "" if len(failures) <= 5 else f" (+{len(failures)-5} more)"
        raise RuntimeError(
            f"[pull] Verification failed for {len(failures)} files in chunk {chunk_idx}: {details}{more}"
        )

    print(
        f"[pull] Chunk {chunk_idx}: ✓ downloaded & verified {len(need_download)} files; "
        f"{len(already_ok_local)} were already present; {len(to_mark_postcheck)} already pushed."
    )
