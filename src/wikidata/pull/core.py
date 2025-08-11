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

Parameterisation:
- Pass a mapping of target repos, one per table (labels, descriptions, aliases, links, claims).
  If provided, we will skip download and set state to POST_CHECK(5) for files that are
  already visible remotely in *every* target (≥1 language subset exists for that filename).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import polars as pl

from ..config import Table
from ..initial import hf_fs
from ..state import Step, get_all_state, update_state
from .download import download_files
from .file_management import _move_into_root
from .remote_check import _already_pushed_in_all_targets, _check_all_targets
from .size_verification import _expected_sizes, _verify_local_files

unpulled = pl.col("step") <= Step.PULL  # INIT or interrupted PULL


def _files_to_pull(state_dir: Path, chunk_idx: int) -> pl.DataFrame:
    """Return state rows for this chunk where step is INIT or PULL (resume-safe).

    Replace the .jsonl extension of the state files for the .parquet of the data files.
    """
    current_chunk = pl.col("chunk") == chunk_idx
    as_pq = pl.col("file").str.replace(r"\.jsonl", ".parquet")
    return get_all_state(state_dir).filter(current_chunk, unpulled).with_columns(as_pq)


def pull_chunk(
    chunk_idx: int,
    state_dir: Path,
    local_data_dir: Path,
    repo_id: str,
    target_repos: dict[Table, str],
) -> None:
    """Pull all needed files for a chunk, with remote-skip and size verification.

    - Set per-file state to PULL(1) *before* downloading each file.
    - Skip files already present in *all* target repos
      and set them to POST_CHECK(5) for the audit stage.
    - Only download files not already present locally with exact source size.
    """
    chunk_state = _files_to_pull(state_dir, chunk_idx)
    if chunk_state.is_empty():
        print(f"[pull] Chunk {chunk_idx}: no files in INIT/PULL.")
        return

    # Get the expected file sizes for this chunk from the remote tree
    expected = _expected_sizes(repo_id, chunk_idx=chunk_idx)

    # Validate state consistency
    files_with_sizes = chunk_state.join(expected, on="file")
    if (missing_expected := rows.height - files_with_sizes.height) > 0:
        raise RuntimeError(
            f"[pull] {missing_expected} files in state have no expected size in source repo. "
            "Has the source listing changed?"
        )

    # Step 1: Vectorized remote presence check
    print(
        f"[pull] Chunk {chunk_idx}: checking remote presence for {len(files_with_sizes)} files..."
    )

    try:
        remote_check_result = _check_all_targets(
            files_with_sizes.get_column("file").to_list(), target_repos
        )
        files_with_remote = files_with_sizes.with_columns(
            pl.Series("already_pushed", remote_check_result).alias("already_pushed")
        )
    except Exception as e:
        print(
            f"[pull] Remote check failed, proceeding without skip optimization: {e!r}"
        )
        files_with_remote = files_with_sizes.with_columns(
            pl.lit(False).alias("already_pushed")
        )

    # Step 2: Vectorized local file verification
    print(f"[pull] Chunk {chunk_idx}: verifying local files...")

    local_verification = _verify_local_files(
        local_data_dir,
        files_with_remote.get_column("file").to_list(),
        files_with_remote.get_column("size").to_list(),
    )

    files_with_checks = files_with_remote.with_columns(
        pl.Series("local_verified", local_verification).alias("local_verified")
    )

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
