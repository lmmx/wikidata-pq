# src/wikidata/pull/core.py
"""Reliable, resumable pull step for a single chunk.

Design goals:
- Only download what's needed for this chunk.
- State always reflects *current* activity:
  INIT(0) -> set to PULL(1) when we *start* downloading each file.
- Idempotent on re-runs:
  - If a local file already exists and exactly matches the source size, we keep it.
  - Whether a chunk is already uploaded is known from local state (see push), not the Hub.
- Size verification:
  Compare on-disk size to the authoritative bytes from the source repo's tree listing.
- Large-batch efficiency:
  Use a single `snapshot_download` with allow_patterns listing the specific files still needed.

Assumptions:
- Source repo structure puts parquet files under `data/`.
- `state.init_files(...)` used only the filename (no subdir) in state.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from ..config import REMOTE_REPO_PATH
from ..state import Step, get_all_state, update_state
from .download import download_files
from .size_verification import _expected_sizes, _verify_local_files

unpulled = pl.col("step") <= Step.PULL  # INIT or interrupted PULL


def _hf_dl_subdir(parent_dir: Path, repo_id: str) -> Path:
    return parent_dir / "huggingface_hub" / repo_id


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
    root_data_dir: Path,
    repo_id: str,
) -> None:
    """Pull all needed files for a chunk, with size verification.

    - Set per-file state to PULL(1) *before* downloading each file.
    - Only download files not already present locally with exact source size.
    """
    chunk_state = _files_to_pull(state_dir, chunk_idx)
    if chunk_state.is_empty():
        print(f"[pull] Chunk {chunk_idx}: no files in INIT/PULL.")
        return

    # Get the expected file sizes for this chunk from the remote tree
    expected = _expected_sizes(repo_id, chunk_idx=chunk_idx)

    # Validate state consistency using DataFrame join
    files_with_sizes = chunk_state.join(expected, on="file", how="left")

    missing_count = files_with_sizes.to_series().null_count()
    if missing_count:
        raise RuntimeError(
            f"[pull] {missing_count} files in state have no expected size in source repo. "
            "Has the source listing changed?"
        )

    # Step 1: Local file verification
    print(f"[pull] Chunk {chunk_idx}: verifying local files...")

    local_verification = _verify_local_files(
        root_data_dir,
        files_with_sizes.get_column("file").to_list(),
        files_with_sizes.get_column("size").to_list(),
    )

    files_with_checks = files_with_sizes.with_columns(
        pl.Series("local_verified", local_verification).alias("local_verified")
    )

    # Step 2: Categorize files
    already_ok_local = files_with_checks.filter(pl.col("local_verified"))
    need_download = files_with_checks.filter(~pl.col("local_verified"))

    # Step 3: Batch state updates
    # Update INIT files that are locally verified to PULL state
    init_but_local_ok = already_ok_local.filter(pl.col("step") == Step.INIT)
    if len(init_but_local_ok) > 0:
        for fname in init_but_local_ok.get_column("file").to_list():
            update_state(
                Path(fname.replace(".parquet", ".jsonl")), Step.PULL, state_dir
            )

    if len(need_download) == 0:
        print(
            f"[pull] Chunk {chunk_idx}: nothing to download "
            f"({len(already_ok_local)} present locally)."
        )
        return

    # Step 4: Batch state update for files about to download
    need_download_files = need_download.get_column("file").to_list()
    for fname in need_download_files:
        update_state(Path(fname.replace(".parquet", ".jsonl")), Step.PULL, state_dir)

    # Step 5: Batch download
    allow_patterns = [f"{REMOTE_REPO_PATH}/{fname}" for fname in need_download_files]
    print(
        f"[pull] Chunk {chunk_idx}: downloading {len(need_download)} files "
        f"(batched) from {repo_id}…"
    )

    hf_download_dir = _hf_dl_subdir(root_data_dir, repo_id=repo_id)
    download_files(
        repo_id=repo_id,
        root_data_dir=hf_download_dir,
        allow_patterns=allow_patterns,
        chunk_idx=chunk_idx,
    )

    # Step 6: Post-download verification
    print(f"[pull] Chunk {chunk_idx}: verifying downloaded files...")

    failures: list[str] = []

    # Use Polars operations where possible

    for fname, expected_bytes in need_download.select(["file", "size"]).iter_rows():
        path = hf_download_dir / REMOTE_REPO_PATH / fname
        if not path.exists() or path.stat().st_size != expected_bytes:
            failures.append(fname)

    if failures:
        # Fail loudly; safer to stop than to progress corrupt/incomplete files.
        details = ", ".join(failures[:5])
        more = "" if len(failures) <= 5 else f" (+{len(failures) - 5} more)"
        raise RuntimeError(
            f"[pull] Verification failed for {len(failures)} files in chunk {chunk_idx}: {details}{more}"
        )

    print(
        f"[pull] Chunk {chunk_idx}: ✓ downloaded & verified {len(need_download)} files; "
        f"{len(already_ok_local)} were already present."
    )
