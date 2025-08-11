# src/wikidata/pull/core.py
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

Parameterisation:
- Pass a mapping of target repos, one per table (labels, descriptions, aliases, links, claims).
  If provided, we will skip download and set state to POST_CHECK(5) for files that are
  already visible remotely in *every* target (≥1 language subset exists for that filename).
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from ..config import REMOTE_REPO_PATH, Table
from ..state import Step, get_all_state, update_state
from .download import download_files
from .remote_check import _check_all_targets
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

    # Validate state consistency using DataFrame join
    files_with_sizes = chunk_state.join(expected, on="file", how="left")

    missing_count = files_with_sizes.to_series().null_count()
    if missing_count:
        raise RuntimeError(
            f"[pull] {missing_count} files in state have no expected size in source repo. "
            "Has the source listing changed?"
        )

    # Step 1: Remote presence check
    print(
        f"[pull] Chunk {chunk_idx}: checking remote presence for {len(files_with_sizes)} files..."
    )

    try:
        remote_check_series = _check_all_targets(
            files_with_sizes, target_repos, chunk_idx
        )
        files_with_remote = files_with_sizes.with_columns(remote_check_series)
    except Exception as e:
        print(
            f"[pull] Remote check failed, proceeding without skip optimization: {e!r}"
        )
        files_with_remote = files_with_sizes.with_columns(
            pl.lit(False).alias("already_pushed")
        )

    # Step 2: Local file verification
    print(f"[pull] Chunk {chunk_idx}: verifying local files...")

    local_verification = _verify_local_files(
        local_data_dir,
        files_with_remote.get_column("file").to_list(),
        files_with_remote.get_column("size").to_list(),
    )

    files_with_checks = files_with_remote.with_columns(
        pl.Series("local_verified", local_verification).alias("local_verified")
    )

    # Step 3: Categorize files using Polars conditions
    to_mark_postcheck = files_with_checks.filter(pl.col("already_pushed"))
    already_ok_local = files_with_checks.filter(
        (~pl.col("already_pushed")) & pl.col("local_verified")
    )
    need_download = files_with_checks.filter(
        (~pl.col("already_pushed")) & (~pl.col("local_verified"))
    )

    # Step 4: Batch state updates
    # Mark files that are already pushed remotely as POST_CHECK
    if len(to_mark_postcheck) > 0:
        for fname in to_mark_postcheck.get_column("file").to_list():
            update_state(
                Path(fname.replace(".parquet", ".jsonl")), Step.POST_CHECK, state_dir
            )
        print(
            f"[pull] Chunk {chunk_idx}: {len(to_mark_postcheck)} files already pushed remotely "
            "→ advanced to POST_CHECK."
        )

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
            f"({len(already_ok_local)} present locally; {len(to_mark_postcheck)} remote-complete)."
        )
        return

    # Step 5: Batch state update for files about to download
    need_download_files = need_download.get_column("file").to_list()
    for fname in need_download_files:
        update_state(Path(fname.replace(".parquet", ".jsonl")), Step.PULL, state_dir)

    # Step 6: Batch download
    allow_patterns = [f"{REMOTE_REPO_PATH}/{fname}" for fname in need_download_files]
    print(
        f"[pull] Chunk {chunk_idx}: downloading {len(need_download)} files "
        f"(batched) from {repo_id}…"
    )

    hf_download_dir = _hf_dl_subdir(local_data_dir, repo_id=repo_id)
    download_files(
        repo_id=repo_id,
        local_data_dir=hf_download_dir,
        allow_patterns=allow_patterns,
        chunk_idx=chunk_idx,
    )

    # Step 7: Post-download verification and file relocation
    print(f"[pull] Chunk {chunk_idx}: verifying downloaded files and relocating...")

    failures: list[str] = []

    # Use Polars operations where possible

    for fname, expected_bytes in zip(
        need_download.select(["file", "size"]).iter_rows()
    ):
        # Files are now at their natural HuggingFace location
        actual_path = hf_download_dir / REMOTE_REPO_PATH / fname

        if not actual_path.exists() or actual_path.stat().st_size != expected_bytes:
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
