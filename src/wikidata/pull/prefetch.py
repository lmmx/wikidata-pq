# src/wikidata/pull/prefetch.py
from __future__ import annotations

import shutil
from functools import cache
from pathlib import Path

import polars as pl

from ..config import CHUNK_RE, REMOTE_REPO_PATH
from ..state import get_all_state
from .core import _hf_dl_subdir, pull_chunk
from .size_verification import _expected_sizes


def _chunk_is_complete(
    root_data_dir: Path, repo_id: str, chunk_idx: int, state_dir: Path
) -> bool:
    """Check if chunk has all expected files (is complete)."""
    ds_dir = _hf_dl_subdir(root_data_dir, repo_id=repo_id)
    pattern = f"{REMOTE_REPO_PATH}/chunk_{chunk_idx}-*.parquet"

    # Count actual files matching the pattern
    actual_files = len(list(ds_dir.glob(pattern)))

    # Get expected file count for this chunk from state
    expected_files = _get_expected_file_count(state_dir, chunk_idx)

    return actual_files == expected_files


def _get_expected_file_count(state_dir: Path, chunk_idx: int) -> int:
    """Get the expected number of files for a given chunk from state metadata."""
    state = get_all_state(state_dir)
    chunk_files = state.filter(pl.col("chunk") == chunk_idx)
    return chunk_files.height  # Number of files in this chunk


def _chunk_has_any_local_files(
    root_data_dir: Path, repo_id: str, chunk_idx: int
) -> bool:
    ds_dir = _hf_dl_subdir(root_data_dir, repo_id=repo_id)
    pattern = f"{REMOTE_REPO_PATH}/chunk_{chunk_idx}-*.parquet"
    return any(ds_dir.glob(pattern))


def _local_cache_gb(root_data_dir: Path, repo_id: str) -> float:
    ds_dir = _hf_dl_subdir(root_data_dir, repo_id=repo_id)
    total = 0
    for p in ds_dir.glob(f"{REMOTE_REPO_PATH}/chunk_*.parquet"):
        try:
            total += p.stat().st_size
        except OSError:
            pass
    return round(total / 1024**3, 3)


@cache
def _expected_chunk_sizes(repo_id: str) -> pl.LazyFrame:
    # file,size,size_gb → +chunk → per-chunk totals
    sizes = _expected_sizes(repo_id, chunk_idx=None).lazy()
    return (
        sizes.with_columns(
            pl.col("file").str.extract(CHUNK_RE, 1).cast(pl.Int64).alias("chunk")
        )
        .group_by("chunk")
        .agg(
            pl.col("size").sum().alias("size"),
            pl.col("size_gb").sum().alias("size_gb"),
        )
        .sort("chunk")
    )


def _choose_chunks_to_prefetch(
    current_chunk: int,
    repo_id: str,
    root_data_dir: Path,
    state_dir: Path,
    budget_gb: float,
    max_ahead: int,
) -> list[int]:
    """Greedy: walk forward from current+1 while we have budget and no local files exist."""

    # def no_local_files(chunk_idx: int) -> bool:
    #     """Whether the chunk has no local files yet."""
    #     return not _chunk_has_any_local_files(
    #         root_data_dir, repo_id=repo_id, chunk_idx=chunk_idx
    #     )

    def incomplete_files(chunk_idx: int) -> bool:
        """Whether the chunk has an incomplete set of local files."""
        return not _chunk_is_complete(
            root_data_dir, repo_id=repo_id, chunk_idx=chunk_idx, state_dir=state_dir
        )

    return (
        _expected_chunk_sizes(repo_id)
        .filter(pl.col("chunk") > current_chunk)
        .head(max_ahead)
        .filter(pl.col("chunk").map_elements(incomplete_files, return_dtype=pl.Boolean))
        .filter(pl.col("size_gb").cum_sum().alias("cumsum_gb") <= budget_gb)
        .collect()
        .get_column("chunk")
        .to_list()
    )


def prefetch_worker(
    current_chunk: int,
    state_dir: Path,
    root_data_dir: Path,
    repo_id: str,
    target_repos: dict,
    *,
    budget_gb: float,
    max_ahead: int,
    min_free_gb: float,
) -> None:
    """Background task: prefetch upcoming chunks respecting disk budget & guard rails."""
    try:
        # Disk headroom check
        free_gb = shutil.disk_usage(root_data_dir).free / 1024**3
        if free_gb < min_free_gb:
            print(f"[prefetch] Skipping (free {free_gb:.1f} GB < {min_free_gb} GB).")
            return

        # Remaining budget after what we already have on disk
        have_gb = _local_cache_gb(root_data_dir, repo_id)
        remaining_budget = max(0.0, budget_gb - have_gb)
        if remaining_budget <= 0:
            print(f"[prefetch] Skipping (budget exhausted: have {have_gb:.1f} GB).")
            return

        to_fetch = _choose_chunks_to_prefetch(
            current_chunk=current_chunk,
            repo_id=repo_id,
            root_data_dir=root_data_dir,
            state_dir=state_dir,
            budget_gb=remaining_budget,
            max_ahead=max_ahead,
        )
        if not to_fetch:
            print("[prefetch] Nothing to prefetch under budget/guards.")
            return

        print(
            f"[prefetch] Prefetching chunks {to_fetch} (budget rem ≈ {remaining_budget:.1f} GB)…"
        )
        for ch in to_fetch:
            # Double-check right before we start each chunk
            if _chunk_is_complete(
                root_data_dir, repo_id=repo_id, chunk_idx=ch, state_dir=state_dir
            ):
                continue
            pull_chunk(
                chunk_idx=ch,
                state_dir=state_dir,
                root_data_dir=root_data_dir,
                repo_id=repo_id,
                target_repos=target_repos,
            )
    except Exception as e:
        print(f"[prefetch] Aborted after error: {e!r}")
