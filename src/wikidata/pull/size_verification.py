from __future__ import annotations

import os
import re
from functools import cache
from pathlib import Path

import polars as pl
from huggingface_hub.hf_api import HfApi, RepoFile

from ..config import CHUNK_RE, REMOTE_REPO_PATH

# Use to replace the capture group of a regex
CAPTURE_GROUP_RE = r"\(([^)]*)\)"

remote_file_cols = [
    pl.col("path").str.split("/").list.last().alias("filename"),
    (pl.col("size") / 1024**3).round(3).alias("size_gb"),
]

sizes_schema = {"file": pl.String, "size": pl.Int64, "size_gb": pl.Float64}


@cache
def _list_all_files(repo_id: str) -> list[RepoFile]:
    """Cached API call for all file tree entries with their sizes (LFS or regular)."""
    ls = HfApi().list_repo_tree
    return list(ls(repo_id=repo_id, path_in_repo=REMOTE_REPO_PATH, repo_type="dataset"))


def _expected_sizes(repo_id: str, chunk_idx: int | None = None) -> pl.DataFrame:
    """Return filename, size (in both bytes and GB) for all parquets in source 'data/'.

    If `chunk_idx` is passed, only return the expected sizes for the specific chunk.
    """
    files_info = _list_all_files(repo_id=repo_id)
    chunk_pattern = re.sub(
        CAPTURE_GROUP_RE, "" if chunk_idx is None else str(chunk_idx), CHUNK_RE
    )
    sizes = (
        pl.DataFrame(
            [{"path": f.path, "size": f.size} for f in files_info if hasattr(f, "size")]
        )
        .filter(pl.col("path").str.contains(rf"{chunk_pattern}.*\.parquet$"))
        .with_columns(remote_file_cols)
        .sort("file")
    )
    return sizes.match_to_schema(sizes_schema)


def _verify_local_file_single(path_str: str, expected_bytes: int) -> bool:
    """Verify a file exists and size matches our expectation."""
    if (path := Path(path_str)).exists():
        try:
            actual_size = path.stat().st_size
            result = actual_size == expected_bytes
        except (OSError, IOError):
            # Handle permission errors, etc.
            result = False
    else:
        result = False
    return result


def _verify_local_files(
    local_data_dir: Path, filenames: list[str], expected_sizes: list[int]
) -> list[bool]:
    """Batch version using Polars map_elements for local file verification.

    Returns a list of booleans indicating whether each file exists and matches expected size.
    If the local_data_dir doesn't exist or has no parquet files, returns all False without individual file checks.

    shape: (1, 4)
    ┌─────────────┬───────────────┬──────────────────┬──────────┐
    │ file        ┆ expected_size ┆ full_path        ┆ verified │
    │ ---         ┆ ---           ┆ ---              ┆ ---      │
    │ str         ┆ i64           ┆ str              ┆ bool     │
    ╞═════════════╪═══════════════╪══════════════════╪══════════╡
    │ example.csv ┆ 246           ┆ data/example.csv ┆ true     │
    └─────────────┴───────────────┴──────────────────┴──────────┘
    """
    # Quick checks: directory doesn't exist? No parquet files? Then all files missing
    if not local_data_dir.exists() or not any(local_data_dir.glob("*.parquet")):
        return [False] * len(filenames)

    full_path_col = (
        pl.lit(str(local_data_dir)).add("/").add(pl.col("file")).alias("full_path")
    )
    files_info = pl.DataFrame(
        {"file": filenames, "expected_size": expected_sizes}
    ).with_columns(full_path_col)

    # Use map_elements to verify each file
    verification_result = files_info.select(
        pl.struct(["full_path", "expected_size"])
        .map_elements(
            lambda row: _verify_local_file_single(
                row["full_path"], row["expected_size"]
            ),
            return_dtype=pl.Boolean,
        )
        .alias("verified")
    )

    return verification_result.to_series().to_list()
