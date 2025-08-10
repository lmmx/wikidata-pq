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

sizes_schema = {"filename": pl.String, "size": pl.Int64, "size_gb": pl.Float64}


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
        .sort("filename")
    )
    return sizes.match_to_schema(sizes_schema)


def _verify_local_file(path: Path, expected_bytes: int) -> bool:
    """True if file exists and size matches exactly."""
    return path.stat().st_size == expected_bytes if path.exists() else False
