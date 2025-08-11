from __future__ import annotations

import re

import polars as pl

from ..config import CHUNK_RE, Table
from ..initial import hf_fs
from .size_verification import CAPTURE_GROUP_RE

just_filename = pl.col("file").str.split("/").last()

pushed_col = pl.Series("already_pushed", dtype=pl.Boolean)

# Logical AND across all table columns to check if present in ALL targets
in_all = pl.all_horizontal(Table).alias(pushed_col.name)


def _check_all_targets(
    files: pl.DataFrame, target_repos: dict[Table, str], chunk_idx: int
) -> pl.Series:
    """Check if files are already pushed to all target repos.

    Returns a boolean Series indicating whether each file is already pushed.
    Uses chunk-specific glob patterns to reduce API calls.
    """
    if files.is_empty():
        return pushed_col

    # Use chunk pattern to filter remote files
    chunk_pattern = re.sub(CAPTURE_GROUP_RE, str(chunk_idx), CHUNK_RE)

    # Create one DataFrame per table with filename and presence columns
    presence_tbls = []

    for table in Table:
        repo_id = target_repos.get(table)
        if not repo_id:
            # If any table has no target repo, no files can be considered "pushed"
            return pushed_col.extend_constant(False, n=len(files))

        # Get chunk-specific files across all language partitions for this repo
        pattern = f"datasets/{repo_id}/*/{chunk_pattern}*"

        try:
            chunk_files = hf_fs.glob(pattern)
            repo_info = pl.DataFrame({"file": chunk_files, table: True})
            presence_tbls.append(repo_info.with_columns(just_filename))

        except Exception:
            # If we can't list files for any repo, be conservative
            return pushed_col.extend_constant(False, n=len(files))

    # Start with the full file list and join presence from each table
    result = files.select("file")

    for table_name, presence in zip(Table, presence_tbls):
        in_tbl = pl.col(table_name).fill_null(False)  # Missing = not present
        result = result.join(presence, on="file", how="left").with_columns(in_tbl)

    return result.select(in_all).to_series()
