from __future__ import annotations

from ..config import Table
from ..initial import hf_fs
from .size_verification import CAPTURE_GROUP_RE


def _check_all_targets(
    files: pl.DataFrame,
    target_repos: dict[Table, str],
    hf_fs: HfFileSystem,
    chunk_idx: int,
) -> pl.Series:
    """Check if files are already pushed to all target repos.

    Returns a boolean Series indicating whether each file is already pushed.
    Uses chunk-specific glob patterns to reduce API calls.
    """
    if files.is_empty():
        return pl.Series("already_pushed", dtype=pl.Boolean)

    # Use chunk pattern to filter remote files
    chunk_pattern = re.sub(CAPTURE_GROUP_RE, str(chunk_idx), CHUNK_RE)

    # Create one DataFrame per table with filename and presence columns
    presence_dfs = []

    for table in Table:
        repo = target_repos.get(table)
        if not repo:
            # If any table has no target repo, no files can be considered "pushed"
            return pl.Series(
                "already_pushed", [False] * len(files_df), dtype=pl.Boolean
            )

        try:
            # Get chunk-specific files across all language partitions for this repo
            pattern = f"datasets/{repo}/*/{chunk_pattern}*"
            chunk_files = hf_fs.glob(pattern)

            # Create DataFrame with filenames from this repo
            repo_df = pl.DataFrame(
                {
                    "file": [path.split("/")[-1] for path in chunk_files],
                    str(table): True,  # Table enum coerces to string
                }
            )
            presence_dfs.append(repo_df)

        except Exception:
            # If we can't list files for any repo, be conservative
            return pl.Series(
                "already_pushed", [False] * len(files_df), dtype=pl.Boolean
            )

    # Start with files and join presence from each table
    result_df = files_df.select("file")

    for table_name, presence_df in zip(Table, presence_dfs):
        result_df = result_df.join(presence_df, on="file", how="left").with_columns(
            pl.col(table_name).fill_null(False)  # Missing = not present
        )

    # Logical AND across all table columns to check if present in ALL targets
    table_cols = [str(table) for table in TABLE_TYPES]
    already_pushed = result_df.select(
        pl.all_horizontal(table_cols).alias("already_pushed")
    ).get_column("already_pushed")

    return already_pushed
