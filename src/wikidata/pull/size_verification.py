from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi


def _expected_sizes(repo_id: str) -> dict[str, int]:
    """Return {filename -> byte_size} for all parquet files in source 'data/'."""
    api = HfApi()
    # One API call that returns tree entries with sizes (LFS or regular).
    entries = api.list_repo_tree(
        repo_id=repo_id, path_in_repo="data", repo_type="dataset"
    )
    sizes: dict[str, int] = {}
    for e in entries:
        # entries can include directories or files with/without size attribute
        size = getattr(e, "size", None)
        path = getattr(e, "path", None)
        if size is None or path is None:
            continue
        if path.endswith(".parquet"):
            fname = os.path.basename(path)
            sizes[fname] = int(size)
    return sizes


def _verify_local_file(path: Path, expected_bytes: int) -> bool:
    """True if file exists and size matches exactly."""
    try:
        st = path.stat()
        return st.st_size == expected_bytes
    except FileNotFoundError:
        return False
