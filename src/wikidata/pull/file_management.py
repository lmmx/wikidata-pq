# src/wikidata/pull/file_management.py
from __future__ import annotations

from pathlib import Path

from ..config import REMOTE_REPO_PATH


def _move_into_root(local_data_dir: Path, rel_path_under_data: Path) -> Path:
    """Ensure pulled file lives at LOCAL_DATA_DIR/<filename> (not under REMOTE_REPO_PATH/).

    Cleans up empty directory structure left behind after moving files.
    """
    assert (
        rel_path_under_data.parent.name == REMOTE_REPO_PATH
    ), f"Expected '{REMOTE_REPO_PATH}/' prefix, got: {rel_path_under_data}"
    # Source (downloaded) and desired destination paths:
    src = local_data_dir / rel_path_under_data
    dst = local_data_dir / rel_path_under_data.name
    if not src.exists():
        return dst  # Let caller check existence/size and raise if needed
    if dst.exists():
        # If destination exists, prefer keeping dst if it is complete;
        # otherwise replace with the freshly downloaded file.
        src_size = src.stat().st_size
        dst_size = dst.stat().st_size
        if src_size != dst_size:
            dst.unlink()  # replace
            src.replace(dst)
        else:
            # Same size; drop the duplicate under REMOTE_REPO_PATH/
            src.unlink()
    else:
        # Ensure parent exists (it does: local_data_dir), then move
        src.replace(dst)

    # Clean up empty directory structure left behind
    src_parent = src.parent  # This is local_data_dir/REMOTE_REPO_PATH
    try:
        # Only remove if directory is empty
        src_parent.rmdir()
    except OSError:
        # Directory not empty or doesn't exist - that's fine
        pass

    return dst
