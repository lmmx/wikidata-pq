from __future__ import annotations


from huggingface_hub import HfFileSystem

from ..config import TABLE_TYPES, Table


def _already_pushed_in_all_targets(
    filename: str, target_repos: dict[Table, str], hf_fs: HfFileSystem
) -> bool:
    """True if every target repo has at least one language subset containing the file.

    We look for any path like: datasets/{repo}/*/{filename}
    """
    for table in TABLE_TYPES:
        repo = target_repos.get(table)
        if not repo:
            return False  # Unknown target for this table -> cannot assert pushed
        pattern = f"datasets/{repo}/*/{filename}"
        found = hf_fs.glob(pattern)
        if not found:
            return False
    return True
