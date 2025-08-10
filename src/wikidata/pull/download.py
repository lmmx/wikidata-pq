from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download


def download_files(
    repo_id: str, local_data_dir: Path, allow_patterns: list[str], chunk_idx: int
) -> None:
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(local_data_dir),
            local_dir_use_symlinks=False,  # we want real files on disk
            allow_patterns=allow_patterns,
        )
    except Exception as e:
        # Keep state at PULL (reflects 'in progress'); caller can re-run safely.
        raise RuntimeError(
            f"[pull] Download failed for chunk {chunk_idx}: {e!r}"
        ) from e
