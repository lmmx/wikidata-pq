"""Initialize state for the wikidata processing pipeline."""

from pathlib import Path

import polars as pl
from huggingface_hub import HfFileSystem

from .state import init_files

repo_id = "philippesaade/wikidata"
hf_fs = HfFileSystem()


def setup_state(state_dir: Path) -> None:
    """Set up initial state for chunk files."""
    chunk_files = get_all_chunk_files()
    init_files(chunk_files, state_dir)
    return


def get_all_chunk_files() -> list[Path]:
    """Get list of all chunk files to process."""
    ls = hf_fs.glob(f"datasets/{repo_id}/data/*.parquet")
    filenames = pl.Series(ls).str.split("/").list.last()
    return list(map(Path, filenames))
