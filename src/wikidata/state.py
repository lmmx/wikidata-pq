"""Simple state management for the wikidata processing pipeline."""

from enum import IntEnum
from pathlib import Path

import polars as pl

from .config import CHUNK_RE, PART_RE


class Step(IntEnum):
    INIT = 0
    PULL = 1
    PROCESS = 2
    PARTITION = 3
    PUSH = 4
    POST_CHECK = 5
    COMPLETE = 6


state_schema = {"file": pl.String, **dict.fromkeys(["chunk", "part", "step"], pl.Int64)}

state_cols = [
    pl.col("path").str.split("/").list.last().alias("file"),
    pl.col("path").str.extract(CHUNK_RE, 1).cast(pl.Int64).alias("chunk"),
    pl.col("path").str.extract(PART_RE, 1).cast(pl.Int64).alias("part"),
]


def update_state(source: Path, step: Step, state_dir: Path) -> None:
    """Update state for a file."""
    assert step in Step, f"{step=} is not a valid Step: {[*enumerate(Step)]}"
    state_file = (state_dir / source.stem).with_suffix(".jsonl")
    pl.LazyFrame({"step": [step]}).sink_ndjson(state_file, mkdir=True)
    return


def get_all_state(state_dir: Path, pattern: str = "*") -> pl.DataFrame:
    """Load all current state."""
    files = f"{pattern}.jsonl"
    if not any(state_dir.glob(files)):
        state = pl.DataFrame(schema=state_schema)
    else:
        state = (
            pl.read_ndjson(state_dir / files, include_file_paths="path")
            .with_columns(state_cols)
            .sort(by=["chunk", "part"])
            .select(*state_schema)
        )
    return state


def init_files(files: list[Path], state_dir: Path) -> None:
    """Initialize state for all files as INIT."""
    for file_path in files:
        update_state(file_path, Step.INIT, state_dir)


def get_next_chunk(state_dir: Path) -> int | None:
    """Get the lowest chunk index that has files that are not COMPLETE."""
    state = get_all_state(state_dir)
    incomplete_chunks = state.filter(pl.col("step") < Step.COMPLETE).get_column("chunk")
    return None if incomplete_chunks.is_empty() else incomplete_chunks.min()
