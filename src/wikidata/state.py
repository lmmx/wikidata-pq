"""Simple state management for the wikidata processing pipeline."""

from enum import IntEnum
from pathlib import Path

import polars as pl


class Step(IntEnum):
    INIT = 0
    PULL = 1
    PROCESS = 2
    PARTITION = 3
    PUSH = 4
    POST_CHECK = 5
    COMPLETE = 6


def update_state(source: Path, step: Step, state_dir: Path) -> None:
    """Update state for a file."""
    assert step in Step, f"{step=} is not a valid Step: {[*enumerate(Step)]}"
    state_file = (state_dir / source.stem).with_suffix(".jsonl")
    pl.LazyFrame({"step": [step]}).sink_ndjson(state_file, mkdir=True)
    return


def get_all_state(state_dir: Path, pattern: str = "*") -> pl.DataFrame:
    """Load all current state."""
    if not any(state_dir.glob(f"{pattern}.jsonl")):
        return pl.DataFrame(
            {"step": [], "file": [], "chunk": []},
            schema={"step": pl.Int64, "file": pl.String, "chunk": pl.Int64},
        )

    files = pl.read_ndjson(state_dir / f"{pattern}.jsonl", include_file_paths="file")
    return files.with_columns(
        pl.col("file").str.split("/").list.last(),
        pl.col("file").str.extract(r"chunk_(\d+)-", 1).cast(pl.Int64).alias("chunk"),
        pl.col("file").str.extract(r"chunk_\d+-(\d+)", 1).cast(pl.Int64).alias("part"),
    ).sort(by=["chunk", "part"])


def init_files(files: list[Path], state_dir: Path) -> None:
    """Initialize state for all files as INIT."""
    for file_path in files:
        update_state(file_path, Step.INIT, state_dir)


def get_next_chunk(state_dir: Path) -> int | None:
    """Get the chunk index with the lowest number that has INIT files."""
    state = get_all_state(state_dir)
    init_chunks = state.filter(pl.col("step") == Step.INIT).get_column("chunk")
    return None if init_chunks.is_empty() else init_chunks.min()
