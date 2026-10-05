"""Simple state management for the wikidata processing pipeline."""

import re
from enum import IntEnum
from pathlib import Path

import polars as pl

from .config import CHUNK_RE


class Step(IntEnum):
    INIT = 0
    PULL = 1
    PROCESS = 2
    PARTITION = 3
    PUSH = 4
    POST_CHECK = 5
    COMPLETE = 6


state_schema = {"file": pl.String, **dict.fromkeys(["chunk", "step"], pl.Int64)}

state_cols = [
    pl.col("path").str.split("/").list.last().alias("file"),
    pl.col("path").str.extract(CHUNK_RE, 1).cast(pl.Int64).alias("chunk"),
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
        return pl.DataFrame(schema=state_schema)
    return _read_state(state_dir / files)


def get_chunk_state(state_dir: Path, chunk_idx: int) -> pl.DataFrame:
    """The state of one chunk's files (chunk_{N}.jsonl, or the source repo's
    chunk_{N}-*.jsonl), without reading every chunk's (0.4 s over 10k files)."""
    paths = [
        *state_dir.glob(f"chunk_{chunk_idx}.jsonl"),
        *state_dir.glob(f"chunk_{chunk_idx}-*.jsonl"),
    ]
    if not paths:
        return pl.DataFrame(schema=state_schema)
    return _read_state(sorted(paths))


def last_chunk(state_dir: Path) -> int:
    """The run's last chunk index, from the state file names (none read)."""
    names = (p.name for p in state_dir.glob("chunk_*.jsonl"))
    return max(int(m.group(1)) for n in names if (m := re.match(r"chunk_(\d+)", n)))


def _read_state(source: Path | list[Path]) -> pl.DataFrame:
    return (
        pl.read_ndjson(source, include_file_paths="path")
        .with_columns(state_cols)
        .sort(by="chunk")
        .select(*state_schema)
    )


def init_files(files: list[Path], state_dir: Path) -> None:
    """Initialize state for all files as INIT."""
    for file_path in files:
        update_state(file_path, Step.INIT, state_dir)


def get_next_chunk(state_dir: Path, below: Step = Step.COMPLETE) -> int | None:
    """Get the lowest chunk index that has files before step `below`."""
    state = get_all_state(state_dir)
    incomplete_chunks = state.filter(pl.col("step") < below).get_column("chunk")
    return None if incomplete_chunks.is_empty() else incomplete_chunks.min()


def validate_chunk_outputs(
    chunk_idx: int, state_dir: Path, output_dir: Path, tables: list[str]
) -> tuple[list[str], dict[str, list[str]]]:
    """Check all tables have all expected files for a chunk.

    Returns:
        Tuple of (expected_filenames, missing_by_table).
        missing_by_table is empty dict if all files present.
    """
    chunk_state = get_chunk_state(state_dir, chunk_idx)
    expected_files = [
        f.replace(".jsonl", ".parquet")
        for f in chunk_state.get_column("file").to_list()
    ]

    missing = {}
    for tbl in tables:
        table_dir = output_dir / tbl
        for filename in expected_files:
            if not (table_dir / filename).exists():
                missing.setdefault(tbl, []).append(filename)

    return expected_files, missing


def get_file_step(filename: str, state_dir: Path) -> Step | None:
    """Get the current step for a specific file, or None if not in state."""
    jsonl_fname = filename.replace(".parquet", ".jsonl")
    if not (state_dir / jsonl_fname).exists():
        return None
    file_state = _read_state(state_dir / jsonl_fname)
    return Step(file_state.get_column("step").item())


def file_at_or_past(filename: str, step: Step, all_state: pl.DataFrame) -> bool:
    """Check if file is at or past a given step, using pre-loaded state."""
    jsonl_fname = filename.replace(".parquet", ".jsonl")
    file_state = all_state.filter(pl.col("file") == jsonl_fname)
    if file_state.is_empty():
        return False
    return file_state.get_column("step").item() >= step
