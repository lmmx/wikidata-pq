from __future__ import annotations

from pathlib import Path

import polars as pl

from ..state import Step, get_all_state


def _select_chunk_files(state_dir: Path, chunk_idx: int) -> pl.DataFrame:
    """Return state rows for this chunk where step is INIT or PULL (resume-safe)."""
    state = get_all_state(state_dir)
    if state.is_empty():
        return state
    return state.filter(
        (pl.col("chunk") == chunk_idx)
        & (pl.col("step").is_in([int(Step.INIT), int(Step.PULL)]))
    )
