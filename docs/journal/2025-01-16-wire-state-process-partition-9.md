# 2025-01-16: Wire State Updates for Process and Partition

## Current State

- `Step` IntEnum defines INIT=0, PULL=1, PROCESS=2, PARTITION=3, PUSH=4, POST_CHECK=5, COMPLETE=6 (`state.py:8-15`)
- `update_state(source: Path, step: Step, state_dir: Path)` extracts `source.stem`, writes `{"step": N}` to `state_dir/{stem}.jsonl` (`state.py:24-28`)
- `pull_chunk()` receives `state_dir` and calls `update_state()` to advance files to PULL (`pull/core.py`)
- `process()` signature is `process(data_dir: Path, output_dir: Path, repo_id: str, chunk_idx: int | None = None)` (`process.py:174`)
- `main.py` partition loop iterates table-first: for each table, for each file (`main.py:89-99`)
- State files show `{"step":1}` (PULL) for all chunk_0 files

## Missing

- `process()` lacks `state_dir: Path` parameter (`process.py:174`)
- `main.py` does not pass `state_dir` to `process()` (`main.py:86`)
- No `update_state()` call in `process()` after writing 5 tables (`process.py:197-249`)
- No `update_state()` call after partitioning completes for a file (`main.py:89-99`)
- Partition loop structure prevents per-file state update — iterates table-first not file-first