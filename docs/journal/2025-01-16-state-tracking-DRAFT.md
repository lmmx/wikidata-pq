# 2025-01-16: State Tracking

## Current State

- State schema defines steps INIT(0) through COMPLETE(6) (`state.py:8-15`)
- `update_state()` writes step to per-file JSONL (`state.py:24-28`)
- `pull_chunk()` receives `state_dir` and calls `update_state()` to PULL (`pull/core.py:67-142`)
- `get_next_chunk()` returns lowest chunk with incomplete files (`state.py:47-51`)

## Stubbed

- Push step has comment placeholder only (`main.py:64`)
- Post-check step has comment placeholder only (`main.py:67`)

## Missing

- `process()` lacks `state_dir` parameter (`process.py:174`)
- `main.py` does not pass `state_dir` to `process()` (`main.py:58`)
- No `update_state()` call after file processing in `process()` loop (`process.py:197-249`)
- `partition_parquet()` not called from main loop (`partitioning.py:44-53` unused)
- No state update to PARTITION after partitioning
- Push implementation absent
- Post-check implementation absent, including source file deletion

## Divergence

- DESIGN.md specifies state update to PROCESS after processing (`DESIGN.md:49`)
- DESIGN.md specifies source deletion in post-check (`DESIGN.md:70`)
- Files exist in `results/` but state shows PULL not PROCESS