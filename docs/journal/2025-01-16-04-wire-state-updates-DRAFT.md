# 2025-01-16: Wire State Updates

## Blocked by

- Task 3 (implement post-check) — wire all steps once implementations exist

## Current State

- `Step` IntEnum: INIT=0, PULL=1, PROCESS=2, PARTITION=3, PUSH=4, POST_CHECK=5, COMPLETE=6 (`state.py:8-15`)
- `update_state(source: Path, step: Step, state_dir: Path)` writes `{"step": int}` to `state_dir/{source.stem}.jsonl` (`state.py:24-28`)
- `pull_chunk()` signature includes `state_dir: Path` (`pull/core.py:57`)
- `pull_chunk()` calls `update_state(Path(fname.replace(".parquet", ".jsonl")), Step.PULL, state_dir)` after download (`pull/core.py:91-92`)
- `process()` signature: `process(data_dir: Path, output_dir: Path, repo_id: str, chunk_idx: int | None = None)` (`process.py:174`)
- `main.py:58` calls `process(data_dir=data_dir, output_dir=output_dir, repo_id=repo_id)` without `state_dir`

## Missing

- `process()` signature lacks `state_dir: Path` parameter
- `main.py` does not pass `state_dir` to `process()`
- `process()` for-loop processes files but has no `update_state(pq_path, Step.PROCESS, state_dir)` call after writing 5 tables (`process.py:197-249`)
- Partition step (task 1) needs `update_state(pq_path, Step.PARTITION, state_dir)` after all 5 tables partitioned
- Push step (task 2) needs `update_state(pq_path, Step.PUSH, state_dir)` for each file in chunk after upload
- Post-check step (task 3) needs `update_state(pq_path, Step.POST_CHECK, state_dir)` after verification, then `update_state(pq_path, Step.COMPLETE, state_dir)`

## Implementation Elements

- Change `process()` signature to `process(data_dir: Path, output_dir: Path, repo_id: str, state_dir: Path, chunk_idx: int | None = None)` (`process.py:174`)
- Change `main.py:58` to `process(data_dir=data_dir, output_dir=output_dir, repo_id=repo_id, state_dir=state_dir)`
- Add at end of `process()` file loop (`process.py:249`): `update_state(pq_path, Step.PROCESS, state_dir)`
- Add `from .state import Step, update_state` import to `process.py`
- In partition step: after `partition_parquet()` calls complete for all 5 tables, `update_state(pq_path, Step.PARTITION, state_dir)`
- In push step: after `upload_large_folder()` succeeds, for each file in chunk `update_state(pq_path, Step.PUSH, state_dir)`
- In post-check step: after verification passes, `update_state(pq_path, Step.POST_CHECK, state_dir)` then `update_state(pq_path, Step.COMPLETE, state_dir)`