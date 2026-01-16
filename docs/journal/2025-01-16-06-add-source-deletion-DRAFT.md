# 2025-01-16: Add Source Deletion

## Blocked by

- Task 3 (post-check verified working on at least one full chunk cycle)
- Task 4 (state updates verified working)
- Task 5 (remediation complete, existing files at correct state)

## Current State

- DESIGN.md: "Source files deleted locally after successful verification for clean handoff" (`DESIGN.md:70`)
- Source files at `data/huggingface_hub/philippesaade/wikidata/data/chunk_{idx}-*.parquet`
- `_hf_dl_subdir(parent_dir, repo_id)` returns `parent_dir / "huggingface_hub" / repo_id` (`pull/core.py:26-27`)
- `REMOTE_REPO_PATH = "data"` (`config.py:19`)
- `PREFETCH_BUDGET_GB = 300.0` (`config.py:34`)
- `_local_cache_gb(root_data_dir, repo_id)` sums `st_size` of all `chunk_*.parquet` in cache dir (`pull/prefetch.py:40-47`)
- Prefetch worker skips if `budget_gb - have_gb <= 0` (`pull/prefetch.py:84-86`)
- Current cache size ~208GB across chunks 0-12

## Missing

- No deletion code in post-check step
- No `delete_chunk_sources(chunk_idx, root_data_dir, repo_id)` function
- Prefetch budget exhausted, prefetching stalled until deletion frees space
- No logging of bytes freed after deletion

## Implementation Elements

- Add to `postcheck.py` after state advances to COMPLETE:
```python
  def delete_chunk_sources(chunk_idx: int, root_data_dir: Path, repo_id: str) -> int:
      ds_dir = _hf_dl_subdir(root_data_dir, repo_id=repo_id)
      pattern = f"{REMOTE_REPO_PATH}/chunk_{chunk_idx}-*.parquet"
      freed = 0
      for path in ds_dir.glob(pattern):
          freed += path.stat().st_size
          path.unlink()
      return freed
```
- Call `freed = delete_chunk_sources(chunk_idx, root_data_dir, repo_id)` after all files in chunk reach COMPLETE
- Log: `print(f"[postcheck] Deleted chunk {chunk_idx} sources ({freed / 1024**3:.2f} GB freed)")`
- Only delete after ALL files in chunk verified — partial deletion risks inconsistent state
- Guard: check `get_all_state(state_dir).filter(pl.col("chunk") == chunk_idx).get_column("step").min() == Step.COMPLETE` before deletion