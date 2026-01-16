# 2025-01-16: Remediate Existing Files

## Blocked by

- Task 4 (wire state updates) — uses same state machinery and step definitions

## Current State

- State files exist at `state/chunk_0-*.jsonl` containing `{"step":1}` (PULL)
- Processed outputs exist at `results/{labels,descriptions,aliases,links,claims}/chunk_0-*.parquet`
- `get_all_state(state_dir)` returns DataFrame with `file`, `chunk`, `part`, `step` columns (`state.py:31-44`)
- `update_state(source, step, state_dir)` overwrites state file with new step (`state.py:24-28`)
- `OUTPUT_DIR = Path("results")` (`config.py:22`)
- `Table` enum lists all 5 table types (`config.py:27-33`)

## Missing

- No function to check output existence: `all((OUTPUT_DIR / tbl / filename).exists() for tbl in Table)`
- No function to check partition existence: `any((OUTPUT_DIR / tbl).glob(f"language=*/{filename}") for tbl in Table)`
- No function to infer step from disk state
- No remediation entrypoint or `--remediate` CLI flag
- Files at PULL with existing outputs block chunk-completion detection

## Implementation Elements

- Create `src/wikidata/remediate.py` module
- `infer_step_from_disk(filename: str, output_dir: Path, audit_dir: Path) -> Step` function:
  - Check `all((output_dir / tbl / filename).exists() for tbl in Table)` → at least PROCESS
  - Check `any((output_dir / tbl).glob(f"language=*/{filename}"))` → at least PARTITION
  - Check sidecar exists `(audit_dir / tbl / filename).exists()` → at least PARTITION
  - Remote check would require hf:// scan → defer to manual or separate tool
- `remediate_state(state_dir: Path, output_dir: Path, audit_dir: Path)` function:
  - `state = get_all_state(state_dir)`
  - For files where `step < inferred_step`: `update_state(Path(file), inferred_step, state_dir)`
- Add `--remediate` flag to CLI or standalone script `scripts/remediate.py`
- Run once after task 4 to fix chunk_0 before processing chunk_1