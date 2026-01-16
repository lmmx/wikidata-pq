# 2025-01-16: Implement Push

## Blocked by

- Task 1 (integrate partition) — upload requires `results/{table}/language={lang}/` structure

## Current State

- `HF_USER = "permutans"` (`config.py:35`)
- `REPO_TARGET = "{hf_user}/wikidata-{tbl}"` formats to `permutans/wikidata-labels`, `permutans/wikidata-descriptions`, etc. (`config.py:37`)
- `main.py:49-50` builds `target_repos = {tbl: REPO_TARGET.format(hf_user=hf_user, tbl=tbl) for tbl in Table}`
- `Table` enum values are "labels", "descriptions", "aliases", "links", "claims" (`config.py:27-33`)
- DESIGN.md specifies `upload-large-folder` chosen to avoid API rate limits vs individual file uploads (`DESIGN.md:59`)
- DESIGN.md specifies push triggers when entire chunk completed, all files at PARTITION (`DESIGN.md:54-55`)
- `get_all_state(state_dir)` returns DataFrame with columns `file`, `chunk`, `part`, `step` (`state.py:31-44`)
- `Step.PARTITION = 3` (`state.py:12`)

## Missing

- No `push` module or `push_chunk()` function
- No HF CLI subprocess call for `huggingface-cli upload-large-folder {repo_id} {local_folder} --repo-type dataset`
- No chunk-completion predicate checking `state.filter(pl.col("chunk") == idx).select(pl.col("step").min() >= Step.PARTITION)`
- `main.py` push step is comment placeholder at line 65
- Target repos may not exist on HuggingFace — no `huggingface_hub.create_repo()` call

## Implementation Elements

- Create `src/wikidata/push.py` module
- `push_chunk(chunk_idx, state_dir, output_dir, target_repos)` function signature
- Chunk-complete check: `get_all_state(state_dir).filter(pl.col("chunk") == chunk_idx).get_column("step").min() >= Step.PARTITION`
- For each `tbl, repo_id in target_repos.items()`:
  - `local_folder = output_dir / tbl`
  - `subprocess.run(["huggingface-cli", "upload-large-folder", repo_id, str(local_folder), "--repo-type", "dataset"], check=True)`
- Alternative: `huggingface_hub.upload_large_folder(repo_id=repo_id, folder_path=local_folder, repo_type="dataset")`
- Create repos if missing: `huggingface_hub.create_repo(repo_id, repo_type="dataset", exist_ok=True)`