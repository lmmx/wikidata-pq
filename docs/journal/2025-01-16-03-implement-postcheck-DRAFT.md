# 2025-01-16: Implement Post-check (Verify Only)

## Blocked by

- Task 2 (implement push) — cannot verify uploads before they exist

## Current State

- DESIGN.md specifies Polars `scan_parquet()` with predicate pushdown via `hf://` paths (`DESIGN.md:66`)
- DESIGN.md specifies min/max ID validation against sidecar records (`DESIGN.md:67`)
- DESIGN.md notes verification failure handling "currently unspecified" (`DESIGN.md:68`)
- `sink_sidecar()` writes parquet with columns after transform: `language`, `rows`, `min_id`, `max_id` (`partitioning.py:36-40`)
- Polars `hf://` protocol: `pl.scan_parquet("hf://datasets/{repo_id}/{path}")` streams remote parquet
- `REPO_TARGET` produces repo IDs like `permutans/wikidata-labels` (`config.py:37`)
- Partitioned remote path structure: `hf://datasets/permutans/wikidata-labels/language=en/chunk_0-00001-of-00546.parquet`

## Missing

- No `postcheck` module or `postcheck_chunk()` function
- No sidecar loading: `pl.read_parquet(AUDIT_DIR / tbl / f"{chunk_file}.parquet")`
- No remote row count: `pl.scan_parquet(f"hf://datasets/{repo_id}/language={lang}/{chunk_file}").select(pl.len()).collect().item()`
- No remote min/max ID: `pl.scan_parquet(...).select(pl.col("id").min(), pl.col("id").max()).collect()`
- No comparison loop: for each row in sidecar, verify `remote_rows == sidecar_rows` and `remote_min == sidecar_min_id` and `remote_max == sidecar_max_id`
- No verification failure handling strategy
- `main.py` post-check step is comment placeholder at line 67
- Source deletion deferred to task 6

## Implementation Elements

- Create `src/wikidata/postcheck.py` module
- `postcheck_chunk(chunk_idx, state_dir, output_dir, audit_dir, target_repos)` function signature
- For each `tbl, repo_id in target_repos.items()`:
  - Load sidecar: `sidecar = pl.read_parquet(audit_dir / tbl / chunk_file)`
  - For each row in sidecar (`lang, expected_rows, expected_min, expected_max`):
    - Remote path: `f"hf://datasets/{repo_id}/language={lang}/{chunk_file}"`
    - Remote stats: `pl.scan_parquet(remote_path).select(pl.len().alias("rows"), pl.col("id").min().alias("min_id"), pl.col("id").max().alias("max_id")).collect()`
    - Compare: `assert remote["rows"] == expected_rows and remote["min_id"] == expected_min and remote["max_id"] == expected_max`
- Return list of failures or raise on first failure (design decision needed)
- Links table uses `site` not `language` partition key — remote path pattern differs: `site={site}/`