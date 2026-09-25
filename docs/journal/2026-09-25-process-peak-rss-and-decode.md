# 2026-09-25: process() peak RSS and claims decode (#11)

## Current State

- `process()` reads all columns of each chunk with `pl.read_parquet(pq_path)` into `df` (`process.py:285`) — only `normalise_sitelinks(df)` reads `df` (`process.py:322`), and `df` stays referenced through the claims step
- `labels`, `descs`, `aliases`, `links` and `claims` stay referenced after their `sink_parquet` calls until the next loop iteration rebinds them (`process.py:293-348`)
- `normalise_claims_direct` writes a temp parquet of normalised JSON strings via `normalise_from_parquet`, reads it back with `pl.read_parquet`, and decodes with `str.json_decode(dtype=schema)` while the raw temp frame `result` is still referenced (`process.py:208-226`)
- `normalise_map_direct` follows the same temp-file and `str.json_decode` pattern for labels, descriptions and aliases (`process.py:50-95`)
- A 4-core session with polars-genson 0.7.10 on `chunk_5280.parquet` (10k rows, 243 MB `claims`) timed `str.json_decode` at 28.8s of about 50s for `normalise_claims_direct`, against 6.4s for inference (polars-genson `docs/journal/2026-09-25-perf-work-plan.md`)
- `pyproject.toml` requires `polars-genson>=0.7.6` (staged change from `>=0.7.4`)

## Missing

- No timing of `normalise_claims_direct` stages on the i9
- No column selection in the `process()` read of `pq_path`
- No `del` of per-table frames after `sink_parquet`
- No typed-Arrow output from polars-genson, so no `read_parquet`-only replacement for the `str.json_decode` step (polars-genson #194)
