# 2026-10-20: Chunk tables written by genson in their stored schema

process.py writes the claims, labels, descriptions and aliases tables with
`normalise_from_parquet(typed=True, schema=..., unnest=True)` (polars-genson PR #287),
which checks the inferred schema against the stored one and writes the file in the stored
schema. It replaces reading genson's output back with Polars, conforming it
(`scan_parquet(schema=..., cast_options=...)` for claims, `cast` for the maps), unnesting
it and writing it again with `sink_parquet`. The measurements, the compatibility
contract and the scripts are in polars-genson's
`docs/journal/2026-10-20-target-schema-output.md` and the folder beside it.

## Current State

- `normalise_claims_direct` writes the claims to `claims/<chunk>.tmp` unnested in
  `claims_schema` and returns that path with the inferred schema, read from the file's
  `genson_avro_schema` metadata by `inferred_schema` (process.py); `process()` renames
  the file into place once the DeepDiff check (`is_acceptable_diff`) passes, and deletes
  it when the check fails.
- `normalise_map_direct` writes labels, descriptions and aliases to `<table>/<chunk>.tmp`
  in `_map_schema(key, lists)`, runs the same DeepDiff check on the metadata's inferred
  schema and renames the file into place (process.py); it returns nothing, and
  `process()` checks ids with `pl.scan_parquet` of the written file.
- genson rejects an inferred schema that does not fit the stored one before writing
  (`schema_check="compatible"`: missing fields and `Null` fit, extra fields and other
  dtypes do not), naming each path in a `ValueError`; the DeepDiff check still runs after
  it on every table.
- Every genson call passes `map_encoding="kv"` (the maps tables, claims, sitelinks and
  `lookup_to_long`), and `avro_to_polars_schema` takes `map_encoding="kv"` — polars-genson
  1.x defaults to `"mapping"`, whose maps decode to `pl.Map`, while the stored schemas
  hold maps as lists of key-value structs.
- pyproject.toml requires polars>=2.0.0 and polars-genson>=1.2.0; uv.lock still locks
  polars 1.44.2 and polars-genson 0.9.6 (and was already behind pyproject.toml at
  5ad5f96: `uv sync --locked` refuses it).

## Chunk 0 of release 20261012

Chunk 0 (the dump's first 10,000 entities, Q31 to Q83751, 485 MB of JSON, made by
`dump.write_chunk` and `dump.route_chunk`), `process()` timed by polars-genson's
`scripts/profile_process.py`:

| code | polars-genson, Polars | wall (s) | CPU (s) |
|---|---|---|---|
| 5ad5f96, its lock | 0.9.6, 1.44.2 | 71.1 (one cold run) | 148.3 |
| 5ad5f96, upgraded (`map_encoding="kv"` patched in) | 1.1.0, 2.0.0 | 44.6, 41.6, 41.8 | 59.6, 52.0, 51.9 |
| feat/genson-target-schema | PR #287 build, 2.0.0 | 6.5, 5.4, 5.3 | 18.8, 15.1, 15.0 |

- In the upgraded 5ad5f96 runs, the claims' `sink_parquet` took 31.8 s of 41.7 s warm,
  their conforming `collect` 3.6 s and genson 3.3 s; on the branch the claims take 3.3 s
  in all.
- The profiler's process peaked at 4.15 GB on the upgraded 5ad5f96 and 3.37 GB on the
  branch (three runs each).

Outputs of the branch against those of 5ad5f96 on its lock (polars-genson's
`scripts/compare_outputs.py` and `scripts/diff_claims.py`):

- labels, descriptions, aliases, entities, links and claims_labels are equal in rows,
  Polars schema, Arrow schema and Parquet columns;
- claims are equal in Polars schema, Arrow schema and Parquet columns, and differ only
  where 5ad5f96 on its lock writes a struct whose fields are all null and the branch a
  null struct (`mainsnak.datavalue` of 305 somevalue and novalue snaks, 603,420
  `precision`, 613,258 `latitude` and `longitude`, 31 reference snaks' `datavalue`) —
  every leaf value is equal, and 5ad5f96's own conformance on polars-genson 1.1.0 and
  Polars 2.0.0 gives the branch's claims, so the difference comes with the upgrade;
- the claims, labels, descriptions and aliases files carry genson's metadata
  (`genson_avro_schema`, `genson_normalise_config`) and zstd level 1: claims 54.3 MB
  against 57.5 MB, labels 4.2 against 3.8, descriptions 4.5 against 3.7, aliases 1.8
  against 1.6.

## Missing

- polars-genson with PR #287 is unreleased, so uv.lock cannot lock it.
- The DeepDiff check (`is_acceptable_diff`, the deepdiff dependency) runs beside genson's
  check; `check_schema_compatibility` from polars-genson does not replace it yet.
- `lookup_to_long` with `map_encoding="kv"` has run only on an empty lookup: an official
  dump's claims carry no labels.
- Modules other than process.py (compaction, sort, hub, cards) have not run on Polars 2.
- The run covers chunk 0 only.
