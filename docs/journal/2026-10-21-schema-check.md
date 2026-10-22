# 2026-10-21: Schema drift checked by polars-genson alone

Continues docs/journal/2026-10-20-genson-target-schema.md.

## Current State

- `normalise_in_schema` (process.py) wraps `normalise_from_parquet` with a stored
  `schema`, whose `schema_check="compatible"` checks the chunk's inferred schema against
  it before anything is written; a schema that does not fit halts the run with
  `SystemExit` naming the table, the chunk and each offending path (`labels[].value: a
  field the target schema lacks (String)`). Any other `ValueError` propagates unchanged.
- `normalise_claims_direct` and `normalise_map_direct` call `normalise_in_schema`; the
  DeepDiff comparison of `schema_to_dict` outputs after each write (`is_acceptable_diff`,
  `_narrower`) is removed, and deepdiff with it (pyproject.toml; uv.lock drops deepdiff,
  cachebox and orderly-set).
- A chunk whose schema does not fit now writes nothing; before, the claims lookup table
  was written before the check ran.

## Equivalence

- `2026-10-21-schema-check/verdicts.py`, run at feat/genson-target-schema before the
  removal: 873 schemas derived from chunk 0 of release 20261012 (its inferred claims
  schema and `claims_schema` with a field dropped, a field added, a scalar retyped, or a
  field made Null, at every path through structs and lists; the same for the labels,
  descriptions and aliases expected schemas), each judged by `is_acceptable_diff` and by
  `check_schema_compatibility`: 873 agree (483 accepted by both, 390 rejected by both).
- `process()` on chunk 0 writes claims, labels, descriptions and aliases byte for byte as
  `normalise_from_parquet` with the same options does.

## Missing

- `schema_to_dict`'s `{"list": ...}` rendering made a field named `list` or `map`
  indistinguishable from a list or a map; the verdicts above contain no such field, so
  the one known difference between the two checks is not exercised.
