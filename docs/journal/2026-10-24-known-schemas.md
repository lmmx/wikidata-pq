# 2026-10-24: Claims decoded against a stored schema, inferred only where they do not fit

Continues docs/journal/2026-10-24-ci-step-costs.md.

## Current State

- A release's claims are decoded against a stored Avro schema
  (`src/wikidata/schemas/claims.avsc`), passed as polars-genson's
  `normalise_from_parquet(avro_schema=...)` (lmmx/polars-genson#305). A chunk whose rows
  all fit it strictly is decoded with no inference. Any other chunk is inferred and
  checked against the stored Polars schema exactly as before, so a run whose data
  changes shape still halts. Sitelinks, which have no schema check, are normalised
  against `SITELINKS_AVRO` (#56).
- `scripts/claims_avro_schema.py` writes it, inferred with the claims' own options from
  the given chunk files together. The committed one is from the bench's 11 chunk files.
- polars-genson comes from #305's CI wheel (`bench/wheels`) until it is released.

## Why

Inference re-derives a schema that is known. The claims options (forced maps, forced
promotion, unified maps) were chosen to make it chunk-stable, and every table's is
checked against a stored schema anyway. genson's claims call was about 58% of
process CPU on CI (#55), inference about a third of it.

**Strict** (genson's `NormalisePlan::to_arrow_strict`): a row is decoded only where its
own inferred schema adds nothing to the schema. It gives up on:
- a member with no field;
- a value of another type: a non-string at a string, a fraction or a too-large integer
  at an int, a scalar at an array or map, or at a record that no promoted field takes;
- a value where the schema saw only nulls;
- a null at a non-nullable record;
- a union of two or more types, which is not compiled.

On any of these the whole chunk takes the old path. A narrower schema (from the 5
r-chunks only) found the null-at-a-record case: a reference snak without a datavalue
became a record of nulls where the inferred schema, nullable there, gave null.

## Measured

CI bench, quick tier, 3 rounds, each PR against the one below it:

| PR | pipeline wall | process CPU | ETA (pipeline, 4 cores) | parity |
|---|---|---|---|---|
| #56 sitelinks | −8.0% | −9.7% | 10h26m → 9h36m | passed |
| #57 claims | −6.8% | −12.8% | 9h14m → 8h36m | passed |

Locally, c0's claims call went from 15.0 s to 10.1 s.

Measured, not kept: labels, descriptions and aliases against stored schemas (#58). All
33 (chunk, table) outputs were identical and strict, and c0's labels went from 3.7 s to
1.7 s locally, but on CI the pipeline did not move (≈): their inference is cheap there.

## Correctness

- Claims, all 11 chunk files: Arrow output, label lookup and quarantine tables
  identical with and without `claims.avsc`, every chunk on the strict path. With a schema
  from the r-chunks only, also identical: c0 fell back, the other 10 decoded strictly.
- genson: strict accepts what fits (decoding as the non-strict plan does) and rejects
  each kind of misfit (Rust tests). Python tests cover the same, the wider and narrower
  schema cases, and that a schema not fitting the target still meets the schema check's
  error.

## Missing

- A schema from 11 chunks may not hold every shape the full dump has. A chunk that does
  not fit costs a wasted strict decode before inference (genson's profile logs
  `decode (avro_schema): the rows do not all fit it`). Rerun
  `scripts/claims_avro_schema.py` on more chunks, or on the fallen-back ones, to widen
  it.
- polars-genson's release, then the wheel sources swapped for a version pin.
