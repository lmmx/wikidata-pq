# 2026-10-25: The claims weighted by their own rows and bytes in the release ETA (bench 10.0.0)

Continues bench 9.0.0 (lmmx/wikidata-pq#119), which flagged two things it left alone: the
sample's claims per entity, and a possible copying error in the 44.7 GB figures.

## Current State

- Bench 10.0.0 (this PR) scales the pipeline's claims stages by the release's claims rows
  in each set, and finalise's claims share of its part per byte by the release's claims
  bytes. Everything else is scaled by entities, as before. Bench 9.0.0's ETAs are kept
  beside the new ones (`*_eta_9`).
- No constant was wrong: both 44.7 GB figures are measured.
- The full figures, sources and sensitivity are in bench/CHANGELOG.md 10.0.0.

## The facts, from the record

- **Entities:** main 75,432,640, scholarly 46,383,002 (`route.done`, measured).
- **Chunk files:** main 12,182 files, 44.7 GB; scholarly 10,853 files, 49.5 GB (manifest,
  measured on 2026-10-03, before any main chunk was processed).
- **The main set's claims:**
  - group files: 45.1 GB (`release_eta.py`, docs/guide/monitoring.md);
  - compacted: 44.7 GB in 30 files. The 667 buckets of 64 MiB pin it to 44.69-44.76 GB.
  - sorted: 86 part files of 500 MiB.
- **The main set's claims rows:** not measured for 20260928. 774,255,243 were measured in
  the 20260507 build, so 780 million is modelled (7.4.0).
- **The scholarly set's claims:** nothing on record. hub_audit's
  `docs/releases/*/dataset_cards_metadata.json` was never committed.
- **The other tables at 20260928:** nothing on record. Only the 20260507 build's main-set
  figures exist (docs/dataset_cards_metadata.json).

## Is 44.7 GB a copying error?

**No.** Each of the bench's 11 chunk files was processed and partitioned alone
(`2026-10-25-claims-weights/sizes.py`; results in `chunk_tables.json`). In each, the
claims partition came out at 0.91-1.11 times the chunk file:

| file | entities | chunk file | claims partition | claims rows | rows an entity |
|---|--:|--:|--:|--:|--:|
| c0 | 10,000 | 48.81 MB | 44.25 MB (0.91) | 619,789 | 62.0 |
| r10, r30, r50, r70, r90 | 21,085 | 10.23 MB | 10.39 MB (1.02) | 210,166 | 10.0 |
| s10, s30, s50, s70, s90 | 28,915 | 37.04 MB | 38.11 MB (1.03) | 732,074 | 25.3 |

So a set's compacted claims are expected to be about the size of its chunk files. The
main set's two 44.7 GB figures agree for that reason, and they were recorded two days
apart.

The routed main-set chunks have 10.0 claims rows an entity, the same as the release's
main set (10.3). Chunk c0, the dump's first 10,000 entities, has 62. On the quick tier it
is a third of the entities and 53% of the claims rows.

## Per-stage claims share (quick tier, 12 runs of run 40201381756)

- **`normalise_in_schema`:** 90% claims.
- **`write_table`:** 100% claims.
- **`partition_parquet`:** 0% claims. The claims are placed unsplit, and linked.
- **The pipeline as a whole:** 57% of its CPU time is claims (12.9 of 22.4 s).
- **finalise's part per byte:** 85% claims (`phases.table_cpu`, by enclosing call).

The claims' CPU time goes with their rows. Fitted on the full tier's 11 files:
- the main set's: 9.21 ± 0.13 µs a row, with a fixed part of 0.015 ± 0.034 s a chunk;
- per entity, by contrast, it varies 9× between files.

## The model and the change, on the same runs

PR #128's run 40205337787, 12 runs, both models computed by the harness:

| | bench 9.0.0 | bench 10.0.0 | change |
|---|---|---|---|
| ETA: pipeline | 7h20m ± 41m | 5h02m ± 27m | −31.3% ± 0.3% |
| ETA: finalise | 6h06m ± 31m | 2h58m ± 15m | −51.5% ± 0.3% |
| ETA, a release | 16h13m ± 1h28m | 10h46m ± 59m | −33.6% ± 0.2% |

The same model applied offline to #119's run 40201381756 gave −32.7%, −53.6% and −34.8%.
Parity passed, and no measured figure moved beyond noise.

**The scales:**
- the main set's claims rows: 1,172×;
- the scholarly set's: 1,960× (999 million rows, modelled from its 49.5 GB of chunk files);
- the claims key's bytes: 1,322× (96.5 GB);
- entities, for comparison: 3,667×.

**Sensitivity:** varying each weight within what the record allows moves the release
ETA's change between −33.1% and −35.5%.

## Missing

These are still scaled by entities, and likely overstated by it. They are not changed,
because the figures that would weight them aren't on record or aren't measured apart:

- **The other tables' stages:** about −12% of the pipeline ETA, using the 20260507 build's
  rows and the full tier's fixed costs per chunk.
- **The pipeline's CPU outside its stages:** mostly its workers' starts. A release has
  about 2,900, against the 14,700 that entity scaling implies: about −20% of the pipeline
  ETA.
- **`merge_group`'s claims part.**
- **The release's own per-table figures:** a release's
  `docs/releases/{set}/dataset_cards_metadata.json` committed would turn the modelled
  scholarly claims (and the other tables) into measured ones.
