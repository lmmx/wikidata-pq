# 2026-10-25: Shared repetition levels in arrow-rs's writer, measured through CI wheels

arrow-rs's Parquet writer computes repetition levels separately for every leaf column,
even when several leaves sit under the same list with only structs between them, so
their levels are the same. The claims table has many such leaves (a statement's
qualifiers, references and their snaks). Measured locally earlier, sharing those levels
cut the claims write stage by 21% (c0) to 36% (s30), with u32 dictionary indices
included. This entry measures the change end to end on CI. Nothing has shipped: there
is no release, no crates.io publish and no upstream PR.

## The patch

It changes only parquet 60.0.0's `arrow/arrow_writer/levels.rs`: +108/−15 lines of
code, doc comments included, and +207 lines of tests.

- **Grouping:** before writing, `calculate_array_levels` groups the leaves under the
  same innermost list with only structs between them.
- **Computing once:** every leaf in a group gets the same repetition-level operations,
  so the first leaf computes them. The rest are `Absent` and skip every repetition-level
  write.
- **Sharing:** when the levels are complete, the rest take
  `LevelData::Shared(Arc<Vec<i16>>)`, which column writers read exactly as
  `Materialized`.
- **Writes:** a write to a shared copy copies it first.
- **No public API changes:** `Shared` is `pub(crate)`.

## Where it was measured

| PR | What it holds |
|---|---|
| lmmx/pq-glue#11 (branch `exp/shared-rep-levels`) | parquet 60.0.0 vendored unmodified, then the patch, then `[patch.crates-io]`, plus a CI job running arrow-rs's own writer tests on the vendored crate |
| lmmx/polars-genson#316 (branch `bench/shared-rep-levels`) | genson 1.5.0, patched to pq-glue#11's head |
| lmmx/wikidata-pq#120 | the two CI wheels through `bench/wheels`; bench 8.0.0, quick tier, 3 rounds, paired against master d8f2826 |

The branches are kept.

## Byte identity

- **arrow-rs's tests:** all 167 `arrow_writer` unit tests and its 15 layout tests pass.
- **New tests:**
  - A random-equivalence test compares shared levels with per-leaf levels, logically
    and by form. It covers lists, large lists, maps, list views, fixed-size lists and
    nested structs, with nulls from none to all, and slices.
  - A test checks that a shared copy is copied before it is changed.
- **Written files:** on the 11 bench-profile chunks, all 77 `process()` files are
  identical by SHA-256, and so are the 154 `_write_file_parallel` rewrites (hub and
  scratch modes).
- **Bench parity:** passed.

## CI bench, before → after

The ± is the 95% interval of the paired change.

| figure | before | after | change |
|---|---|---|---|
| `partitioning.core.write_table` (pipeline) | 5.5 s | 5.0 s | **−8.8% ±1.4%** |
| `compact._write_file_parallel` (finalise, per byte) | 15.4 s | 14.5 s | **−6.0% ±3.2%** |
| CPU, process | 14.0 s | 13.5 s | **−3.4% ±2.0%** |
| CPU, all processes; CPU, finalise | | | within noise |
| peak RSS, finalise, all processes | 6.054 GB | 5.902 GB | **−2.5% ±1.2%** |
| peak RSS, finalise 3×, one worker | 4.271 GB | 4.175 GB | **−2.2% ±0.9%** |
| ETA: pipeline, finalise, a release | 16h09m53s | 16h00m48s | within noise |

**Release model** (not measured): 8.8% of `write_table`'s 2h14m plus 6.0% of
`_write_file_parallel`'s 2h22m is about −20 min of a 17h release, about 2%. The quick
tier's release ETA interval is about ±2h, so it cannot resolve that.

The local figures were larger for three reasons:
- they were the claims write alone, and included u32 dictionary indices;
- `write_table` covers every table, and most tables are flat;
- the bench runs 4 chunks.

## What shipping would cost

- **A fork** (vendored, as in pq-glue#11):
  - It is 6.2 MB and 238 files, and must match the version pq-glue and genson ask
    for, or `[patch]` does nothing.
  - Every arrow-rs release, about monthly, means re-vendoring, reapplying the patch,
    rerunning the tests, and moving pq-glue, genson-core and polars-genson together.
  - Python wheels get the patch. crates.io users would need a renamed fork crate.
- **Upstream:** arrow-rs main changed `levels.rs` on 2026-10-21 (#11278, a `Required`
  builder variant). The patch still applies, and needs one more arm in
  `collect_rep_groups` to compile.
  - An upstream PR needs a rebase, the random-equivalence test, and arrow-rs's criterion
    `arrow_writer` numbers on list-of-struct columns.
  - u32 dictionary indices belong in a separate PR: earlier measurements found they save
    about nothing.

## Decision

Not shipped by the campaign. The experiment PRs are closed and their branches kept.

The patch is correct, and it is cheaper on CI in both writes and in finalise's memory.
At about 2% of a release, it doesn't justify a fork's monthly maintenance. The low-cost
route is an upstream PR, after which the gain arrives with a normal arrow-rs version
bump. That is the user's call, recorded under "Pending the user" in
lmmx/wikidata-pq#97.
