# 2026-10-24: The sort's check in exact order, and its files written by pq-glue

Continues docs/journal/2026-10-24-finalise-bench.md.

## Current State

- `sort_in_memory` checks each key with `_check_stable_sort` (#44): every output row's
  hash, as Polars reads the outputs, equal in order to the hash of the row Polars' own
  stable sort of the sources puts there, and the sort column ascending through the
  outputs, and their schema. It replaced `_ranked` (two seeds of order-insensitive
  hash sums, each row hashed with its rank in its id, on both sides) and
  `_check_sorted`'s second read of the outputs' ids. Stronger (every row compared, in
  order) and cheaper (each side read and hashed once). It catches a swap within an id,
  a changed value and a dropped row (tested on a real key). The bucket path keeps
  `_ranked`.
- `_sorted` takes a table over 64 MB a column per thread (Arrow releases the GIL): the
  claims key, which the step waits for, uses the cores the other keys' workers leave.
  The same table as `Table.take`.
- Each key's file for the Hub is written by `compact._write_file_parallel`:
  `pq_glue.write_parquet(engine="parallel")`, polars-genson's arrow-rs writer moved to
  pq-glue (lmmx/pq-glue#2), with the Hub's settings (zstd 3, a page index,
  content-defined chunking, sorting by id), row groups encoded in parallel and glued
  in order. pq-glue comes from its master by a git source until it is released.

## Why

After compaction became a glue (#42) the sort was finalise's biggest step: 25 s of the
quick tier's 41 s with finalise, the claims key (1,175,112 rows, 93 leaf columns) the
one job the step waited for. Its check hashed every row twice a side, and pyarrow
wrote its file on one thread.

## Measured

The CI benchmark, quick tier, 3 forward/reverse rounds, parity passing every time:

| PR | against | sort | wall, with finalise | CPU, finalise |
|---|---|---|---|---|
| #44 (exact-order check, columns in threads) | master | 22.89 s → 17.82 s (−22.1%) | −14.1% | −25.5% |
| #45 (pq-glue's parallel writer) | #44 | 18.13 s → 13.74 s (−24.2%) | −13.6% | +1.8% |

Together the sort is about 1.8× faster (25 s → 13.7 s). Peak RSS of all processes
moved inside the noise (3.50 → 3.64 GB). #45's first form, pyarrow on threads with each
row group's file glued by pq-glue, measured −21.4%.

The published files' bytes change with #45: encoded by arrow-rs (parquet-rs 60) in
place of pyarrow, the same values, row order, Arrow schema, row groups, compression and
sorting columns; on the claims key 3% smaller, its pages ending at 1 MiB alone (not
also every 20,000 rows), and its content-defined chunking reusing as much of a next
release as pyarrow's does (pq-glue's docs/journal/2026-10-24-writer-engines.md).

## Missing

- pq-glue released, and the git source replaced by a version.
- Which of the claims key's phases (read, sort, write, check) dominates on CI's
  runners: measured by #49, whose pipeline logs carry each large key's phases.
