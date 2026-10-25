# 2026-10-25: Statistics on `id` alone for the claims partition files (tried, not shipped)

Continues docs/journal/2026-10-25-partition-costs.md.

## Current State

- The claims partition files are still written by `polars_genson.write_parquet` with
  statistics and a column index on all 93 leaves. Nothing here changed on master.
- lmmx/pq-glue#9 (closed, branch `write/statistics-columns` kept) adds
  `statistics_columns` and `page_index` to pq-glue's `write_parquet`. `None` keeps
  today's files byte for byte.
- lmmx/wikidata-pq#94 (closed) switched `write_table` to `pq_glue.write_parquet` with
  `statistics_columns=["id"]`, taking pq-glue from #9's CI wheel.

## What reads the claims files' statistics

- Only `partitioning.core.file_stats`, called by `write_sidecar` from `place_unsplit`,
  reads statistics. It takes `id`'s min and max, and scans the file if they are missing.
  So `id`'s statistics have to stay.
- `pq_glue.concat`/`concat_many` copy row groups and their statistics as they are into
  the group files. The group files are uploaded to the build branch but are transient:
  `sort_by_id` rewrites every key with `compact._write_file_parallel` (full statistics,
  page index, content-defined chunking), and `commit_table` deletes the group files. The
  released part files would not have changed.

## Measured

**On its own** (a claims table read from IPC): `id`-only statistics cut write CPU by
about 5–15%, with no RSS difference.

**CI bench** (quick tier, 3 rounds, #94): parity passed, and every column chunk's
encoded bytes are identical. Only the 92 non-`id` leaves lose their statistics and
column index.

| | before | after |
|---|---|---|
| `write_table` | 4.6s | 4.6s |
| CPU, all processes | 21.4s | 21.4s |
| peak RSS, one worker | 1.742 GB | 2.171 GB (+24.7% ±3.9%) |
| ETA, a release | 14h53m | 14h59m (within noise) |

## Why it didn't pay

The memory rise comes from the writer swap #94 tested, not from selective statistics.
Both packages run the same Rust writer, but `_polars_genson.abi3.so` links mimalloc
while `_pq_glue.abi3.so` uses the system allocator. Inside `normalise_claims_direct` on
c0, pq-glue's writer allocates about 400 MB of fresh pages after genson's stream (RSS
~1400 → ~1810 MB), against +10 MB for genson's own writer. The CPU saving measured on
its own did not show in the pipeline. Changing allocators is ruled out.

## If it comes back

The route without the writer swap is `statistics_columns` through polars-genson's own
`write_parquet`. That needs a pq-glue release with #9 and a polars-genson change and
release, for at most about 1% of the pipeline. It isn't worth it unless later profiling
shows the statistics cost has grown.
