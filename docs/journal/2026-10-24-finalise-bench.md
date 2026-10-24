# 2026-10-24: finalise measured, and compaction glued

Continues docs/journal/2026-10-24-genson-writer.md.

## Current State

- The benchmark runs finalise's local stages (bench 3.0.0, #41): each table compacted and
  sorted after the last group closes, the Hub never touched. The score is `total_wall`,
  the pipeline plus finalise.
- Compaction glues each key's group files with `pq_glue.concat`, which copies their
  encoded row groups without decoding (#42). It does not re-encode them through pyarrow.
  The output is the sort's input only: it is never uploaded.
- A glued file is checked against its group files byte for byte, without decoding: the
  schema, the row-group count, each row group's rows and columns, and each column chunk's
  bytes (from its dictionary or first data page, for its compressed size).
- Finalise's light jobs (`tasks_per_child=None`) share one spawned pool per worker count
  for the run (`wikidata.parallel`), so workers import once, not once per job. Heavy jobs
  (`tasks_per_child=1`) keep a pool of their own. A failed or abandoned call cancels its
  futures, waits for the running ones and drops the pool; the next call makes a new one.
- bench 3.0.1: each finalise step shuts the shared pool down when it ends, so its
  workers' CPU is counted.

## Why

**Finalise outweighed the pipeline** (quick tier, 4 chunks, 33,220 entities, bench 3.0.0
at d97935f): pipeline 13.6 s, compaction 24.3 s, sort 31.6 s, finalise CPU about 2m28s.
At this size finalise is overhead per key, not per row: 2,324 keys (aliases 489,
descriptions 442, labels 564, links 827, claims and entities 1 each).

**Compaction re-encoded what it only concatenated.** Every key's group files share one
schema, and compaction's output exists only to be read by the sort, so each group's
encoded row groups can be copied as they are.

**The sort's workers spent 31% of their samples importing** (py-spy, the sort over the
quick tier's keys): each job spawned a worker that imported polars, pyarrow and wikidata
for one key. Next: `_write_file` with the Hub settings 18.7%, `_read` 17.7%, `_ranked`
13.6%, `_sorted` 10.7%, `_check_sorted` 9.7%.

## Measured

CI benchmark, quick tier, 3 forward/reverse rounds, against d97935f:

| commit | wall, with finalise | compact | sort | parity |
|---|---|---|---|---|
| fdd889b, glue | −14.5% | 21.55 s → 12.36 s (÷1.74) | ≈ | passed |
| fc3cd1e, byte check | 1m09s → 50.60 s (−26.4%) | 23.95 s → 6.10 s (÷3.9) | ≈ | passed |
| 5582a54, shared pool | 1m06s → 41.23 s (−37.7%) | 23.06 s → 3.05 s (÷7.6) | 30.08 s → 25.24 s (−16.1%) | passed |
| 71a78c6, bench 3.0.1 | 1m02s → 40.05 s (−34.8%) | 21.36 s → 2.94 s (÷7.3) | 28.04 s → 24.94 s (−10.9%) | passed |

The pipeline's own wall is unchanged (≈): only finalise moved. At fc3cd1e the Full
Wikidata ETA went from 70h03m to 51h32m. That figure extrapolated finalise by entity
count, which #43 drops because finalise's cost is per key.

5582a54's `CPU, finalise` read 2m22s → 4.5s, which is wrong: the shared pool's workers
outlived the measurement, so their CPU went uncounted. bench 3.0.1 fixes it: at 71a78c6
`CPU, finalise` is 2m12s → 1m14s (−43.6%), each step paying its own workers' start.

## Correctness

- Parity passed on every commit: all 2,324 keys the same, quarantine and sidecars the
  same.
- The byte check, run locally on a real glue, passes, and fails both on a group file
  with one row changed and on a run in the wrong order.
- The shared pool, run locally: reused across calls; a failing job raises and drops
  the pool; the next call works.

## Missing

- The sort is now the largest finalise step (25 s). Leads: feed the sort straight from
  the group files, skipping compaction where a table needs no deduplication; batch
  several keys per job to cut per-key overhead; the Hub-settings write; `_check_sorted`
  on data the sort just made.
