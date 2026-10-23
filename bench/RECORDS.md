# Records

The CI record is on the `bench-results` branch, a data-only store that the Benchmark
workflow appends to on each push to master that changes the pipeline, and that is never
merged:
[RECORDS.md](https://github.com/lmmx/wikidata-pq/blob/bench-results/RECORDS.md) (one row
per push, the new commit against the previous one),
[records.jsonl](https://github.com/lmmx/wikidata-pq/blob/bench-results/records.jsonl)
(the same as JSON) and `runs/` (each run's raw results).

## Before CI

Measured on the development VM before the benchmark ran in CI, on the earlier 66-chunk
workload (the 11 files cycled 6 times), 3 interleaved rounds, each commit on its own
lockfile: not comparable with the CI record.

| date | commit | previous | speed | wall (s), previous / new | peak RSS (GB), previous / new | device written (MB), previous / new | source |
|---|---|---|---|---|---|---|---|
| 2026-10-23 | `6090f4f` (#25 pq-glue merges, #26 claims from genson's Arrow stream) | `2d25be6` | 2.28× (medians 337.7 / 148.0 s) | 337–360 / 134–157 | 11.2–12.4 / 5.9–6.4 | 2,080–2,196 / 1,034–1,101 | docs/journal/2026-10-23-public-benchmark.md |
