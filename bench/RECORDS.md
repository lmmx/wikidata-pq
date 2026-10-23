# Records

One row per push to master that changes the pipeline, newest first, added by the
Benchmark workflow (`bench/record.py`): the new commit against the previous one, the
4-chunk workload (README.md), one run each in the same GitHub Actions job, each on its
own lockfile. Speed is the previous commit's wall time over the new one's.

## CI results

| date | commit | previous | speed | median wall (s), previous / new | peak RSS (GB), previous / new | device written (MB), previous / new | run |
|---|---|---|---|---|---|---|---|
| 2026-10-23 | `7ad8c9b` | `68a1d41` | 0.98× (+2.5%) | 15.4 / 15.8 | 3.6 / 3.5 | 204 / 206 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40032386680) |

## Before CI

Measured on the development VM before the benchmark ran in CI, on the earlier 66-chunk
workload (the 11 files cycled 6 times), 3 interleaved rounds, each commit on its own
lockfile: not comparable with the CI rows.

| date | commit | previous | speed | wall (s), previous / new | peak RSS (GB), previous / new | device written (MB), previous / new | source |
|---|---|---|---|---|---|---|---|
| 2026-10-23 | `6090f4f` (#25 pq-glue merges, #26 claims from genson's Arrow stream) | `2d25be6` | 2.28× (medians 337.7 / 148.0 s) | 337–360 / 134–157 | 11.2–12.4 / 5.9–6.4 | 2,080–2,196 / 1,034–1,101 | docs/journal/2026-10-23-public-benchmark.md |
