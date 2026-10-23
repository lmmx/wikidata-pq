# Records

Scores of the benchmark (README.md): the reference `2d25be6`'s median wall time over the
candidate's, 66 chunks, 3 rounds, both arms in the same GitHub Actions job. The CI rows
are added by the workflow, one for each push to master (`bench/record.py`).

## CI results

| date | candidate | score | median wall (s), reference / candidate | peak RSS (GB), reference / candidate | device written (MB), reference / candidate | run |
|---|---|---|---|---|---|---|
| | | | | | | |

## Before CI (not comparable)

Measured on the development VM before this benchmark existed, not in CI. The setup was
the same workload, 3 interleaved rounds, but each arm used its own lockfile (the
reference on polars-genson 1.2.0). The score is of the same kind; the times are not
comparable with CI's.

| date | candidate | score | wall (s), reference / candidate | peak RSS (GB) | device written (MB) | source |
|---|---|---|---|---|---|---|
| 2026-10-23 | `6090f4f` (#25 pq-glue merges, #26 claims from genson's Arrow stream), PyPI polars-genson 1.3.1, pq-glue 0.1.0 | 2.28× (medians 337.7 / 148.0 s) | 337–360 / 134–157 | 11.2–12.4 / 5.9–6.4 | 2,080–2,196 / 1,034–1,101 | docs/journal/2026-10-23-public-benchmark.md |
