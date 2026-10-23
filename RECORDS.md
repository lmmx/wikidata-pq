# Records

Each push to master that changes the pipeline, oldest first: the new commit against the
previous one, the 4-chunk workload, one run each in the same GitHub Actions job, each on
its own lockfile. Speed is the previous commit's wall time over the new one's.

| date | commit | previous | speed | median wall (s), previous / new | peak RSS (GB), previous / new | device written (MB), previous / new | run |
|---|---|---|---|---|---|---|---|
| 2026-10-23 | `7ad8c9b` | `68a1d41` | 0.98× (+2.5%) | 15.4 / 15.8 | 3.6 / 3.5 | 204 / 206 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40032386680) |
| 2026-10-23 | `8172d2c` | `28dc50e` | 0.99× (+1.3%) | 15.2 / 15.4 | 3.5 / 3.6 | 195 / 199 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40038870040) |
| 2026-10-23 | `89a9bb4` | `8172d2c` | 0.95× (+4.9%) | 10.2 / 10.7 | 3.7 / 3.9 | 381 / 389 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40039429402) |
