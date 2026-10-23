# Records

Each push to master that changes the pipeline, oldest first: the new commit against the
previous one, the 4-chunk workload, one run each in the same GitHub Actions job, each on
its own lockfile. Speed is the previous commit's wall time over the new one's.

| date | commit | previous | speed | median wall (s), previous / new | peak RSS (GB), previous / new | device written (MB), previous / new | run |
|---|---|---|---|---|---|---|---|
| 2026-10-23 | `7ad8c9b` | `68a1d41` | 0.98× (+2.5%) | 15.4 / 15.8 | 3.6 / 3.5 | 204 / 206 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40032386680) |
