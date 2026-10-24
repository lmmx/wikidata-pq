# Records

One entry per run, appended by the Benchmark workflow (`bench/record.py`) to
`records.jsonl`, of which this file is a view: a push to master in the quick tier (the
new commit against the previous one), or a run of the full tier from the Actions tab.
Each entry carries the version of the harness that measured it (`bench/pyproject.toml`,
changes in `bench/CHANGELOG.md`): entries compare only within a major version. Speed is
the commit before's mean wall time over the commit after's, over the rounds.

## bench 1.x, full tier (66 chunks, 6 workers, groups of 22)

| date | bench | commit | previous | speed | wall (s), previous / new | peak RSS (GB), previous / new | device written (MB), previous / new | rounds | run |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-24 | 1.1.0 | `c69dcac` | `2d25be6` | 1.87× ± 0.11 (-46.5%) | 205.90 / 109.28 | 12.924 / 12.618 | 3,225 / 1,945 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40042031069) |
| 2026-10-23 | 1.1.0 | `6090f4f` | `c69dcac` | 1.54× ± 0.01 (-35.2%) | 103.35 / 66.96 | 12.590 / 7.034 | 1,786 / 1,575 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40042033399) |

## bench 1.x, quick tier (4 chunks, 4 workers, groups of 4)

| date | bench | commit | previous | speed | wall (s), previous / new | peak RSS (GB), previous / new | device written (MB), previous / new | rounds | run |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-23 | 1.0.0 | `7c5aeba` | `68fdc4b` | 1.02× ± 0.02 (-1.7%) | 15.69 / 15.43 | 3.641 / 3.632 | 197 / 196 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40041103525) |
| 2026-10-23 | 1.0.0 | `68fdc4b` | `89a9bb4` | 1.01× ± 0.01 (-0.6%) | 13.53 / 13.44 | 3.570 / 3.804 | 396 / 378 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40040615022) |
| 2026-10-23 | 1.0.0 | `89a9bb4` | `8172d2c` | 0.95× (+4.9%) | 10.23 / 10.73 | 3.819 / 3.952 | 381 / 389 | 1 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40039429402) |
| 2026-10-23 | 1.0.0 | `8172d2c` | `28dc50e` | 0.99× (+1.3%) | 15.25 / 15.45 | 3.562 / 3.690 | 195 / 199 | 1 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40038870040) |
| 2026-10-23 | 1.0.0 | `7ad8c9b` | `68a1d41` | 0.98× (+2.5%) | 15.39 / 15.78 | 3.663 / 3.551 | 204 / 206 | 1 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40032386680) |
