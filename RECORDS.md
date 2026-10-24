# Records

One entry per run, appended by the Benchmark workflow (`bench/record.py`) to
`records.jsonl`, of which this file is a view: a push to master in the quick tier (the
new commit against the previous one), or a run of the full tier from the Actions tab.
Each entry carries the version of the harness that measured it (`bench/pyproject.toml`,
changes in `bench/CHANGELOG.md`): entries compare only within a major version. Speed is
the commit before's mean wall time over the commit after's, over the rounds.

## bench 3.x, quick tier (4 chunks, 4 workers, groups of 4)

| date | bench | commit | previous | speed | wall (s), previous / new | peak RSS (GB), previous / new | device written (MB), previous / new | rounds | run |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-24 | 3.0.0 | `d97935f` | `c4a090f` | 0.94× ± 0.06 (+6.4%) | 11.40 / 12.15 | 3.695 / 3.663 | 504 / 491 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40054339707) |

## bench 2.x, quick tier (4 chunks, 4 workers, groups of 4)

| date | bench | commit | previous | speed | wall (s), previous / new | peak RSS (GB), previous / new | device written (MB), previous / new | rounds | run |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-24 | 2.1.1 | `c4a090f` | `fb3cc80` | 1.19× ± 0.01 (-16.1%) | 15.85 / 13.30 | 3.719 / 3.356 | 198 / 194 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40052912989) |
| 2026-10-24 | 2.1.1 | `fb3cc80` | `3abf40b` | 1.04× ± 0.06 (-3.8%) | 15.63 / 15.06 | 3.703 / 3.683 | 196 / 196 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40050433365) |
| 2026-10-24 | 2.1.0 | `3abf40b` | `9374bda` | 0.98× ± 0.02 (+2.0%) | 14.42 / 14.68 | 3.663 / 3.660 | 252 / 254 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40046630308) |
| 2026-10-24 | 2.0.1 | `9374bda` | `01333f6` | 1.00× ± 0.01 (-0.3%) | 13.02 / 12.98 | 3.741 / 3.686 | 240 / 239 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40045725437) |
| 2026-10-24 | 2.0.0 | `01333f6` | `059b583` | 0.99× ± 0.01 (+0.6%) | 13.62 / 13.69 | 3.615 / 3.758 | 332 / 295 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40045274880) |

## bench 1.x, full tier (66 chunks, 6 workers, groups of 22)

| date | bench | commit | previous | speed | wall (s), previous / new | peak RSS (GB), previous / new | device written (MB), previous / new | rounds | run |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-24 | 1.1.0 | `c69dcac` | `2d25be6` | 1.87× ± 0.11 (-46.5%) | 205.90 / 109.28 | 12.924 / 12.618 | 3,225 / 1,945 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40042031069) |
| 2026-10-23 | 1.1.0 | `6090f4f` | `c69dcac` | 1.54× ± 0.01 (-35.2%) | 103.35 / 66.96 | 12.590 / 7.034 | 1,786 / 1,575 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40042033399) |

## bench 1.x, quick tier (4 chunks, 4 workers, groups of 4)

| date | bench | commit | previous | speed | wall (s), previous / new | peak RSS (GB), previous / new | device written (MB), previous / new | rounds | run |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-24 | 1.1.1 | `059b583` | `a955439` | 1.02× ± 0.02 (-1.7%) | 13.87 / 13.63 | 3.672 / 3.942 | 372 / 365 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40043511839) |
| 2026-10-24 | 1.1.0 | `a955439` | `7c5aeba` | 1.00× ± 0.01 (+0.3%) | 11.95 / 11.97 | 3.817 / 3.761 | 459 / 196 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40043147066) |
| 2026-10-23 | 1.0.0 | `7c5aeba` | `68fdc4b` | 1.02× ± 0.02 (-1.7%) | 15.69 / 15.43 | 3.641 / 3.632 | 197 / 196 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40041103525) |
| 2026-10-23 | 1.0.0 | `68fdc4b` | `89a9bb4` | 1.01× ± 0.01 (-0.6%) | 13.53 / 13.44 | 3.570 / 3.804 | 396 / 378 | 3 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40040615022) |
| 2026-10-23 | 1.0.0 | `89a9bb4` | `8172d2c` | 0.95× (+4.9%) | 10.23 / 10.73 | 3.819 / 3.952 | 381 / 389 | 1 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40039429402) |
| 2026-10-23 | 1.0.0 | `8172d2c` | `28dc50e` | 0.99× (+1.3%) | 15.25 / 15.45 | 3.562 / 3.690 | 195 / 199 | 1 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40038870040) |
| 2026-10-23 | 1.0.0 | `7ad8c9b` | `68a1d41` | 0.98× (+2.5%) | 15.39 / 15.78 | 3.663 / 3.551 | 204 / 206 | 1 | [run](https://github.com/lmmx/wikidata-pq/actions/runs/40032386680) |
