# 2026-10-25: The bench's run order, measured: an order effect the pairs cancel, no bias

Three master ledger runs in a row read a split CPU rise: #140 (79438b1), #147 (5df0c40)
and #148 (0a574cf), none of which touched the split. In each one the forward jobs, where
the commit before runs first, read more of the rise than the reverse jobs. #158's PR run
read finalise CPU +1.2% ±1.1 the same way. This entry measures what running second does to
each figure, and whether the paired estimate and its interval carry any of it, from
benchmark runs already made. No new CI runs.

## Current State

### How a run is scheduled and combined

- A benchmark run is six jobs: rounds 1 to 3, each as a forward job and a reverse job
  (`.github/workflows/bench.yml`, `round` matrix). Each job runs on its own runner and
  measures both arms once, one after the other (`bench/ab.py` `main`, `--rounds 1`).
  Forward runs the commit before first; reverse runs the commit after first.
- Before its timed run, each arm reads its input files, imports its Polars, pyarrow and
  polars-genson, and touches 3 GB of fresh memory (`bench/ab.py` `run_once`,
  `bench/primer.py`). The timed run starts with the bz2 split (`bench/pipeline.py`
  `split_stage`), then the pipeline, then finalise. The orjson and gz splits run after the
  timed run (`bench/pipeline.py` `split_gz`).
- Each round's ratio is the mean of its two jobs' after/before ratios (`bench/combine.py`
  `paired`). Each direction runs exactly 3 times, so an effect of running first or second
  falls on each arm equally and cancels within every round.
- The reported change is the mean of the 3 round ratios, ± Student's t at 2 degrees of
  freedom (4.30) × their standard deviation / √3. A figure is flagged when its mean lies
  outside that interval (`bench/combine.py` `noise`, `interval`, `T95`). This is the t
  interval on 3 points, not a normal one.

### The data

- 72 master quick-tier runs: bench-results' `runs/quick/<sha>.json`, each a list of the
  run's six `results.json`, bench 1.1.1 to 15.2.0, up to 3046fdb (#158's ledger run).
- 99 pull-request runs: the round artifacts' `results.json` of the 100 most recent
  successful pull-request bench runs (2026-10-23 to 2026-10-25, bench 7.4.0 to 15.2.0),
  gathered one artifact at a time (`2026-10-25-bench-order-bias/gather.sh`). One more
  run's artifacts were gone (40197351876).
- 45 of the 171 runs are A/A: both arms have the same `cache_key` (src, pyproject.toml,
  uv.lock). These are the harness-only commits on master, #145 (b989521) among them, and
  the bench-only pull requests.

### The order effect, o

o is the mean log ratio of the arm run second over the arm run first. Each run estimates
it as (mean forward log ratio − mean reverse log ratio) / 2, whatever the change between
the arms. The A/A runs also estimate it directly, from each job's second/first. Both
are pooled with 95% t intervals (`2026-10-25-bench-order-bias/order.py`).

| figure | o, all 171 runs | o, A/A jobs | o, bench 13.0.0 on (47 runs) | per-run o, SD |
|---|---|---|---|---|
| split CPU (`split_cpu_total`; the split ETA scales it) | **+0.81% ±0.43** | +1.04% ±0.98 | +0.34% ±0.65 | 2.3% |
| orjson split CPU (after the timed run) | +0.43% ±0.61 | −0.71% ±1.08 | +0.50% ±0.78 | |
| gz split CPU (after the timed run) | +0.20% ±0.16 | +0.01% ±0.34 | +0.17% ±0.20 | |
| finalise CPU | +0.17% ±0.20 | +0.19% ±0.44 | +0.06% ±0.41 | 1.2% |
| measured CPU | +0.16% ±0.34 | +0.12% ±0.76 | +0.14% ±0.41 | 1.3% |
| measured wall | −0.29% ±0.37 | −0.12% ±0.73 | −0.18% ±0.39 | 1.4% |
| ETA, a release | −0.43% ±0.30 | −0.24% ±0.41 | +0.21% ±0.36 | 1.6% |
| ETA: finalise | +0.30% ±0.34 | +0.33% ±0.64 | **+1.15% ±0.47** | 1.9% |
| ETA: pipeline | **−1.69% ±0.51** | −1.41% ±0.56 | −0.93% ±0.62 | 2.8% |
| pipeline wall (`wall`) | **−1.78% ±0.47** | −1.59% ±0.56 | −0.79% ±0.54 | 2.7% |
| device bytes written | **−12.0% ±2.4** | −13.4% ±4.3 | | |

- A run's second arm spends about 0.8% more split CPU than its first. The pipeline's wall
  and ETA run 1.7–1.8% faster in the second arm, and the second arm writes 12% fewer
  device bytes. Since bench 13.0.0 the finalise ETA reads +1.2% in the second arm. These
  are the effects distinct from zero.
- The split's order effect falls on the bz2 split, the first heavy stage of each arm's
  timed run. The orjson and gz splits run later on the same slice and carry none of it.
  For the first arm, the stage before the bz2 split is the job's setup (uv sync, the
  reshape build). For the second arm it is the first arm's whole run.
- The per-run o scatters far more than its pooled mean: SD 2.3% for split CPU and 1.2% for
  finalise CPU. Its sign varies too: 61% of runs read a positive split o, and 55% a
  positive finalise o.

### Does the paired change carry it: the A/A runs

| figure | A/A runs | mean reported change | flagged at 95% | observed / claimed variance |
|---|---|---|---|---|
| split CPU | 33 | +0.36% ±1.06 | 0/33 | 1.06 |
| finalise CPU | 41 | +0.03% ±0.44 | 4/41 | 1.21 |
| measured CPU | 17 | +0.05% ±0.71 | 1/17 | 0.86 |
| measured wall | 17 | +0.03% ±0.72 | 0/17 | 0.96 |
| ETA, a release | 38 | +0.16% ±0.39 | 0/38 | 0.72 |
| ETA: finalise | 38 | +0.50% ±0.73 | 2/38 | 1.16 |
| ETA: pipeline | 38 | +0.18% ±0.58 | 0/38 | 1.07 |
| pipeline wall | 45 | +0.25% ±0.64 | 0/45 | 1.36 |

- Every timing figure's mean A/A change is within its interval of zero. That holds for the
  figures with the largest order effects too (pipeline wall: o −1.8%, mean A/A change
  +0.25% ±0.64). The pairing removes the order effect from the estimate.
- Master's A/A runs alone read split CPU +1.37% ±1.23 (10 runs). The pull requests' A/A
  runs read −0.07% ±1.38 (23 runs), and all 33 together read +0.36% ±1.06.
- 24 of the 566 (A/A run, figure) changes are flagged: 4.2%, with a 95% bootstrap interval
  over runs of 1.9% to 6.8%. The duplicate figures (split ETA, split's own CPU) are left
  out of that count. The changes over their claimed standard error (interval / 4.30)
  exceed t₂'s 80%, 90% and 95% points in 17.8%, 7.2% and 4.2% of cases (expected 20%,
  10% and 5%).
- The A/A changes' observed variance over the variance their intervals claim is 0.72 to
  1.63 per figure (n = 17 to 45). The 3 rounds' scatter accounts for the run-to-run scatter
  of the change: the rounds behave as independent.
- #145 (b989521) flagged 2 of about 50 figures. At 5% each, the bench flags about 2.5 per
  A/A run.

### Does the commit after read high: the same commit measured twice

- A commit measured as the commit after in one run can be measured again as the commit
  before in the next master run, or in a pull request on it. There are 90 such links on
  the same pipeline trees (src, reshape, pyproject.toml, uv.lock) and the same bench
  major. Per CPU model, the second measurement over the first reads −1.10% ±1.89 for
  split CPU (66 links). It reads −0.17% ±0.95 for finalise CPU (83), −0.27% ±1.10 for
  the release ETA (72), and +0.01% ±0.83 for pipeline wall (89)
  (`2026-10-25-bench-order-bias/links.py`). None of the links reads an arm effect.

### The three split rises

| run | split CPU change | forward jobs | reverse jobs | o |
|---|---|---|---|---|
| #140, 79438b1 ([40218127750](https://github.com/lmmx/wikidata-pq/actions/runs/40218127750)) | +5.5% ±10.4 | +10.2, +18.8, −0.4 | +0.7, +0.7, +3.2 | +3.7% |
| (between: f806305) | −1.6% ±9.5 | −2.0, +4.5, +4.4 | −1.1, +0.0, −15.2 | +4.1% |
| #147, 5df0c40 ([40221195177](https://github.com/lmmx/wikidata-pq/actions/runs/40221195177)) | +5.2% ±6.1 | +16.5, +6.2, +0.0 | −0.3, +1.2, +7.8 | +2.1% |
| #148, 0a574cf ([40223102466](https://github.com/lmmx/wikidata-pq/actions/runs/40223102466)) | +2.6% ±4.8 | +4.2, +7.1, +2.6 | +0.9, +2.0, −1.2 | +2.0% |

- None of the three rises was flagged: each lies inside its own interval.
- Each run's o lies within 1.6 per-run SDs (2.3%) of the pooled +0.8%, and f806305's
  between them read +4.1% with a fall. With 61% of runs reading a positive split o, four
  runs in a row reading forward above reverse comes about 15% of the time.
- Measured again as the commit before in later runs, on the same CPU model
  (`2026-10-25-bench-order-bias/revisit.py`):
  - 79438b1 read 42.0 s on an EPYC 7763 (1 job) and 28.1 s on a 9V45 (3) in its own run.
    In 25 and 16 later jobs it read 38.5 s and 25.7 s. Its predecessor read 38.1 s and
    26.0 s in the same run.
  - 5df0c40 read 42.8 s on a 7763 in its own run, and 40.3 s in 4 later jobs.
  - 0a574cf read 41.4 s on a 7763 in its own run, and 39.5 s in 16 later jobs; its
    predecessor read 40.2 s.

  Each rise falls back to its predecessor's level when the same code is measured again.

### #158's finalise CPU

- #158's PR run ([40231608057](https://github.com/lmmx/wikidata-pq/actions/runs/40231608057))
  read finalise CPU +1.23% ±1.09: forward +0.9, +3.4, +2.7, reverse +0.6, −0.5, +0.3. Its
  o was +1.06%, within one per-run SD (1.2%) of the pooled +0.17%. Its round ratios were
  1.0075, 1.0145 and 1.0150, three points whose small spread made the interval narrow.
- The same change on master (3046fdb,
  [40232702757](https://github.com/lmmx/wikidata-pq/actions/runs/40232702757)) read
  finalise CPU +0.51% ±12.18: forward +5.8, −0.7, +0.5, reverse +0.2, −9.6, +6.9. One
  reverse job read −9.6% to −10.1% on every CPU figure at once.

### The mean of ratios

- `combine.paired` averages each round's two ratios arithmetically. A mean of x and 1/x is
  never below 1, so a noisy figure reads high in A/A runs. Device bytes written read
  +7.4% ±5.2 over the 45 A/A runs this way, against +1.6% ±3.7 as a geometric mean of the
  same jobs. For the timing figures the two differ by 0.2% or less: split CPU +0.36% and
  +0.20%, finalise CPU +0.03% and −0.01%, peak RSS +0.30% and +0.14%.

### Reproducing

- `2026-10-25-bench-order-bias/gather.sh OUT RUN_ID...` writes a run's six round
  artifacts' `results.json` as `OUT/<run id>.json`.
- `python3 -I 2026-10-25-bench-order-bias/order.py [--min-bench N] RUN.json...` reports
  o per figure, the A/A changes and the interval's calibration. It also takes the
  bench-results branch's `runs/quick/*.json`.
- `python3 -I 2026-10-25-bench-order-bias/links.py --prs OUT REPO records.jsonl RUNS_DIR`
  reports the same-commit links. `revisit.py SHA... -- RUN.json...` reports one commit's
  split CPU across runs.
