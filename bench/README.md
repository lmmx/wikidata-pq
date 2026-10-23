# Benchmark

A fixed workload of the publication pipeline, run in GitHub Actions: a commit against
the commit before it, on the same runner in the same job.

| run | before | after | result |
|---|---|---|---|
| a pull request that changes the pipeline | trunk: its merge base with master | its head | a comment on the pull request, updated by each push |
| a push to master | the previous master commit | the new one | a row appended to the `bench-results` branch (see [RECORDS.md](RECORDS.md)) |
| the Benchmark workflow, from the Actions tab | the commit's parent | any commit, branch or tag | the run's summary |

## The score

Speed is the commit before's wall time over the commit after's, one run of each:
1.10× (−9.1%) means the change took 9.1% off the wall time. A change that does not touch
the pipeline's speed reads about 1.00×, within the runner's noise. Absolute times move
with the runner's hardware, so they are recorded but not scored.

## The workload

- **Input:** 4 chunks, one each of the benchmark data's `c0` (a main-set chunk of 10,000
  entities), `r50` (a routed main-set chunk), `s30` and `s90` (scholarly-article chunks).
  The list is `FILES` in `ab.py`: as the pipeline gets faster, more of the 11 files make
  a bigger workload, and a new series of results.
- **Pipeline:** `pool.process_chunks` with the real `process` and `partition_chunk` for
  each chunk, 4 workers (the runner's cores), as `process-wikidata` runs a release.
- **Group:** the 4 chunks are one group, closed as `close_group` does it: the ledger's
  stages, `merge_group`, and the staging files' clean-up.
- **Not included:**
  - the upload to Hugging Face and its verification (network-bound, and it would
    publish);
  - `split-dump` and `route-release`, which make the chunks from the dump;
  - `finalise` (compaction, sorting, cards), run once a release is uploaded.

## The data

The 11 chunk files are on the `bench-data` branch, which is data only and never merged.
The workflow reads it at a fixed commit (`DATA_COMMIT` in
`.github/workflows/bench.yml`) and checks every file against `SHA256SUMS`. A new version
of the data is a new commit, and a new series of scores.

- **Contents:** 60,000 entities of the 2026-10-12 Wikidata dump, as `split-dump` writes
  them: 7 columns of JSON text (`id`, `labels`, `descriptions`, `aliases`, `sitelinks`,
  `claims`, `entity`). polars-genson reads exactly this text in production.
- **Files:** `c0` is chunk 0; `r10` to `r90` are routed chunks of the main set, and
  `s10` to `s90` are chunks of the scholarly-article set.
- **Size:** 96 MB. A default `git clone` fetches every branch, `bench-data` included; to
  skip it, clone with `--single-branch`.

## The commit before, from the store

The `bench-results` branch keeps each master commit's measurement in `cache/`, keyed by
what it depends on: the commit's `src/`, `pyproject.toml` and `uv.lock`, the harness,
the data and the workload (`cache_key` in `ab.py`). When the commit before has an entry
measured on the same CPU model with the same Polars, it is not run again: its figures and
its outputs' fingerprint (`parity.py`) come from the entry, and only the commit after
runs. Otherwise both run, as before. Runners vary by CPU model, so an entry from another
model is never used. The summary says which it was.

## The pull request comment

One comment per pull request, edited in place. As a run starts it says the result is
pending; once a pull request has results, a new push keeps them and marks them as being
recomputed, "(pending 🔁)" on the score and "(pending 🔁 <commit>)" beside the commit
they are for, until the new results replace them. Each row gives both commits' figures
and the change: absolute, and as a percentage, or as a multiple once a figure more than
doubles (2.3×) or falls below half (÷2.5). A figure that did not change shows as "=".
Times are durations: 1h23m45s, 10m23s, 11.42s (seconds keep their decimals only under a
minute). Sizes are in the largest unit in which the commit before's value is at least 1
(1234 MB shows as 1.234 GB), set by that value for the whole row, with the decimals that
keep the measurement's precision.

## Rules

- **Packages:** each commit runs on its own `uv.lock`, so a dependency change shows in
  the comparison.
- **Order:** each round runs both commits once, in an order shuffled from a fixed
  seed (one round by default; the workflow's `rounds` input runs more). `primer.py` touches 3 GB of fresh memory before each run, so each starts from the
  same memory history.
- **Parity:** round 1 keeps both arms' outputs, and `parity.py` checks the second arm's
  against the first's. The check covers the group files' rows and their order, the
  Arrow schemas and metadata, the quarantine files, and every audit sidecar column but
  `file_size`. A change whose output differs fails the job and makes no record.
- **Harness:** both commits run under the same harness, the one of the commit after (or
  of the pull request). The job summary gives its commit.

## What is recorded

Each run records:

- wall time;
- CPU time of all processes, and of process, partition and the group merges;
- peak RSS of all processes, and of the largest worker;
- the peak rise in disk use, and the bytes the disk device wrote;
- temporary directories left behind.

Each job also records the runner's CPU, cores, memory and image, the Python and package
versions, both commits, and the data's checksum. The job summary shows the score and a
table of ranges. The raw JSON, each run's per-chunk figures and RSS samples included,
is the `bench-results` artifact.

## Running it

**In GitHub Actions:** it runs on every pull request that changes `src/`,
`pyproject.toml`, `uv.lock`, the harness or the workflow, again on each push (a push
cancels the run in progress), and keeps one comment on the pull request up to date. On
every such push to master it appends the record to the `bench-results` branch, a
data-only store that is never merged: nothing is committed to master. To run it on any
commit, branch or tag, use the Benchmark workflow in the Actions tab. Packages and data
are cached, so the runs start within a minute. A run takes a few minutes on a
standard runner (4 cores, 16 GB).

**Locally**, with the data checked out in `data/`:

```sh
git fetch origin bench-data
git worktree add --detach ../data origin/bench-data
(cd ../data && sha256sum -c SHA256SUMS)
git worktree add --detach ../before master   # or any commit
(cd ../before && uv sync --frozen)
uv sync --frozen
.venv/bin/python bench/ab.py --data ../data --out bench-results \
  --arm before=../before --arm after=.
```
