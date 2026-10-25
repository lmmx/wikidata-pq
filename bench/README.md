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

## Two tiers

- **quick** (every pull request and master push): 4 chunks, 4 workers, one group. A fast
  regression signal. It runs the real pipeline on real data, but not at the scale of a
  release: each worker takes one chunk, and the one group merges after processing, so
  worker reuse and merges overlapping processing are not exercised. It is not, by
  itself, evidence of full-pipeline throughput.
- **full** (from the Actions tab, tier `full`): the configuration of the campaign's
  66-chunk A/Bs: the 11 files cycled 6 times, 6 workers taking 8 chunks each, groups of
  22 merged while chunks are still processing. Its `before` input sets the commit to
  compare against: the campaign's reference is `2d25be6`, the last commit before #25.
  Its records are tagged with its tier, as every record is with its tier and harness
  version (`pyproject.toml`, `CHANGELOG.md`): results compare only within a tier and a
  major version.

### Does quick predict full?

Both tiers measured the campaign's two big changes, three rounds each (bench 1.1.0,
2026-10-24; the full runs are in the results store):

| change | quick | full |
|---|---|---|
| #25, `2d25be6` → `c69dcac` (merges, contention) | 1.77× ± 0.04, group merges ÷17.2 | 1.87× ± 0.11, group merges ÷23.7 |
| #26, `c69dcac` → `6090f4f` (partition writer, worker memory) | 1.36× ± 0.06, partition CPU ÷5.4, peak RSS −46% | 1.54× ± 0.01, partition CPU ÷5.1, peak RSS −44% |

Quick gets the direction and the mechanism of both right: the stage each change speeds
up, by about the same factor, and the same memory effect. It understates the end-to-end
speed-up, by 5% for #25 and 12% for #26. At full scale, merges and partitioning are a
larger share of the wall time, and the overlap of merges with processing and worker
reuse are exercised. So a quick result is a regression signal and a lower bound on
a change to those stages; a claim about release throughput needs a full run.

## The workload

- **Input:** 4 chunks, one each of the benchmark data's `c0` (a main-set chunk of 10,000
  entities), `r50` (a routed main-set chunk), `s30` and `s90` (scholarly-article chunks).
  The list is `FILES` in `ab.py`: as the pipeline gets faster, more of the 11 files make
  a bigger workload, and a new series of results.
- **Pipeline:** `pool.process_chunks` with the real `process` and `partition_chunk` for
  each chunk, 4 workers (the runner's cores), as `process-wikidata` runs a release.
- **Group:** the 4 chunks are one group, closed as `close_group` does it: the ledger's
  stages, `merge_group`, and the staging files' clean-up.
- **finalise:** after the last group closes, its local stages as `main.finalise` runs
  them for a release, claims first: each table compacted (`compact.compact_table`, the
  group files moved, as each group closes, to where its download puts them) and sorted
  (`sort_by_id`'s check, write and metadata stages). Timed as `finalise: compact` and
  `finalise: sort`; the score is the whole run, `wall, with finalise`.
  - **The claims key's buckets:** a key the release would sort through buckets (claims)
    is sorted through buckets here too: the in-memory threshold is scaled by the
    sample's fraction of the release's entities, and the bucket size is the release's,
    64 MiB of source (2 buckets at the sample's rows, 4 at three times them). The part of
    finalise that grows with the rows has the release's buckets per row, and about its
    reads per row (each source row group read directly by the one or two buckets its
    rows are in: 1.81 reads a row against 1.75-2.39 at a release, modelled). It spills
    less than a release (15% of its rows in that part against 20-30%) and in larger
    flushes (27k rows against about 13k), and its buckets each have one source file, a
    release's about 16. Finer buckets would cut through the sample's four chunks, each
    a large share of its key, and read each row group several times. The figures, and
    how the release's are modelled, are in CHANGELOG.md (7.4.0). Each run records the
    shape it measured (`sort_buckets`, `sort_bucket_rows`, `sort_spilled_share`,
    `sort_flush_rows`, and `_x3` for three times the rows), shown under the ETA.
- **The Hub is never touched:** `HF_HUB_OFFLINE=1`, and every Hub entry point of the
  finalise modules (`download`, `snapshot_download`, `HfApi`, `source_table`,
  `commit_table`, `verify_table`) is replaced with one that raises; the workflow holds no
  `HF_TOKEN`.
- **Not included:**
  - the upload to Hugging Face and its verification (network-bound, and it would
    publish), and finalise's commits, verification and cards;
  - `split-dump` and `route-release`, which make the chunks from the dump;
  - `claims_labels`, which a release builds from both sets' sorted labels.

## What the numbers mean

The quick tier is a sample: 33,220 entities (76 MB of source) of a dump of over 120
million. Its wall time is the time for those entities through the stages above, not a
release. Each result also gives the sample's throughput (entities per second) and a full
Wikidata ETA: the release 20260928's 121,815,642 entities (75,432,640 main, 46,383,002
scholarly, in 23,035 chunks; `docs/guide/performance.md`) through the same stages at
the same rate on the same runner. It is a straight extrapolation, for reading a change
at release scale ("−2h10m a release") rather than a prediction of a release's length:

- it leaves out downloading, splitting and routing the dump, the uploads to the Hub, and
  finalise (compaction, the sort, cards);
- a release runs on a bigger machine (20 cores there, 4 on a runner) with more workers;
- the sample is 51% main-set and 49% scholarly entities, against 62% and 38% in the
  release, and costs per entity differ between the sets.

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

The `bench-results` branch keeps each recorded commit's results in `runs/`. The pull
request comment's opening table takes the commit before's figures from there, as a
preview while the rounds run. The result itself always measures both commits, paired on
one runner: the same code's wall time differs by up to 1.5× across runners, and by 2–9%
across runners of one CPU model, against about 1% between two runs paired in one job.
(`ab.py --cache` can still reuse a stored measurement for local runs; the workflow does
not use it.)

## The pull request comment

Work in progress is not benchmarked: a draft pull request, or one whose head commit's
subject starts with `wip` or holds `[wip]`. Push such commits freely; mark the pull
request ready for review, or push a commit without the marker, to run it.

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
- **Order:** a round is two jobs on two runners, in parallel with every other round's:
  forward runs the commit before first, reverse the commit after first. The round's
  figures are the mean of the two, so a change is the mean of the forward and reverse
  changes, and what running first or second does falls on each commit once and cancels.
  A round shows in the comment once both its halves are in. `primer.py` touches 3 GB of
  fresh memory before each run, so each starts from the same memory history.
- **Parity:** round 1 keeps both arms' outputs, and `parity.py` checks the second arm's
  against the first's. The check covers the group files' rows and their order, the
  Arrow schemas and metadata, the quarantine files, and every audit sidecar column but
  the partition files' layout (`path`, `row_groups`, `file_size`). A change whose output differs fails the job and makes no record.
- **Harness:** both commits run under the same harness, the one of the commit after (or
  of the pull request). The job summary gives its commit.

## Measuring another repository's unreleased change

A pull request can measure a dependency's unreleased build (a polars-genson or pq-glue
branch) without building it in every job, from the wheel that repository's CI built:

1. `bench/wheels`, one line per wheel: `<owner/repo> <run id> <artifact>`, e.g.
   `lmmx/polars-genson 40082865651 wheels-linux-x86_64` (the runners are Linux x86_64).
2. The wheel downloaded to `wheels/` (`gh run download <run id> -R <owner/repo> -n
   <artifact> -D wheels`), and the dependency taken from it in `pyproject.toml`:
   `[tool.uv.sources] polars-genson = { path = "wheels/<file>.whl" }`; `uv lock` pins its
   hash. `wheels/` is not committed.

The workflow downloads each listed artifact into that commit's `wheels/` before making
its environment, and lists them in the job summary. A git source works too, built from
source in every job. Such a pull request is a measurement, not for merging.

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
