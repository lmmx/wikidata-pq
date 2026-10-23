# Benchmark

A fixed workload of the publication pipeline, run in GitHub Actions: a candidate commit
against a fixed reference commit, on the same runner, in the same job. Records are in
[RECORDS.md](RECORDS.md).

## The score

The reference's median wall time over the candidate's, over 3 rounds: 2.00× means the
candidate ran the workload in half the time. A score is only comparable with scores
against the same reference, workload and data. Absolute times move with the runner's
hardware, so they are recorded but not scored.

The reference is `2d25be6` (partitioning: write the claims partition with pyarrow, #23),
the last commit before the 2026-10-23 campaign (pq-glue merges, #25; claims written
once from genson's Arrow stream, #26).

## The workload

- **Input:** 66 chunks, the 11 chunk files of the benchmark data cycled 6 times.
- **Pipeline:** `pool.process_chunks` with the real `process` and `partition_chunk` for
  each chunk, 6 workers taking 8 chunks each, as `process-wikidata` runs a release.
- **Groups:** 3 groups of 22 chunks, each closed in the background as `close_group` does
  it: the ledger's stages, `merge_group`, and the staging files' clean-up.
- **Not included:**
  - the upload to Hugging Face and its verification (network-bound, and it would
    publish);
  - `split-dump` and `route-release`, which make the chunks from the dump;
  - `finalise` (compaction, sorting, cards), run once a release is uploaded.

The input is repeated, so caching across chunks would score better here than on a real
release. Nothing in the pipeline does that.

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

## Rules

- **Both arms on the same packages:** one environment, synced from the candidate's
  `uv.lock`. The reference's code runs in it, so the score is the code's, not the
  packages'. A dependency upgrade's gain shows in the absolute times, not the score.
- **Interleaved:** each round runs both arms once, in an order shuffled from a fixed
  seed. `primer.py` touches 3 GB of fresh memory before each run, so each starts from the
  same memory history.
- **Parity:** round 1 keeps both arms' outputs, and `parity.py` checks the candidate's
  against the reference's. The check covers the group files' rows and their order, the
  Arrow schemas and metadata, the quarantine files, and every audit sidecar column but
  `file_size`. A candidate whose output differs fails the job and sets no record.
- **Harness:** a record needs the harness (`bench/` and the workflow) unchanged from
  master. The job summary gives the harness commit.

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

**In GitHub Actions:** it runs on every pull request that changes more than docs and
Markdown, again on each push (a push cancels the run in progress). To run it on any
commit, branch or tag, use the Benchmark workflow in the Actions tab. Packages and data
are cached, so the runs start within a minute. A run takes about 40 minutes on a
standard runner (4 cores, 16 GB). The reference's peak RSS is about 12 GB.

**Locally**, with the data checked out in `data/`:

```sh
git fetch origin bench-data
git worktree add --detach ../data origin/bench-data
(cd ../data && sha256sum -c SHA256SUMS)
git worktree add --detach ../reference 2d25be612c2def01f682011f4e06b7ef4a6d4e9c
uv sync --frozen
.venv/bin/python bench/ab.py --data ../data --out bench-results \
  --arm reference=../reference --arm candidate=.
```
