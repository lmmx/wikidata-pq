# Benchmark

A fixed workload of the publication pipeline, run in GitHub Actions, two commits on the
same runner in the same job. Each kind of run answers one question:

| run | arms | answers | result |
|---|---|---|---|
| a pull request that changes the pipeline | its merge base with master, against its head | does this pull request make the pipeline faster or slower, and is its output the same? | a comment on the pull request |
| a push to master | the fixed reference, against master | how fast is the pipeline now, against where the campaign started? | a row in [RECORDS.md](RECORDS.md) |
| the Benchmark workflow, from the Actions tab | the fixed reference, against any commit, branch or tag | the same, for that commit | the run's summary |

## The score

The first arm's median wall time over the second's, over 3 rounds: 2.00× means the
second arm ran the workload in half the time, and the change in wall time is given as a
percentage too. On a pull request that does not change the pipeline's speed it reads
about 1.00×, within the runner's noise. A record is only comparable with records against
the same reference, workload and data. Absolute times move with the runner's hardware,
so they are recorded but not scored.

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

- **Packages:** on a pull request, each arm runs on its own `uv.lock`, so a dependency
  change shows in the comparison. A record runs the reference's code on master's
  `uv.lock`, so it measures the code on current packages.
- **Interleaved:** each round runs both arms once, in an order shuffled from a fixed
  seed. `primer.py` touches 3 GB of fresh memory before each run, so each starts from the
  same memory history.
- **Parity:** round 1 keeps both arms' outputs, and `parity.py` checks the second arm's
  against the first's. The check covers the group files' rows and their order, the
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

**In GitHub Actions:** it runs on every pull request that changes `src/`,
`pyproject.toml`, `uv.lock`, the harness or the workflow, again on each push (a push
cancels the run in progress), and keeps one comment on the pull request up to date. On
every such push to master it adds the record to RECORDS.md itself. To run it on any
commit, branch or tag, use the Benchmark workflow in the Actions tab. Packages and data
are cached, so the runs start within a minute. A run takes about 30 to 40 minutes on a
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
