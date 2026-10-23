# 2026-10-23: A public benchmark in CI

Continues docs/journal/2026-10-23-claims-stream.md.

## Current State

The pipeline's benchmark runs in GitHub Actions (`.github/workflows/bench.yml`, rules and
reproduction in `bench/README.md`, scores in `bench/RECORDS.md`).

- **Why:** every figure so far came from a development VM. The VM changed mid-campaign
  (about 1.9 times slower), and each comparison had its own baseline, so absolute times
  from different entries cannot be compared. In CI, the two commits compared run in the
  same job.
- **What is compared:** a commit against the one before it. A pull request is its merge
  base with master against its head, and the result is a comment on the pull request. A
  push to master is the previous master commit against the new one, and the result is a
  row in `bench/RECORDS.md`. Each commit runs on its own lockfile.
- **Workload:** 4 chunks (`c0`, `r50`, `s30`, `s90`), one run each: `pipeline.py` (moved
  from the journal harness, plus the CPU of all processes), 4 workers, one group merged,
  without the upload. The campaign's A/Bs used 66 chunks and 3 rounds, about 40 minutes
  a comparison in CI; 4 chunks keep a comparison to a few minutes, and more of the 11
  files make a bigger workload as the pipeline gets faster.
- **Data:** the 11 chunk files, on the `bench-data` branch at a fixed commit, checked
  against `SHA256SUMS`. A GitHub release asset or a separate data repository would keep
  them out of default clones, but neither could be created from the session that set
  this up. The branch can be moved to either later; the workflow names only a commit.

### Data format, measured

Before choosing what to host, the 11 chunks (96.1 MB of Parquet) against the same rows as
text:

| form | size | unpack | rebuilds the Parquet byte for byte |
|---|---|---|---|
| Parquet, plain tar | 96.1 MB | none | (it is the input) |
| Parquet tar, zstd -19 / xz -9 | 95.3 / 95.7 MB | | yes |
| JSON lines, raw | 1,347 MB | | |
| JSON lines, xz -9 | 77.9 MB | 2.7 s, then 31 s to write the Parquet | yes, 11/11, with polars 2.0.0 |
| JSON lines, zstd -19 --long | 79.1 MB | 1.1 s, then 31 s | yes, 11/11, with polars 2.0.0 |

The JSON saves 19% but makes the inputs depend on the Polars version that rebuilds them,
so the Parquet is hosted as it is. The chunks' columns are JSON text already: the
benchmark starts where polars-genson's input starts in production. Starting from the dump
itself (`split-dump` and `route-release`) would time stages no change here touched, and
would be a separate score.

## The campaign so far

Measured on the development VM, 66 chunks, 3 interleaved rounds, each arm on its own
lockfile (`2026-10-23-claims-stream/`, run again with the packages from PyPI):

| | reference `2d25be6` | `6090f4f` |
|---|---|---|
| wall | 337.3–359.5 s | 134.2–156.9 s |
| CPU, chunks (process and partition) | 686–701 s | 397–429 s |
| CPU, partition | 299–304 s | 74–84 s |
| CPU, group merges | 473–547 s | 29–32 s |
| peak RSS, all processes | 11.2–12.4 GB | 5.9–6.4 GB |
| peak RSS, one worker | 3.3 GB | 1.7–1.8 GB |
| device bytes written | 2,080–2,196 MB | 1,034–1,101 MB |

Medians: 337.7 s against 148.0 s, 2.28 times as fast. `6090f4f` holds two changes:

- **#25 (pq-glue):** group merges copy the partition files' row groups as they are,
  without decoding them. This removed most of the merges' CPU.
- **#26 (genson's Arrow stream):** the claims come from polars-genson as an Arrow C
  stream and are written once, one row per statement. Before, they were written, read
  back and written again. This removed most of partitioning's CPU and half the disk
  writes.

Any per-change split of the wall time other than the journal entries' own A/Bs is an
estimate, not a measurement. The first CI run gives the record on CI's hardware.
