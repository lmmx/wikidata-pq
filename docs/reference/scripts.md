# Scripts

Tools in `scripts/` that sit outside the package. Run them from the repository root.

## For a release

| Script | Does |
|---|---|
| `release_eta.py RELEASE [MINUTES]` | each step of `just release` marked done, in progress or to do, the step in progress with its progress and time left at the rate of the last `MINUTES` (default 30); see [Monitor and resume](../guide/monitoring.md#progress-and-time-left) |
| `bench_workers.py RELEASE [--set] [--chunks] [--workers]` | throughput by number of workers on the set's own unprocessed chunks; see [Tuning](../guide/tuning.md#how-many-workers) |
| `test_pool.py` | tests of `pool.process_chunks` with stand-in work: out-of-order finishes, contiguous groups, uploads overlapping processing, a failed chunk, a failed upload |
| `test_sort.py [KEY_DIR] [ROWS]` | tests of the bucketed sort on a sample of claims shuffled across 4 files: stable id order, bucketing resumed, a row changed in a bucket caught by the end check, the end check's fallback |
| `test_finalise.py [KEY_DIR] [ROWS]` | tests of finalise's jobs on small local data: compaction per file and resumed, the sort's input checked against compaction's manifest, keys sorted as jobs, the commit replacing group files, claims_labels refs and languages per job and resumed |
| `hub_audit.py [RELEASE]` | every table of both sets on the Hub, from the Parquet footers only: branches and tags, keys and rows on `main` and the release's branch, against the local card metadata, the entities routed to each set, and 20260507 |
| `rebuild_entities.py build\|hub\|promote [RELEASE]` | the entities table of both sets made again from the dump, by the pipeline's own code, then compacted, sorted and promoted on its own; the repair for release 20260928 ([journal](../journal/2026-10-06-entities-dropped.md)) |
| `p31_survey.py` | which "instance of" classes the philippesaade copy left out, from a release's chunks; the source of `SCHOLARLY_CLASSES` |

`bench_workers.py`, `test_pool.py`, `test_sort.py` and `test_finalise.py` run the pipeline's code on copies or stand-ins in
scratch directories. They do not touch a set's state, its chunks, or the Hub.

## For the philippesaade build

| Script | Does |
|---|---|
| `eta.py` | progress and time left from the audit files |
| `profile_chunk.py N` | profiles one downloaded chunk's processing in a scratch directory |
| `calculate_chunk_sizes.py`, `source_size/`, `plot_file_sizes*.py`, `plot_language_hist*.py` | sizes of the source files and of the languages |
| `reset_run.py` | deletes every output, locally and on the Hub; written to be run by hand only |
| `check_transforms.py`, `test_partitioning.py`, `debug_claims*.py` | checks of the table transforms from the first version of the pipeline |

## Elsewhere in the repository

- `demos/`: example analyses on the published tables (an item's statements in one
  language, family trees, a country's divisions, timelines, property rankings). The
  project README lists them.
- `sae/`: a sparse autoencoder trained on the external identifiers each item has, with its
  own README.
- `space/`: a browser app for the autoencoder's features and item neighbours.
- `misc/`: debugging scripts and logs from earlier runs.

??? info "Documented against"
    Commit `6243a2c` (2026-10-04). See [About these docs](../about.md) to check for changes.

    | File | SHA-256 |
    |---|---|
    | `scripts/release_eta.py` | `159fe420cf298e35217c1526a62a6d4bd2b700d01d3b49a85acd79d6d6602a92` |
    | `scripts/bench_workers.py` | `76be82eec264b3351f42329187409f8e4d1de6cb2f2e92a8cfc2024b27ffaf00` |
    | `scripts/test_pool.py` | `aff3e88e3d9d37870e32319159ce9d2d9d4cc12a537c2725f84f9ca5064b9698` |
    | `scripts/p31_survey.py` | `11d4dcabace715b57cd03e84b59b57565a8748dda28c153be3f52ded781c7997` |
    | `scripts/eta.py` | `9d588f9d79badfcefec26b6fac58fa50ea384ec65ed5009ff9980c2e8a72fe41` |
    | `scripts/profile_chunk.py` | `4cde89eb670958ce44e7accbe4f7b1e9a662eea52e3ac777ce17a936073d74c1` |
    | `scripts/reset_run.py` | `7f415f4e19a9b2be8278fb991a07ea5f41ce470cf7982c46c129346117300a28` |
