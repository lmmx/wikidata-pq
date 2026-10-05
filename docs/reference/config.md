# Configuration

`src/wikidata/config.py` holds every path, table name and tunable constant. The
environment variables are read once, when the module is imported, and the constants
derived from them (paths, repo names, the table list) are fixed for that process.

## Environment variables

| Variable | Effect |
|---|---|
| `WIKIDATA_RELEASE` | The release (dump date) to build. Unset: build from the philippesaade copy. |
| `WIKIDATA_SCHOLAR` | `1`: the scholarly set of the release. Requires `WIKIDATA_RELEASE`. |
| `WIKIDATA_WORKERS` | Chunks processed at once (`CHUNK_WORKERS`, default 6). |
| `WIKIDATA_SORT_WORKERS` | Buckets sorted, and part files packed, at once (`SORT_WORKERS`, default 6). |
| `WIKIDATA_DUMPS_URL` | A mirror for `download-dump` (dumps.wikimedia.org by default). |
| `WIKIDATA_PREVIOUS_RELEASE` | For `promote-release`: the tag for `main`'s current files. |

The Justfile recipes set these from their arguments.

## Working directories

`WORK_DIR` is `releases/{release}` for the main set, `releases/{release}-scholar` for the
scholarly set, and `.` without a release. `OTHER_WORK_DIR` is the other set of the same
release; `claims_labels` reads its labels. All paths are relative to the current
directory, so the commands run from the repository root.

| Constant | Path under `WORK_DIR` | Holds |
|---|---|---|
| `DUMP_DIR` | `dump/` | the bz2 dump |
| `ROOT_DATA_DIR` | `data/` | source chunks and their manifest |
| `STATE_DIR` | `state/` | per-chunk state and every ledger |
| `OUTPUT_DIR` | `results/` | processed tables and their partitions |
| `AUDIT_DIR` | `audit/` | partition sidecars (kept) |
| `QUARANTINE_DIR` | `quarantine/` | snaks pruned from claims (kept, never uploaded) |
| `STAGING_DIR` | `staging/` | one group's merged files before upload |
| `COMPACT_DIR` | `compact/` | compaction's sources and outputs |
| `SORT_DIR` | `compact/sort/` | the sort's buckets and outputs |
| `HUB_COPY_DIR` | `hub/` | the local copy of the repos, for the sort and the card figures |

Card figures and rendered cards go under `docs/releases/{set}/` for a release
(`docs/` and `docs/dataset_cards/rendered/` without one). Card templates are in
`docs/dataset_cards/dump/` (main set), `docs/dataset_cards/scholar/` (scholarly set) or
`docs/dataset_cards/` (the philippesaade build).

## Tables

`Table` is a `StrEnum` of the table names: `labels`, `descriptions`, `aliases`, `links`,
`claims`, `claims_labels`, and `entities` for a release only. The enum is built at import,
so `entities` exists only when `WIKIDATA_RELEASE` is set, and loops over `Table` cover the
right tables in each mode.

`PARTITION_COLS` gives each table's partition column: `language` for labels, descriptions,
aliases and claims_labels, `site` for links, and `UNSPLIT_COL` for claims and entities.
Rows of an unsplit table get `UNSPLIT_COL` (`partition`) set to `UNSPLIT_KEY` (`all`), so
they take the same partition, merge and upload path as the split tables, into one folder
`all/`. The column is not written to the files.

## Repos and the Hub

- `HF_USER` (`permutans`) owns the repos. `REPO_TARGET` is `{hf_user}/wikidata-{tbl}`, or
  `{hf_user}/wikidata-scholar-{tbl}` for the scholarly set (`REPO_PREFIX`).
- `HUB_REVISION` is `build-{release}` for a release, the branch everything is uploaded to
  until promotion, and `None` (`main`) without one. Every call to the table repos passes
  it.
- `HF_REPO_PRIVATE` (False) applies when a repo is created.

## Constants

| Constant | Default | Used by |
|---|---|---|
| `CHUNK_WORKERS` | 6 | [run loop](run.md) |
| `GROUP_TARGET_COUNT` | 30 | [push](push.md#group-size) |
| `GROUP_MIN_GB`, `GROUP_MAX_GB` | 1, 25 | [push](push.md#group-size) |
| `CLEAN_UP_LOCAL` | True | every stage that deletes files |
| `COMPACT_FILE_BYTES` | 500 MiB | [compaction](compact.md), [sort](sort.md) |
| `COMPACT_ROW_GROUP_BYTES` | 128 MiB | [compaction](compact.md), [sort](sort.md) |
| `COMPACT_COMMIT_MAX_ADDS`, `COMPACT_COMMIT_MAX_OPS` | 50, 2000 | Hub commits in compaction and sort |
| `COMPACT_DOWNLOAD_WORKERS` | 32 | Hub downloads |
| `SORT_IN_MEMORY_BYTES`, `SORT_BUCKET_BYTES` | 2 GiB, 64 MiB | [sort](sort.md) |
| `PREFETCH_*` | budget 60 GB, 60 chunks ahead, 100 GB free, 1 at a time | [pull](pull.md) (philippesaade build only) |
| `CHUNK_RE` | `chunk_(\d+)\.` | chunk number from a file name |

??? info "Documented against"
    Commit `6243a2c` (2026-10-04). See [About these docs](../about.md) to check for changes.

    | File | SHA-256 |
    |---|---|
    | `src/wikidata/config.py` | `99b048d1fd90ac8475cf1443d97a3e33f18c58c228630cf85467a85981f3503a` |
