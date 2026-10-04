# Wikidata Processing Pipeline Design

## Overview

The pipeline processes the 7,449 Parquet files (959 GB in all, one per chunk) of the
`philippesaade/wikidata` dataset, transforming nested JSON columns into 6 datasets (labels,
descriptions, aliases, links, claims, claims_labels), each split by language (links by site;
claims not split), totalling 35.5 GB of Parquet on the Hub.

`process-wikidata` runs steps 0-5 chunk by chunk and then `finalise-wikidata`, which compacts,
sorts and documents each dataset once every chunk is uploaded (step 6).

## State Management

It uses a file-based state system where each source file has its own tracking record,
allowing fine-grained resume capability and progress monitoring across the 7,449 files.

- **File-level tracking**: One state file per source file (`chunk_0.jsonl`)
- **Step enumeration**: `INIT(0) → PULL(1) → PROCESS(2) → PARTITION(3) → PUSH(4) → POST_CHECK(5) → COMPLETE(6)`
- **Chunk-based processing**: One file per chunk index (0-7448), processed sequentially by chunk
- **State queries**: `get_next_chunk()` returns lowest chunk with `INIT` files, enabling resumable processing
- State files use `.jsonl` extension with stem matching source files
- A regex (`CHUNK_RE`) extracts the chunk number, and files are processed in chunk order

## Pipeline Steps

### 0. Initialize State

This one-time setup phase discovers all files in the remote dataset and creates initial state tracking files for each one.
After this, files are only processed chunk by chunk, by the chunk number in the filename.

- **Trigger**: `state/` directory doesn't exist
- **Action**: List the source repo's `data/*.parquet` (7,449 files), create state tracking at `Step.INIT`
- **Module**: `initial.setup_state()`

### 1. Pull

The download phase pulls a chunk's file from the Hugging Face Hub.

It acts as a gate: if you've already uploaded a corresponding processed file for a given source file,
we don't download and reprocess that file again.

- **Input**: Files at `Step.INIT` for current chunk
- **Action**: Download the source file with `snapshot_download`, checked against the repo's size
- **Output**: Local file `data/chunk_N.parquet`
- **State update**: `Step.PULL`
- **Optimization**: Skip a file already downloaded at the right size; whether a chunk is uploaded is
  known from local state, not by asking the Hub
- Uses `snapshot_download` with `allow_patterns` listing the files still needed
- Avoids repeated HF API calls by using cached state inventory

### 2. Process

The transformation phase extracts five distinct tables from the nested JSON columns,
flattening nested schemas and ensuring they have scalar data types.

We validate that entity IDs are all preserved (except for aliases, we allow dropping null rows there).

- **Input**: Files at `Step.PULL`
- **Action**: Extract 5 tables (labels, descriptions, aliases, links, claims) from nested JSON
- **Output**: Processed parquet files in `results/{table_type}/chunk_N.parquet`
- **Module**: `process.process_single_file()` (extracted from current batch processor) (**!!TODO!!**)
- **State update**: `Step.PROCESS`
- Extracts 5 specific tables from nested JSON: labels, descriptions, aliases, links, claims
- Claims use temporary batching system due to memory constraints
- ID preservation validated between input/output for each table
- Claims are conformed to the stored claims schema (fields a chunk lacks are null), so every
  chunk, group and uploaded file has the same schema
- Intermediate batch files cleaned up after processing

### 3. Partition

The partitioning phase splits each of the six processed tables by language (labels,
descriptions, aliases, links by site, claims, and `claims_labels`: the label maps extracted
from claims as `field, ref, language, label` rows, a dataset of its own so users of one
language download only that language's labels),
creating subdirectories named according to the language (the partition key) e.g. `en`.

It generates 'audit sidecars' storing row counts and min/max IDs for each subset.
The 'sidecar file' contains metadata from the partitioning (what got put into which subsets) for auditing in Step 5.

- **Input**: Files at `Step.PROCESS`
- **Action**: Split each table by language column into subdirectories
- **Output**: `results/{table_type}/{lang}/chunk_N.parquet`
- **Sidecar**: Audit files tracking row counts per language per source file
- **Module**: `partitioning.partition_parquet()`
- **State update**: `Step.PARTITION`
- Custom file path naming preserves source filename in partitioned output
- Callback mechanism automatically triggers sidecar writing during partitioning
- Languages with 0 rows naturally omitted from sidecar files
- Claims are not split by language: a claim has no language of its own, and its property,
  value and unit labels are in claims_labels, which is. Every claims row gets the same
  constant partition key (`UNSPLIT_KEY`, "all"), so claims go through the same merge and
  upload path as one file per group. Users join claims to claims_labels (and to labels,
  for the entity) in the languages they want.

### 4. Push (grouped)

Chunks are processed several at a time (`CHUNK_WORKERS`, default 3, or `WIKIDATA_WORKERS`;
each in its own spawned process, see `pool.py`) but uploaded in **groups**: a contiguous
range of chunks whose language subsets are merged into one file per language before upload.
Uploading one file per language per chunk would put ~2,800 files per chunk on the Hub
(~20M in total), far past the Hub's recommended <100k files per repo, while a group of
many chunks gives one file per language per group.

Each chunk joins the open group once partitioned. Chunks finish out of order, so the open
group is the partitioned chunks below every chunk still running or queued. The group is
closed when its buffered partition files reach the group size threshold, or when no chunks
are left to partition. One group closes at a time, in a background thread, while the
chunks after it are processed.

**Adaptive group size.** The threshold is set so that the whole dataset comes to about
`GROUP_TARGET_COUNT` groups (default 30), keeping each repo under ~100k files with up
to ~1,000 language or site folders:

- `projected_bytes = (partition bytes so far / source bytes so far) x total source bytes`
- `threshold = clamp(projected_bytes / GROUP_TARGET_COUNT, GROUP_MIN_GB, GROUP_MAX_GB)`

Source bytes are the yardstick because source files vary 50x in size (median 104 MB,
largest 1.14 GB): a group is a fixed share of the data, not a fixed number of chunks.
`GROUP_MAX_GB` bounds local disk; if it binds, there are more groups (more files) than
targeted.

**Closing a group** (each stage is recorded in the group ledger, so a crash resumes there):

1. **Merge**: for each table and language, the chunk files are concatenated (streaming)
   into `staging/{table}/{lang}/chunks-{first:04d}-{last:04d}.parquet`. The merged row
   count must equal the sum of the chunks' audit sidecar counts; the chunk files for that
   language are then deleted. `claims_labels` rows are deduplicated within the group.
2. **Upload**: each table's staging dir goes to `{HF_USER}/wikidata-{table}` with
   `upload_large_folder` (resumable, splits into commits itself). Repos are created if
   missing.
3. **Verify**: every staged file's size and sha256 are compared to the Hub's record of the
   uploaded file.
4. **Clean up**: the staging dir is deleted and the group's chunks are marked `COMPLETE`.

- **Input**: Chunks at `Step.PARTITION`
- **Remote layout**: `{lang}/chunks-{first:04d}-{last:04d}.parquet` in each table repo
  (one file per group per language folder, well under the Hub's 10k per folder), replaced
  by compaction and the sort in step 6
- **Ledger**: `state/groups.jsonl`, one line per group stage (`merged`, `pushed`, `verified`)
- **State update**: `Step.PUSH` when the group is uploaded, `Step.POST_CHECK` when verified,
  `Step.COMPLETE` after clean up

### 5. Post-check

Row counts are checked locally at merge time against the audit sidecars, and the upload is
checked byte-for-byte (sha256) against the staged file, so the uploaded data is verified
without reading it back from the Hub.

### 6. Finalise

Once every chunk is complete, `finalise-wikidata` runs three stages on each table. Each has
its own ledger in `state/` (`compact.jsonl`, `sort.jsonl`) and resumes where it stopped; a
table already done is skipped.

**Compaction** (`compact.py`). The grouped upload leaves one file per key per group, most of
them small. Each key's group files are downloaded to `compact/src/{table}`, rewritten into
files of about `COMPACT_FILE_BYTES` (500 MiB, the Hub's guidance), split only between groups,
with row groups of `COMPACT_ROW_GROUP_BYTES` (128 MiB of Arrow memory), ZSTD, a page index and
content-defined chunking. Stages: downloaded, written (each file checked against its group
files by an ordered row fingerprint), committed (a key's new files added and its group files
deleted in one commit, keys batched), verified (the Hub has exactly the new files, by size and
sha256), done (files, bytes and rows per key in `docs/dataset_cards_metadata.json`).

**Sort by id** (`sort_by_id.py`). Compacted rows are in source chunk order, which runs through
the id space many times, so no row group could be skipped on an id lookup. Each key's rows are
sorted by `id` (`ref` for claims_labels) in string order, stably, across all its files, and
each row group declares the order in `sorting_columns`; files are named
`part-{i}-of-{n}.parquet`. The stage reads the local copy (`hub/`, from `download-wikidata`):

- sourced: `hub/{table}` has exactly the Hub's files, by size and sha256; the emptied
  `compact/src/{table}` is then removed.
- written: a key under `SORT_IN_MEMORY_BYTES` (every key but claims/all) is read whole,
  sorted, and split into `ceil(bytes / COMPACT_FILE_BYTES)` files of equal rows, checked by
  row hash sums with each row's rank within its id (same rows, same order within an id) and
  sorted within and across files. claims/all (about 17 GB of Parquet, several times that in
  memory) is range-partitioned: bucket boundaries of equal row counts from the id column,
  one streaming pass writing each row to its bucket in source order, each bucket sorted in
  memory and checked, and consecutive sorted buckets packed into files of about
  `COMPACT_FILE_BYTES`, each pass resumable.
- committed, verified: as for compaction.
- done: the metadata JSON rewritten, and `hub/{table}` holds the sorted files, so the local
  copy matches the Hub.

**Dataset cards** (`card_stats.py`, `cards.py`). Each repo's README.md is rendered from a
template in `docs/dataset_cards/{table}.md`, whose placeholders are filled from the metadata
JSON and `docs/dataset_cards_stats.json`: one subset (config) per key plus `all`, the default
subset, the size of every key, sample rows, and the coverage of `en` and `mul`. The stats are
computed from `hub/` (checked against the metadata) with a digest of their inputs, and a card
does not render from stats whose digest differs from the current one; `finalise` recomputes
stale stats first. The rendered cards are written to `docs/dataset_cards/rendered` and pushed
only where they differ from the Hub's.

### Local disk and clean up

Everything deleted can be regenerated from the source repo; only the Hub uploads are product.

| Files | Deleted when |
|---|---|
| Source `data/.../chunk_N.parquet` | chunk reaches `PROCESS` |
| Processed `results/{table}/chunk_N.parquet` | chunk reaches `PARTITION` |
| Partitions `results/{table}/{lang}/chunk_N.parquet` | merged into staging |
| Staging `staging/{table}/...` | group verified |
| Audit sidecars `audit/...` | kept (small) |
| Compaction group files `compact/src/{table}` | table compacted; the directory once the local copy is checked |
| Compacted and sorted files `compact/out`, `compact/sort/{out,buckets}` | committed and verified (their `.jsonl` bookkeeping is kept) |
| Local copy `hub/{table}` | never: the sort replaces its files with the sorted ones |

Peak local disk is about: prefetched sources (`PREFETCH_BUDGET_GB`) + one group's
partitions (`GROUP_MAX_GB`) + one group's staging (about the same size, as partitions are
deleted language by language as they are merged).

## Orchestration

```python
def run():
    if not state_dir.exists():
        setup_state(state_dir)
    finish_closed_group()  # resume a group interrupted mid-merge/upload/verify
    while (chunk_idx := get_next_chunk(state_dir, below=Step.PARTITION)) is not None:
        pull_chunk(chunk_idx)       # prefetch runs ahead in the background
        process(chunk_idx)          # then delete the source file
        partition(chunk_idx)        # then delete the processed files
        if open_group_bytes() >= group_threshold():
            close_group()           # merge, upload, verify, clean up
    close_group()                   # the remainder
    finalise()


def finalise():
    for table in Table:
        compact_table(table)        # group files -> ~500 MB files, per key
    for table in Table:
        sort_table(table)           # rows sorted by id across each key's files
    update_stats()                  # card figures, from hub/, where stale
    for table, card in write_cards().items():
        push_card(table, card)      # only where it differs from the Hub's
```

- A chunk's progress is its state file; a group's progress is the ledger.
- Errors halt the run (e.g. a schema mismatch); re-running resumes from the ledger and states.

## Key Design Decisions

- **Chunk-level processing, group-level upload**: disk use is bounded by the prefetch budget and group size, and the Hub file count by the number of groups
- **File-level state**: Enables fine-grained resume capability
- **Single state per file**: Avoids complex multi-table state management
- **Sidecar auditing**: Enables reliable post-upload verification
- **Language partitioning**: Reduces download requirements for end users
- **Finalise after the run**: files sized for the Hub and sorted by id are written once from
  the complete data, rather than kept balanced during the run
- **String order for ids**: exact min/max pruning and a truthful `sorting_columns`, where
  numeric order would not match the byte order Parquet statistics use
- **Cards rendered from the data**: every figure in a card comes from the metadata and stats
  files, and a card cannot render from stale figures
