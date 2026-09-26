# Wikidata Processing Pipeline Design

## Overview

The pipeline processes 9,687 parquet files (1.6TB total) from the `philippesaade/wikidata` dataset,
transforming nested JSON columns into 5 separate language-partitioned datasets totaling ~100GB.

## State Management

It uses a file-based state system where each source file has its own tracking record,
allowing fine-grained resume capability and progress monitoring across the 9,687 files.

- **File-level tracking**: One state file per source file (`chunk_0-00001-of-00546.jsonl`)
- **Step enumeration**: `INIT(0) → PULL(1) → PROCESS(2) → PARTITION(3) → PUSH(4) → POST_CHECK(5) → COMPLETE(6)`
- **Chunk-based processing**: Files grouped by chunk index (0-112), processed sequentially by chunk
- **State queries**: `get_next_chunk()` returns lowest chunk with `INIT` files, enabling resumable processing
- State files use `.jsonl` extension with stem matching source files
- Regex patterns extract chunk/part numbers for sorting
- Files sorted by chunk, then part for predictable processing order

## Pipeline Steps

### 0. Initialize State

This one-time setup phase discovers all files in the remote dataset and creates initial state tracking files for each one.
After this, files are only processed chunk by chunk based on the chunk prefix in the filename (e.g. `chunk0*`).

- **Trigger**: `state/` directory doesn't exist
- **Action**: Query HF repo for all 9,687 files, create state tracking at `Step.INIT`
- **Module**: `initialise.setup_state()`

### 1. Pull

The download phase pulls files from the HuggingFace Hub within a given chunk using the chunk prefix.

It acts as a gate: if you've already uploaded a corresponding processed file for a given source file,
we don't download and reprocess that file again.

- **Input**: Files at `Step.INIT` for current chunk
- **Action**: Download source parquet files using HF CLI with acceleration
- **Output**: Local files in `data/chunk_N-XXXXX-of-XXXXX.parquet`
- **State update**: `Step.PULL`
- **Optimization**: Skip if file already processed locally or exists in target datasets
- Uses `hf download` with `--include` patterns for chunk-specific file selection
- Avoids repeated HF API calls by using cached state inventory

### 2. Process

The transformation phase extracts five distinct tables from the nested JSON columns,
flattening nested schemas and ensuring they have scalar data types.

We validate that entity IDs are all preserved (except for aliases, we allow dropping null rows there).

- **Input**: Files at `Step.PULL`
- **Action**: Extract 5 tables (labels, descriptions, aliases, links, claims) from nested JSON
- **Output**: Processed parquet files in `results/{table_type}/chunk_N-XXXXX-of-XXXXX.parquet`
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

### 4. Push (grouped)

Chunks are processed one at a time but uploaded in **groups**: a contiguous range of chunks
whose language subsets are merged into one file per language before upload.
Uploading one file per language per chunk would put ~2,800 files per chunk on the Hub
(~20M in total), far past the Hub's recommended <100k files per repo, while a group of
many chunks gives one file per language per group.

Each chunk joins the open group once partitioned. The group is closed when its buffered
partition files reach the group size threshold, or when no chunks are left to partition.

**Adaptive group size.** The threshold is set so that the whole dataset comes to about
`GROUP_TARGET_COUNT` groups (default 100), keeping each repo under ~100k files with up
to ~800 language folders:

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
  (~100 files per language folder, well under the Hub's 10k per folder)
- **Ledger**: `state/groups.jsonl`, one line per group stage (`merged`, `pushed`, `verified`)
- **State update**: `Step.PUSH` when the group is uploaded, `Step.POST_CHECK` when verified,
  `Step.COMPLETE` after clean up

### 5. Post-check

Row counts are checked locally at merge time against the audit sidecars, and the upload is
checked byte-for-byte (sha256) against the staged file, so the uploaded data is verified
without reading it back from the Hub.

### Local disk and clean up

Everything deleted can be regenerated from the source repo; only the Hub uploads are product.

| Files | Deleted when |
|---|---|
| Source `data/.../chunk_N.parquet` | chunk reaches `PROCESS` |
| Processed `results/{table}/chunk_N.parquet` | chunk reaches `PARTITION` |
| Partitions `results/{table}/{lang}/chunk_N.parquet` | merged into staging |
| Staging `staging/{table}/...` | group verified |
| Audit sidecars `audit/...` | kept (small) |

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
```

- A chunk's progress is its state file; a group's progress is the ledger.
- Errors halt the run (e.g. a schema mismatch); re-running resumes from the ledger and states.

## Key Design Decisions

- **Chunk-level processing, group-level upload**: disk use is bounded by the prefetch budget and group size, and the Hub file count by the number of groups
- **File-level state**: Enables fine-grained resume capability
- **Single state per file**: Avoids complex multi-table state management
- **Sidecar auditing**: Enables reliable post-upload verification
- **Language partitioning**: Reduces download requirements for end users
