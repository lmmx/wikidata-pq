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
- Intermediate batch files cleaned up after processing

### 3. Partition

The partitioning phase splits each of the five processed tables by language,
creating subdirectories named according to the language (the partition key) e.g. `en`.

It generates 'audit sidecars' storing row counts and min/max IDs for each subset.
The 'sidecar file' contains metadata from the partitioning (what got put into which subsets) for auditing in Step 5.

- **Input**: Files at `Step.PROCESS`
- **Action**: Split each table by language column into subdirectories
- **Output**: `results/{table_type}/language={lang}/chunk_N-XXXXX-of-XXXXX.parquet`
- **Sidecar**: Audit files tracking row counts per language per source file
- **Module**: `partitioning.partition_parquet()`
- **State update**: `Step.PARTITION`
- Custom file path naming preserves source filename in partitioned output
- Callback mechanism automatically triggers sidecar writing during partitioning
- Languages with 0 rows naturally omitted from sidecar files

### 4. Push

The upload phase uses HuggingFace CLI's large folder upload to transfer all the language-partitioned tables
once an entire chunk of files is complete. The tables get uploaded to subdirectories named under these language subsets.

- **Input**: Files at `Step.PARTITION` (entire chunk completed)
- **Action**: Upload language partitions using HF CLI `upload-large-folder`
- **Target**: 5 separate HF datasets, each with language-based configs
- **Module**: HF CLI integration
- **State update**: `Step.PUSH`
- "Upload large folder" chosen to avoid API rate limits vs individual file uploads
- Targets 5 separate HF datasets with language-based configs

### 5. Post-check

The audit phase verifies that the uploaded files match the local files' row counts,
and if not, checks whether they have the min and max IDs.
There's not much we can do otherwise except regenerate and put it back because we want a continuous pipeline rather than exiting.

Presuming everything went well, we clean up the local files and set the state to complete,
allowing the pipeline to process the next chunk once all files in a chunk are completed.
Once there are no more chunks left, this post-check is the end of the pipeline.

- **Input**: Files at `Step.PUSH`
- **Action**: Verify uploaded file row counts match sidecar audit records
- **Method**: Polars `scan_parquet()` with predicate pushdown via `hf://` paths
- **State update**: `Step.POST_CHECK` → `Step.COMPLETE`
- Uses Polars `scan_parquet()` with `hf://` paths for remote row counting
- Includes min/max ID validation against sidecar records
- Verification failure handling currently unspecified
- Source files deleted locally after successful verification for clean handoff

## Orchestration

The main pipeline orchestrates all steps through a chunk-based iteration system that processes files in manageable groups,
enabling resumable execution and controlled resource usage.

```python
def run():
    # 0. Initialize state if needed
    if not state_dir.exists():
        setup_state(state_dir)
    
    # Process chunks sequentially
    while (chunk_idx := get_next_chunk(state_dir)) is not None:
        # 1-5. Execute pipeline steps for chunk
        process_chunk(chunk_idx, state_dir)
```

- Error handling and resumption strategy partially unspecified
- "Get next chunk" processes file groups sequentially to limit concurrent downloads
- Estimated 2-4 worker parallelization based on CPU utilization observations
- Disk space managed by processing one chunk before advancing to next

## Key Design Decisions

- **Chunk-level processing**: Fits within ~100GB disk constraints (largest chunk: 94GB, expected size after processing: ~5GB)
- **File-level state**: Enables fine-grained resume capability
- **Single state per file**: Avoids complex multi-table state management
- **Sidecar auditing**: Enables reliable post-upload verification
- **Language partitioning**: Reduces download requirements for end users
