# Run loop and state

`process-wikidata` (`main.run`) takes every chunk of a set through processing,
partitioning and upload. `pool.py` runs the chunks in parallel, and `state.py` records each
chunk's progress.

## Steps of a chunk

A chunk's state is a number in `state/chunk_{N}.jsonl`, one file per chunk:

| Step | Value | Reached when |
|---|--:|---|
| `INIT` | 0 | the state is set up (`initial.setup_state`) |
| `PULL` | 1 | the source file is local and its size checked |
| `PROCESS` | 2 | the seven tables are written to `results/{table}/` |
| `PARTITION` | 3 | each table is split into `results/{table}/{key}/` |
| `PUSH` | 4 | its group is uploaded |
| `POST_CHECK` | 5 | its group is verified on the Hub |
| `COMPLETE` | 6 | its group's staging is cleaned up |

`setup_state` creates the files at `INIT` on the first run: one per chunk in the split
manifest for a release, one per file of the source repo otherwise. `get_chunk_state`
reads one chunk's file; `get_all_state` reads every file, which takes about 0.4 s over
13,000 files, so per-chunk code reads only its own.

## The loop

`run()`:

1. Sets up the state if `state/` does not exist.
2. Finishes a group left unfinished by an earlier run (`unfinished_group`), from the stage
   after its last recorded one.
3. Lists the chunks below `PARTITION` and hands them to `pool.process_chunks`, with the
   partitioned chunks not yet in a group.
4. Removes the partition directories left empty by merging.
5. Without a release, runs `finalise()`. A release's sets are finalised by the `release`
   recipe once both are processed.

`process_chunks` receives the steps as functions:

| Function | Runs in | Does |
|---|---|---|
| `start(c)` | the parent, before the chunk's process | the pull: `dump.check_chunk` for a release, `pull_chunk` and prefetch otherwise |
| `work(c)` | the chunk's own process | `process_and_partition`: [process](process.md), then [partition](partitioning.md) |
| `done(c)` | the parent, after success | appends the chunk's source and partition bytes to `partition_sizes.jsonl` |
| `ready(chunks)` | the parent | whether the closable chunks' partition bytes reach the group threshold |
| `close(chunks)` | a background thread | merge, upload and verify the group ([push](push.md)) |

## Worker processes

Each chunk runs in a fresh process, started with `spawn`. When the process exits, all the
memory it used goes back to the operating system. Native memory freed within one long-lived
process stays with its allocator, and its size grew chunk after chunk. `spawn` rather than
`fork` because the parent has threads (the upload thread, the prefetch thread).

`CHUNK_WORKERS` processes run at once. Each chunk's work is mostly single-threaded, with
parallel stretches in Polars and polars-genson, so one chunk at a time used about half of
a 20-core machine. See [Tuning](../guide/tuning.md#how-many-workers).

## Groups with out-of-order chunks

A group is a range of consecutive chunk numbers. Chunks finish out of order, so the chunks
that can form a group are the partitioned chunks below every chunk still running or queued
(`closable`). When their partition bytes reach the threshold, they close as one group in a
background thread, and the workers continue with the next chunks. One group closes at a
time. Once every chunk is done, the remaining partitioned chunks close as the last group.

## Failures and interruption

- A chunk's process exiting non-zero stops new chunks from starting. The running chunks
  finish, and then the run raises, naming each failed chunk and its exit code or signal.
- A failed upload raises when the parent next checks the upload thread. Running chunks
  finish first.
- On Ctrl-C, the chunk processes receive the signal too. The upload thread is a daemon
  thread and ends with the run; the group ledger resumes the group on the next run.
- An interrupted chunk is redone from its recorded step. Its partly written outputs were
  written to `.tmp` paths and are overwritten.

## Partitioning a chunk

`partition_chunk` checks that every table's processed file exists, prepares each table
(`prepare_for_partition`) and writes its partitions and audit sidecar, moves the chunk to
`PARTITION`, and deletes the processed files.

??? info "Documented against"
    Commit `6243a2c` (2026-10-04). See [About these docs](../about.md) to check for changes.

    | File | SHA-256 |
    |---|---|
    | `src/wikidata/main.py` | `578eeb24bd587fe45e004c8423ff02648c8235b58ddd35714351f9dd2e1ec03c` |
    | `src/wikidata/pool.py` | `fdb5bb426ceeef9ae4cb3dd1ce649969035a5f21f965b594152995d619bc257a` |
    | `src/wikidata/state.py` | `23f8bba8b8e9fdc1727ab26d5252b06cac0721a160b0a6cb680adecb958c6429` |
    | `src/wikidata/initial.py` | `82eaa1d525ad4a2a02c6c0e35c4581d4b761cf98028c903c3e427ba46e67ad64` |
