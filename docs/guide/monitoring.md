# Monitor and resume

Every step writes its progress to files under the set's directory, so progress can be read
from disk while the run goes on, and a run that stops continues from the same files when
the same command is run again.

## Progress and time left

```sh
python scripts/release_eta.py 20260928       # rate over the last 30 minutes
python scripts/release_eta.py 20260928 60    # rate over the last 60 minutes
```

For each set, it prints the chunks processed and their source GB, then the rate in GB of
source per hour over the chunks finished in the window, and the hours left at that rate
for both sets' processing. A chunk counts as processed when its claims audit file
(`audit/claims/chunk_{N}.parquet`) is written. The GB rate depends on the mix of chunk
sizes in the window, and a window that contains a group upload or a restart reads lower,
so the estimate varies by tens of percent from one window to the next.

## Files that record progress

Within `releases/{release}/` or `releases/{release}-scholar/`:

| File | Records |
|---|---|
| `data/manifest.jsonl` | each chunk's rows, bytes and id range |
| `data/chunk_{N}.parquet` | chunks not yet processed (each is deleted once processed) |
| `state/chunk_{N}.jsonl` | the chunk's step: 0 init, 1 pulled, 2 processed, 3 partitioned, 4 pushed, 5 verified, 6 complete |
| `state/partition_sizes.jsonl` | a line per partitioned chunk: source bytes and partition bytes |
| `state/groups.jsonl` | a line per group stage: `closed`, `merged`, `pushed`, `verified`, `done` |
| `audit/{table}/chunk_{N}.parquet` | the chunk's partition files with row counts and id ranges |
| `quarantine/chunk_{N}.parquet` | snaks on deleted properties, removed from that chunk's claims |
| `state/compact.jsonl`, `state/sort.jsonl` | each table's finalise stages |
| `state/claims_labels_build.jsonl` | `refs`, `written`, `uploaded` |
| `state/finalise.done` | the set is finalised; promotion requires it |

Some useful checks:

```sh
tail -3 releases/20260928-scholar/state/groups.jsonl          # the last group's stages
tail -1 releases/20260928-scholar/state/partition_sizes.jsonl # the last chunk partitioned
ls -t --full-time releases/20260928-scholar/audit/claims | head -3
ls releases/20260928-scholar/data | wc -l                     # chunks left (plus manifests)
```

## When a run stops

A run halts on any error rather than skip data. The common causes:

- **A chunk's process fails.** No further chunks start, the chunks already running finish,
  any group upload in progress finishes, and the run exits with
  `Chunk N failed in its subprocess (...)`. The failed chunk's source file is still in
  `data/` (a chunk is deleted only once processed), so it can be read to reproduce the
  failure.
- **Schema mismatch.** A table's inferred schema has a field or type the stored schema does
  not (`Schema mismatch - update ...`). The data has something new, or polars-genson
  inferred it differently. See [Processing](../reference/process.md#schema-checks).
- **Fields the pipeline would drop.** The split found entity or sitelink fields that the
  process step does not read. They need adding to `process.py` and to `EXPECTED_FIELDS` in
  `dump.py`.
- **ID loss.** A table has a different number of distinct ids than its source chunk.
- **Hub errors.** An upload or commit fails. Transient errors in downloads are retried for
  up to about three hours; other errors stop the run.

After a fix, run the same command again (`just release 20260928 20260507`). The run:

1. finishes a group that was interrupted, from the stage after its last recorded one;
2. processes every chunk not yet partitioned, redoing any chunk that was interrupted
   (its partial outputs are overwritten through `.tmp` files and atomic renames);
3. continues through the rest of the recipe, skipping steps already done.

Interrupting with Ctrl-C is safe at any point, including during a group upload.

There is no command to reset one chunk. A chunk that failed or was interrupted before
`PARTITION` is redone by the rerun. Once processed, its source chunk is deleted
(`CLEAN_UP_LOCAL`), so a chunk past `PROCESS` cannot be redone from its source.

!!! warning "Editing the code during a run"
    Each chunk's process imports the package from disk when it starts. An edit to the
    code while a run is going takes effect on the next chunk, and an edit that changes a
    function the running parent process calls (such as `process_and_partition`'s
    signature) fails that chunk and stops the run. Develop on another checkout or
    worktree while a run is in progress.

??? info "Documented against"
    Commit `6243a2c` (2026-10-04). See [About these docs](../about.md) to check for changes.

    | File | SHA-256 |
    |---|---|
    | `scripts/release_eta.py` | `159fe420cf298e35217c1526a62a6d4bd2b700d01d3b49a85acd79d6d6602a92` |
    | `src/wikidata/state.py` | `23f8bba8b8e9fdc1727ab26d5252b06cac0721a160b0a6cb680adecb958c6429` |
    | `src/wikidata/pool.py` | `fdb5bb426ceeef9ae4cb3dd1ce649969035a5f21f965b594152995d619bc257a` |
    | `src/wikidata/push/groups.py` | `29cc2b01bd164cf8dbd0182564bad31877feaf16dc9be825f0bae765c5712caa` |
