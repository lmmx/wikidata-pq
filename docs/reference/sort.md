# Sort by id

`sort_by_id.py` sorts each key's rows by id across all its files, so that a lookup by id
reads only the row groups whose id range can hold it. After compaction, a key's rows are
in source chunk order. That order runs through the id space many times, so every file and
most row groups span nearly every id.

## Order

Rows are sorted by `id` (`ref` for claims_labels, `SORT_COLUMN`) in **string order**:
`Q10` comes before `Q2`. Parquet's min/max statistics compare the column's bytes, so a
string order makes the statistics exact and lets each row group declare its order
truthfully in `sorting_columns`; a numeric order would not match the statistics. The sort
is stable, so each id's rows keep their order (a statement's order among the entity's
statements, for example). Files are named `part-{i}-of-{n}.parquet`.

## Stages

Recorded in `state/sort.jsonl`, after a table's compaction is `done`:

| Stage | Does |
|---|---|
| `sourced` | `hub/{table}` brought up to date with the Hub (it already holds the compacted files, so for a release only a file that differs is downloaded) and checked to have exactly the Hub's files, by size and sha256; the empty compaction source directory removed |
| `written` | each key sorted into `compact/sort/out/{table}/{key}/`, checked, and listed in the manifest |
| `committed` | each key's part files added and its old files deleted, in batched commits |
| `verified` | the Hub has exactly the part files of every key |
| `done` | the card metadata JSON rewritten; the sorted files moved into `hub/{table}`, so the local copy matches the Hub |

## Keys that fit in memory

A key under `SORT_IN_MEMORY_BYTES` (2 GiB of Parquet; every key but claims' `all`) is read
whole, sorted with pyarrow's stable sort, and cut into `ceil(bytes / COMPACT_FILE_BYTES)`
files of equal rows (`sort_in_memory`). The files are read with pyarrow one at a time and
concatenated, not as one dataset, which would cast them to a unified schema; casting
corrupts nested claims structs.

## Keys that do not: buckets

Claims are about 17 GB of Parquet (45 GB in release 20260928) and several times that in
memory, so they go through id-range buckets, each step resumable and run by
`SORT_WORKERS` processes at once, the parent recording each job as it finishes:

1. **Boundaries** (`_boundaries`). The ids' counts give `n` buckets of about equal rows
   (`n` from `SORT_BUCKET_BYTES`, 64 MiB of source per bucket), never splitting an id.
2. **Bucketing** (`bucket_key`). Each source file is a job: its rows go to its own
   fragment of every bucket (`bucket-{i}-{j}`), in source order, each bucket's rows
   buffered to about `SORT_BUCKET_WRITE_BYTES` of Arrow memory before a row group is
   written (a piece per source row group would give each bucket thousands of tiny row
   groups). The job also takes the file's sums (below). `bounds.json` holds the
   boundaries, `bucketed.jsonl` each file done (reused on a restart), and `buckets.json`
   the result.
3. **Sorting buckets** (`sort_buckets`). Each bucket's fragments are read in source order,
   sorted in memory, written as a scratch file (zstd 1, no content-defined chunking or
   page index: packing reads it once), checked, listed in `sorted.jsonl` with the
   bucket's sums, and the fragments deleted.
4. **Packing** (`pack_key`). Consecutive sorted buckets are joined into part files of about
   `COMPACT_FILE_BYTES` (the count from the source files' size, as the scratch files are
   larger). Batches are re-cut across bucket boundaries so all row groups have the same
   size. Each file is listed in `files.jsonl` and reused on a restart if its size and
   sha256 match.

## Checks

- **Sorted** (`_check_sorted`): each file's sort column ascends, and each file starts at
  or after the previous file's end.
- **Same rows, same order within each id** (`_ranked`): the row count and two hash sums,
  each row hashed with its rank among its id's rows. The sums are the same for the same
  rows with each id's rows in the same order, whatever the order of the ids.
- **Packing** is checked by the positional fingerprint of compaction, since packing must
  not reorder anything.
- **Bucketing** (`_check_whole`): the row count and, for two seeds, the sums of the low
  and high 32 bits of each row's hash (`_additive`). The sums are exact integers, so the
  parts of any split of the rows add up to the whole's: the sources' sums, taken file by
  file as they are bucketed, must equal the buckets' sums, taken as they are sorted. With
  the other checks this ties the part files to the sources without reading either again.
  Buckets made before 2026-10-05 have no sums; their key is checked by reading the
  sources and part files again, a file at a time with a progress bar.
- Every file has the key's schema.

??? info "Documented against"
    Commit `6243a2c` (2026-10-04). See [About these docs](../about.md) to check for changes.

    | File | SHA-256 |
    |---|---|
    | `src/wikidata/sort_by_id.py` | `755a6246b4a6141ec141bd0942b917eb5fb7eb84ca2b3b6bb647d2ae2ee1a6c7` |
