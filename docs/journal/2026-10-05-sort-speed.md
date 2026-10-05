# 2026-10-05: Speeding up the bucketed sort (to do)

Continues docs/journal/2026-10-03-release-20260928-run.md. The sort (sort_by_id.py,
docs/journal/2026-09-30-sort-by-id.md) was written to be correct and resumable, not for
speed; this entry records what its first full-size run took and the changes that would cut
it, and which are made ("Done" below).

## Current State

### Main set claims/all, release 20260928

- 30 compacted files, 44.7 GB, 3,769 row groups; 667 buckets (`SORT_BUCKET_BYTES`, 64 MiB of
  source each); 86 part files (`COMPACT_FILE_BYTES`, 500 MiB).
- Hashing the local copy (`sourced`): 72 s.
- Bucketing (`bucket_key`, one process): 53 min 42 s, 1.17 row groups/s.
- Sorting the buckets (`sort_buckets`, one at a time): 2 h 2 min, 11.0 s a bucket.
- Packing (`pack_key`), one at a time: about 73 s a file, from the part files' mtimes and
  `files.jsonl` (part-08: written 16:09:33 to 16:10:18, about 45 s; checked and hashed to
  16:10:46, about 28 s). 86 files: about 1 h 45 min, mostly on one of the 20 cores.
- Packing with 6 workers (362c379, `WIKIDATA_SORT_WORKERS`), from file 11 on: 117 files/h by
  `release_eta.py`, against about 49 files/h one at a time: 2.4 times, not 6. Not yet
  explained: disk, or the workers' Polars and pyarrow thread pools (each sized for all 20
  cores) competing in the check.
- Bucket sorting has not yet run with workers: 362c379 came after claims' buckets were
  sorted. The first bucketed sort with them will be the next key over
  `SORT_IN_MEMORY_BYTES`.
- Disk at packing: `compact/sort/buckets` 42 GB (sorted buckets), `hub/` 43 GB.

### Keys that take the bucket path

- Any key over `SORT_IN_MEMORY_BYTES` (2 GiB of Parquet). Known: claims/all of both sets.
  Likely: entities/all of both sets (one row per entity, 75 and 46 million). Possible:
  the scholarly set's largest language keys (labels/en, descriptions/en) and
  claims_labels keys. In the philippesaade build only claims/all was over 1 GiB
  (docs/dataset_cards_metadata.json).

## Why it is slow

1. **Bucketing writes about 3,769 row groups into every bucket.** Each source row group
   is split by id range and each piece written with `writers[b].write_table(slice)`, one
   row group per call. Source files are in chunk order, which runs through the whole id
   range many times, so nearly every source row group sends a piece to nearly every
   bucket: about 2.5 million row-group writes of about 190 KB each (128 MiB of Arrow per
   source row group over 667 buckets), each with its column chunks, compression and
   footer entries. The cost comes twice: in bucketing, and when each bucket is read back
   to be sorted. (The row-group count follows from the code; its share of the time is an
   estimate, not measured: claims' bucket files were deleted as they were sorted.)
2. **Sorted buckets are encoded with the final settings, then encoded again.**
   `_sort_bucket` writes each sorted bucket with `_write_file` (zstd 3, content-defined
   chunking, page index), the settings for the Hub's files. Packing decodes those buckets
   and encodes every row again into the part files. The dearest encode runs twice over
   the whole key, and the first copy is read only once.
3. **Bucketing is one process.** The only step of the bucket path not run by workers.
4. **Per-job process start.** `_in_parallel` uses `max_tasks_per_child=1`: every bucket
   and part file starts a process that imports polars, pyarrow and huggingface_hub (1-2 s),
   a few minutes over 667 buckets.
5. **Thread oversubscription.** Each worker's Polars and pyarrow pools are sized for all
   cores; 6 workers contend in their parallel stretches (the checks).

## To do

In order of expected gain for the work:

- [x] **Buffer bucket writes** (1). Keep each bucket's pieces in memory and write a row
  group when they reach about 16 MB of Arrow: about 10 GB of buffers for 667 buckets, a
  few dozen row groups per bucket instead of about 3,769. Change local to `bucket_key`;
  file names and `buckets.json` unchanged, so a restart's resume is unaffected.
- [x] **Cheap settings for sorted buckets** (2, first form). Write sorted buckets with
  zstd 1, no content-defined chunking, no page index; packing still writes the part files
  with the final settings. Change local to `_sort_bucket`.
- [ ] **Sort within packing** (2, second form, instead of the above). Drop the sorted
  bucket files: each packing worker sorts its own buckets in memory, one at a time, and
  streams them into its part file, with the per-bucket ranked check kept. Plans files from
  the unsorted bucket sizes. Removes one write, read and check of the whole key, but
  changes the resume (`sorted.jsonl` goes), so not for a key already part-sorted.
- [x] **Parallel bucketing** (3). Split the source row groups between workers; each
  writes its own fragment of every bucket (`bucket-{i}-{w}`), and the sort reads a
  bucket's fragments together. Combine with buffered writes.
- [x] **Reuse worker processes** (4). `max_tasks_per_child` of about 20: most of the
  start cost back, memory still returned regularly.
- [ ] **Cap threads per worker** (5). For example `POLARS_MAX_THREADS` set for the
  workers to cores / `SORT_WORKERS`. Measure against packing's 2.4 times first.
- [ ] **Measure** each change on a sample of claims (a few source files) in a scratchpad
  venv, against the current code: bucketing, sorting and packing times, and the checks
  still passing.

- [x] **Make the final check free** (6). After packing, `pack_key` checks the key's 86 part
  files against its 30 source files with `_multiset`: every row of both, about 45 GB each,
  hashed again. It shows no progress, held about 45 GB of RAM with every core busy, and
  was still running 26 min after packing ended (16:56 to past 17:22 UTC). Every link but
  bucketing is already checked: each sorted bucket against its bucket (ranked), each part
  file against its sorted buckets (positional). `_multiset`'s row count and hash sums add
  up over any split of the rows, so the bucketing pass can compute the sources' sums row
  group by row group as it reads them (stored in `buckets.json`), and each per-bucket check
  the bucket's sums; the end check is then a comparison of sums already computed, with no
  rows read again. Until then, give the check a progress bar (per file).

### Done

- Bucketing: one job per source file, `SORT_WORKERS` at once, each writing its own
  fragment of every bucket (`bucket-{i}-{j}`, j the source index), buffered to
  `SORT_BUCKET_WRITE_BYTES` (4 MiB) of Arrow per bucket before a row group is written;
  `bounds.json` and `bucketed.jsonl` let a restart skip the files done. Claims, estimated
  (3,769 row groups of 128 MiB of Arrow over 30 files and 667 buckets): about 24 MB a
  fragment, so about 6 row groups a fragment and 180 a bucket, instead of about 3,769.
- Sorted buckets are scratch files (`_write_file(..., scratch=True)`); packing plans its
  file count from the source files' size, as the scratch files are larger.
- Workers take 8 jobs each before being replaced (bucketing: 1, as a job holds every
  bucket's buffer).
- End check (`_check_whole`): exact additive sums (`_additive`: row count, and the sums of
  the low and high 32 bits of two seeded row hashes) of each source file, taken by its
  bucketing job, against the sum of each bucket's, taken in the same read as its ranked
  check. No rows are read again; the order check over the part files has a progress bar.
  For buckets made before this, the fallback reads sources and part files again, a file
  at a time with a progress bar.
- `scripts/test_sort.py`: the bucketed sort on a sample of claims (rows from 8 files
  shuffled into 4, row groups of 2,000 rows, 3 workers): stable id order, bucketing
  resumed, a changed row in a bucket caught by the end check, the fallback. Written in the
  container, which can run none of it (no Python 3.13, no network): first run on the host.

Kept as they are: the checks' coverage. The per-bucket ranked check is the only one that catches a
changed order within an id; the final check over all files is the only one that covers
bucketing.

## Switching the live run

- Spawned sort workers import `wikidata.sort_by_id` from disk, so the checkout must not
  change while a bucketed sort is running (docs/journal/2026-10-03-release-20260928-run.md:
  the same for chunk processes). Write and test the changes outside the checkout.
- Main claims/all is not redone: it is bucketed and sorted, and packed by the time the
  changes are ready.
- Stop the run once the main set's claims sort is complete (`[sort] claims: complete`),
  and before the next bucketed sort begins (likely main entities, after labels,
  descriptions, aliases and links are compacted and sorted). Stop between tables (after a
  `complete` line) to lose nothing; within a table, the stage in progress resumes from
  its ledger. Then bring in the changes and rerun `just release 20260928 20260507`.
