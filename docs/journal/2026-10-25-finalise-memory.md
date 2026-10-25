# 2026-10-25: A bucket's freed pages returned between stages

After #118 the bench's 3× run (bench 8.0.0) had a bucket worker's peak at 3.75 GB and
finalise's process tree at 12.3 GB on 4 workers. At a release's 1.17M rows a bucket,
modelled at about 4.7 KB a row, a worker would peak near 5 GB, and FINALISE_WORKERS=6
near 30 GB. The bench's full tier, 6 workers on 15.6 GB runners, lost 2–3 of its 6
runners to OOM in finalise. This change lowers a worker's peak.

## Where a bucket's memory went

One bucket of the bench's 3× claims key: 881,505 rows, 15 chunks, 1.02 GB of Arrow. RSS
was sampled every 5 ms in one process running `_sort_bucket`, and Arrow's live bytes
were read at each stage's end. Measured locally, one fresh process per run, so the
figures are indicative only.

| stage | peak RSS | Arrow live at its end |
|---|---|---|
| imports | 0.11 GB | 0 |
| `_read_bucket` | 1.70 GB | 1.02 GB |
| `_sorted` | 3.04 GB | 1.02 GB |
| `_write_file_parallel` | 3.55 GB | 1.02 GB |
| `_check_written` | 2.78 GB | 0 |

Arrow never held more than about 1.6 GB, but RSS reached 3.5 GB. Two causes:

- **Freed pages kept.** pyarrow's pool, mimalloc, kept the pages each stage freed. A row
  group is decoded on Arrow's threads and freed on the worker's main thread, and the main
  thread's allocations do not reuse those pages. So each stage started on top of the
  last one's freed pages. The write then added arrow-rs's encoders, a row group per
  thread in glibc's arenas, about 0.5 GB.
- **A column held through its take.** `_sorted` iterated its columns with `enumerate`.
  CPython's `enumerate` keeps its last result tuple, and with it the column it last gave,
  until it gives the next one. So each column's chunks were held through the column's
  take, not freed before it as the code meant. For `references`, 597 MB of the bucket's
  1.02 GB, Arrow held its chunks, its combined copy and its taken copy at once.

## The change

- **`_sorted`:** the columns are iterated by index, so a column's chunks are freed once
  combined.
- **`_release()`:** calls `pa.default_memory_pool().release_unused()`, which returns
  the pool's free pages to the OS. It is called after each column's chunks are freed,
  after the take, and after each row group `_check_written` compares. It is a runtime
  call to the same allocator, with no configuration change (the same call
  docs/journal/2026-10-22-claims-explode-views.md used as a probe). With nothing to
  return it costs about 1 µs.

The output is unchanged.

| stage | master | this change |
|---|---|---|
| `_read_bucket` | 1.70 GB | 1.73 GB |
| `_sorted` | 3.04 GB | 2.31-2.34 GB |
| `_write_file_parallel` | 3.55 GB | 1.97-2.00 GB |
| `_check_written` | 2.73-2.82 GB | 2.24-2.26 GB |
| **the bucket's peak** | **3.54-3.55 GB** | **2.31-2.34 GB** |

The peak now falls in `_sorted`, while it combines `references`: the other columns
(0.42 GB), `references`' chunks and its combined copy (0.6 GB each), and about 0.6 GB of
pages the read left on Arrow's threads, which `release_unused` from the main thread does
not return.

## Tried and not kept (locally, one bucket)

| variant | the bucket's peak | why not |
|---|---|---|
| the write on 1 thread (`max_workers=1`), nothing else | 3.21 GB | the write 2.5× as long; with the pages returned, the write is no longer the peak |
| the write on 2 threads | 3.40 GB | as above |
| glibc `malloc_trim(0)` after the write | check 2.29 → 1.82 GB | the check is not the peak, and the call is glibc's alone |
| `release_unused` after the read too | 3.00 GB (without the index fix) | the read's pages sit in Arrow's threads' heaps |
| the row groups read on one thread (`use_threads=False`) | 2.28 GB | no lower than threaded, and the read slower |

## Modelled for a release

- **Assumes:** a worker's peak is the import floor (0.11 GB) plus a multiple of its
  bucket's Arrow bytes, so it scales with the bucket's rows.
- **Per worker:** locally that multiple is (2.32 − 0.11) / 1.02, about 2.2. A release's
  bucket holds 1.17M rows, about 1.35 GB of Arrow, so a worker peaks near 3.1 GB (about
  2.6 KB a row), against about 4.7 before.
- **6 workers:** near 18 GB with the parent, against about 30 GB. So a 16 GB laptop
  still needs `WIKIDATA_FINALISE_WORKERS=4`, or a budget on the jobs admitted.
- **Measured against modelled:** the bench measures the 3× run at 4 workers. The release
  figures are modelled.

## On CI

(bench 8.0.0, 3 rounds; filled in from the PR's bench comment)
