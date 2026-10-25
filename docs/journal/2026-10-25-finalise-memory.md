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
  and after the take. It is a runtime call to the same allocator, with no configuration
  change (the same call docs/journal/2026-10-22-claims-explode-views.md used as a
  probe). With nothing to return it costs about 1 µs.
- **`_check_written`, a bucket's:** its row groups are read back on the worker's own
  thread (`use_threads=False`), so each one reuses the last one's pages. The in-memory
  path still reads them on Arrow's threads, and returns their pages after each row
  group. The check is otherwise unchanged: every row group is read by pyarrow and
  compared value for value.

The output is unchanged.

| stage | master | this change |
|---|---|---|
| `_read_bucket` | 1.70 GB | 1.71-1.73 GB |
| `_sorted` | 3.04 GB | 2.30-2.34 GB |
| `_write_file_parallel` | 3.55 GB | 1.97-2.00 GB |
| `_check_written` | 2.73-2.82 GB | 2.29 GB |
| **the bucket's peak** | **3.54-3.55 GB** | **2.30-2.34 GB** |

The peak now falls in `_sorted`, while it combines `references`: the other columns
(0.42 GB), `references`' chunks and its combined copy (0.6 GB each), and about 0.6 GB of
pages the read left on Arrow's threads, which `release_unused` from the main thread does
not return.

## The check's reads

The first version of the change returned the check's pages after each row group, still
read on Arrow's threads. On CI that check cost 10.3% ±5.7% more CPU (7.5 → 8.2 s a
byte): each row group faulted in fresh pages. Read on the worker's thread, the pages are
reused instead, with no release. The check is mostly one thread's work anyway (on one
bucket locally: 1.9 s of wall time for 2.2 s of user time). Locally, on one bucket:

| the check | user + sys | its peak |
|---|---|---|
| Arrow's threads, pages returned each row group | 2.7-3.7 s | 2.27 GB |
| this thread, nothing returned (kept) | 2.4 s | 2.31 GB |
| this thread, pages returned too | 2.9-5.9 s | 2.28-2.30 GB |

## Tried and not kept (locally, one bucket)

| variant | the bucket's peak | why not |
|---|---|---|
| the write on 1 thread (`max_workers=1`), nothing else | 3.21 GB | the write 2.5× as long; with the pages returned, the write is no longer the peak |
| the write on 2 threads | 3.40 GB | as above |
| glibc `malloc_trim(0)` after the write | check 2.29 → 1.82 GB | the check is not the peak, and the call is glibc's alone |
| `release_unused` after the read too | 3.00 GB (without the index fix) | the read's pages sit in Arrow's threads' heaps |
| the bucket's row groups read on one thread (`use_threads=False`) | 2.28 GB | no lower than threaded, and the read slower |

## On CI

Bench 8.0.0, 3 rounds, the first version (`7e41dcd`, the check's pages returned each
row group); parity passed.

| | before | after | change |
|---|---|---|---|
| peak RSS, finalise: all processes | 5.495 GB | 3.864 GB | −29.7% ±3.1% |
| peak RSS, finalise: one worker | 2.506 GB | 1.724 GB | −31.2% ±2.6% |
| peak RSS, finalise 3×: all processes | 12.470 GB | 8.650 GB | −30.6% ±3.5% |
| peak RSS, finalise 3×: one worker | 3.739 GB | 2.545 GB | −31.7% ±1.2% |
| `_sorted`, finalise per byte | 5.7 s | 4.1 s | −27.9% ±7.2% |
| `_check_written`, finalise per byte | 7.5 s | 8.2 s | +10.3% ±5.7% |
| `_write_file_parallel`, finalise per byte | 14.8 s | 14.7 s | −0.7% ±0.7% |
| finalise, per byte | 43.6 s | 42.7 s | |
| ETA: finalise | 6h29m05s | 6h34m33s | no change beyond noise |
| ETA, a release | 16h01m33s | 16h15m27s | no change beyond noise |
| CPU, finalise | 1m28s | 1m29s | |

The kept version (`820bb45`, the bucket's check read on its worker's thread) was benched
on another set of runners, against the same `before`; parity passed.

| | before | after | change |
|---|---|---|---|
| peak RSS, finalise: all processes | 5.411 GB | 4.112 GB | −23.8% ±7.2% |
| peak RSS, finalise: one worker | 2.510 GB | 1.816 GB | −27.6% ±4.0% |
| peak RSS, finalise 3×: all processes | 12.180 GB | 9.232 GB | −24.1% ±4.8% |
| peak RSS, finalise 3×: one worker | 3.771 GB | 2.606 GB | −30.6% ±5.3% |
| `_check_written`, finalise per byte | 7.2 s | 7.5 s | no change beyond noise |
| `_sorted`, finalise per byte | 4.9 s | 3.9 s | no change beyond noise |
| `_write_file_parallel`, finalise per byte | 13.6 s | 13.5 s | |
| finalise, per byte | 40.0 s | 39.5 s | |
| ETA: finalise | 6h03m13s | 6h05m30s | no change beyond noise |
| ETA, a release | 15h19m00s | 15h13m48s | no change beyond noise |
| CPU, finalise | 1m25s | 1m25s | |

- **The check's CPU:** the kept version removes the first version's regression.
- **The trade-off:** the kept version's peaks sit a little above the first version's,
  about 0.06 GB a worker and 0.6 GB for the 3× tree, measured on other runners. On the
  first run's smaller buckets (587,556 rows), the single-threaded check is now about
  their peak. Either version cuts every peak beyond noise. The first one costs about 2%
  of finalise's CPU a byte for its extra 0.6 GB.

## Modelled for a release

- **Assumes:** a worker's peak scales with its bucket's rows from the bench's 3× run
  (881,334 rows a bucket, one worker 2.55-2.61 GB), less a floor of 0.11 GB for the
  imports. The tree's peak at 6 workers keeps the 3× run's ratio of the tree to 4 times
  one worker (85-89%).
- **Per worker:** a release's bucket holds 1.17M rows, so a worker peaks near 3.4 GB
  (2.9 KB a row), against near 5.0 GB before.
- **6 workers:** the tree peaks near 18 GB, against near 25 GB before. 6 workers at
  their peaks at once would be 20.5 GB.
- **On a 16 GB laptop:** 6 workers still do not fit, but 4 do: near 12 GB, against
  near 17 GB before. A budget on the jobs admitted would choose this from the memory
  available.
- **Measured against modelled:** the bench measures the 3× run at 4 workers. The release
  figures are modelled.

## A memory budget for the bucket sorts

Even after the change, 6 workers do not fit a 16 GB machine at a release's size. A
stacked change starts each bucket sort only while the estimated peaks of the sorts
running, and of the next, fit a budget. At least one sort always runs.

- **The estimate:** a floor of 0.3 GB, plus 3.2 times the bucket's rows' Arrow bytes as
  the key's row groups size them (`COMPACT_ROW_GROUP_BYTES` / rows per row group).
- **The budget:** `WIKIDATA_FINALISE_MEMORY_GB`, or else Linux's MemAvailable when the
  step starts. With neither, there is no budget.
- **Unchanged:** the jobs, their order, their results and the ledgers, so a restart
  behaves the same.

| | estimated per bucket | measured, one worker |
|---|---|---|
| the bench's 3× key (881,334 rows) | 2.69 GB | 2.32 GB locally, 2.55 GB on CI |
| a release's bucket (1.17M rows, modelled) | 3.48 GB | |

On the bench's runners (15.6 GB, 4 workers on the 3× key), 4 × 2.69 = 10.8 GB fits, so
the budget does not bind and the bench should show no change. On a 16 GB laptop with
about 12 GB available, a release's sorts would run 3 at once. 6 run at once only with
about 21 GB available.

Locally on the 3× key, 4 workers:

| budget | at once | the tree's peak | the bucket sorts' wall time | the buckets |
|---|---|---|---|---|
| none | 4 | 8.72 GB | 21.5 s | |
| 5.5 GB | 2 | 4.83 GB | 31.7 s | byte-identical |
