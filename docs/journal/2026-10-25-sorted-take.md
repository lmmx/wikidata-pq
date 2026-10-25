# 2026-10-25: The sort's take, a column at a time, its rows freed as taken

On the bucket path, `sort_by_id._sorted` cost about 2.4 CPU-s per million rows on CI,
against about 0.5 on the in-memory path (#54): 13% of finalise's per-byte CPU, about 48
min of a release's finalise ETA. lmmx/wikidata-pq#118 changes how it takes the rows.

## Where the time went

One bucket of the bench's 3× claims key: 881,505 rows in 15 chunks, 1.02 GB of Arrow.
Measured locally, one fresh process per variant, so indicative only.

- **`sort_indices`:** 0.06 s.
- **`_check_sort`:** about 0.15 s.
- **The take:** the rest. About 0.7 s of it was user time in every variant. Its system
  time, page faults for fresh copies, ran from 1.2 s to 12.5 s.

A take from chunks combines a nested column inside the take. With a thread a column, a
bucket worker held three things at once: its rows, their sorted copy, and up to four
combined columns. mimalloc (pyarrow's pool) reuses little of what other threads free.

| take | sys | peak RSS |
|---|---|---|
| a thread a column (before) | 1.2-12.5 s | 3.41-3.44 GB |
| `Table.take`, unthreaded | 3.0-9.4 s | |
| the table combined, then taken | 5.2-8.9 s | |
| per column: combine, take, free the combined copy | 3.0-5.4 s | |
| **per column: combine, free its chunks, take** (kept) | **0.25-1.4 s** | **2.53 GB** |
| the same, on 2 or 4 threads | 2.9-6.3 s | 2.5 GB |

The kept variant takes the table from the caller, so nothing else holds it. Arrow's
live bytes then stay at about one copy of the rows, plus one combined column at a time,
through the whole take.

A k-way merge of the sources' sorted runs was not tried. It could save only the sort
indices, 0.06 s, and the runs are short in string order: `Q100000` sorts beside
`Q1000000`…`Q1000009`, then `Q100001`. The bench's 3× key also repeats each id three
times.

## On CI (bench 8.0.0, 3 rounds, parity passed)

| | before | after | change |
|---|---|---|---|
| `_sorted`, finalise per byte | 6.2 s | 4.9 s | −20.2% ±14.7% |
| finalise, per byte | 46.2 s | 44.8 s | |
| ETA: finalise | 6h58m23s | 6h56m44s | no change beyond noise |
| CPU, finalise | 1m32s | 1m32s | |
| wall, with finalise | 44.41 s | 44.33 s | |
| peak RSS, finalise: one worker | 2.770 GB | 2.514 GB | −9.2% ±2.0% |
| peak RSS, finalise: all processes | 6.056 GB | 5.524 GB | −8.8% ±1.0% |
| peak RSS, finalise 3×: one worker | 4.227 GB | 3.750 GB | −11.3% ±10.5% |
| peak RSS, finalise 3×: all processes | 13.491 GB | 12.274 GB | −9.0% ±2.9% |

- **CPU:** the take's saving is real but small on the runners. Their faults are cheaper
  than on the shared machine, and the finalise ETA moves within its noise.
- **Memory:** the gain is in peak memory. A bucket worker's peak falls by about 0.5 GB of
  its 4.2 GB.
- **RSS against live bytes:** RSS falls less than Arrow's live bytes do. Freed input pages
  stay resident under mimalloc, and the allocator is unchanged.
