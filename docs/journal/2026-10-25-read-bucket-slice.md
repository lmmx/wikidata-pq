# 2026-10-25: A bucket's one-run rows sliced, not filtered (tried, not shipped)

## Current State

- `sort_by_id._read_bucket` is as #112 left it. A row group read directly is kept as
  read when every row is the bucket's, and filtered (copied) otherwise. Nothing here
  changed on master.
- lmmx/wikidata-pq#115 (not merged) took a partly kept row group's rows as a zero-copy
  `rg.slice(first, count)` when they form one run (`last − first + 1 == count`). It
  filtered only when they form several runs.
- The output was byte-identical, parity passed, and `scripts/test_sort.py` passed with
  new cases for a one-run row group (checked to be a slice) and a several-runs one.

## How often the kept rows are one run (the bench's claims key, local)

Built with `bench/pipeline.py`'s quick-tier workload. Row groups read directly, by the
rows each bucket keeps:

| key | buckets | whole | one run | several runs | rows: whole / one run / several |
|---|---|---|---|---|---|
| 1× (1.18M rows, 73 MB) | 2 | 10 | 2 | 2 | 929k / 123k / 123k |
| 3× (3.53M rows, 219 MB) | 4 | 24 | 15 | 3 | 2.05M / 937k / 192k |

- **One run is the usual case.** On the 3× key, 15 of 18 partly kept row groups (83%)
  keep one run, and the run is 937k of their 1.13M kept rows.
- **Several runs** (4 each) happen in the last bucket, whose row groups mix id lengths
  (`Q9…` against `Q10…` in string order).
- **Input memory.** The bucket's input as `_read_bucket` returns it (Arrow buffers
  referenced), filtered → sliced:
  - bucket 0: 888 → 1042 MB (+17%);
  - bucket 1: 1019 → 1475 MB (+45%);
  - bucket 2: 1149 → 1422 MB (+24%);
  - bucket 3: 774 → 827 MB (+7%).
- **Peak RSS** of bucket 1's sort, in a process of its own: 3.87 → 4.55 GB (+650 MB,
  +17%). The 3× key overstates this, since its ids repeat three times and each boundary
  crosses three half-kept row groups.

## Measured on CI (#115, quick tier, 3 rounds, bench 7.5.0)

| | before | after |
|---|---|---|
| `_read_bucket` per byte | 7.3 s | 6.9 s (−5.7% ±1.1%) |
| `_sorted` per byte | 6.1 s | 7.4 s (+22.1% ±9.1%) |
| finalise per byte, all stages | 45.7 s | 46.5 s |
| finalise ETA | 6h35m52s | 6h39m02s (within noise) |
| CPU, finalise | 1m33s | 1m32s (within noise) |
| peak RSS, all processes | 3.94 GB | 3.89 GB (−1.2% ±0.6%, the pipeline's peak) |
| peak RSS, one worker | 1.72 GB | 1.72 GB (within noise) |
| disk, peak rise | 756 MB | = |

Parity passed. The bench doesn't yet record finalise's own RSS (bench 8.0.0 had not
landed), so neither peak RSS figure covers a bucket worker. The +650 MB per bucket
worker is local only.

## Why it didn't pay

The copy saved was the filter's fresh pages, and that cost moved rather than going
away. With the filter, each row group is freed once copied, so `_sorted`'s take reuses
pages already faulted in. With slices, every row group stays alive until the sort. The
take then has to fault in about a bucket's worth of fresh memory, so `_sorted` costs
what `_read_bucket` saved and more: +1.35 s against −0.42 s per byte.

## At a release (modelled)

Assumptions, from bench CHANGELOG 7.4.0's release model:
- 780M claims rows, 70–80% read directly;
- 1.13–1.21 direct reads a row read directly;
- 667 buckets of 1.17M rows;
- about 1.16 KB a row in Arrow, as the bench's buckets have.

A shared row group's rows are all partly kept, so partly kept rows ≈ (reads − 1) × rows
read directly ≈ 76–123M. Every partly kept row is assumed one run, which is an upper
bound.

- **CPU:** CI's per-row figures, from the paired changes, are −0.52 µs in `_read_bucket` and +1.66 µs in
  `_sorted`, over the 814k rows sliced in the part per byte. They come to a net +87 to
  +140 CPU-s for a release, about +22–35 s of finalise ETA on 4 cores. Even if the
  `_sorted` cost could be avoided, the saving alone would be 40–64 CPU-s, 10–16 s of a
  6.6 h finalise.
- **Memory:** each bucket worker would hold its shared row groups' rows the bucket
  doesn't keep: 0.13–0.21 × 0.75 × 1.17M ≈ 114k–184k rows, about 0.13–0.21 GB of Arrow.
  At the local RSS-to-buffer ratio (1.43×), that's 0.19–0.30 GB a worker, or 1.1–1.8 GB
  with 6 finalise workers. On a 16 GB laptop that's material, and it buys no CPU.

## If it comes back

Only if the take in `_sorted` stops depending on reusing the input's pages: for
example, if the sort's output were written into buffers allocated before the bucket is
read. Without that, slicing trades memory for no CPU.
