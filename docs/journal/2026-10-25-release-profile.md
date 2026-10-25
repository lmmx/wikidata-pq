# 2026-10-25: Where a release's time goes after the faster download, and `_check_written` a column at a time (tried, not shipped)

Continues docs/journal/2026-10-25-finalise-timeline.md.

## Current State

- Nothing changed on master.
- lmmx/wikidata-pq#114 (not to merge) decoded `_check_written`'s row groups a column at a
  time on one thread. The gain doesn't show on CI (see Measured).

## Where a release's time goes

**Source:** #109's bench (bench 7.5.0, 3 rounds, the "after" means).
- **Release ETA:** 17h06m, made of the split 2h49m, the pipeline 7h31m and finalise 6h46m.
- **The download:** since #109 the bz2 takes about 1h57m. It runs under the split, which
  follows it.

**How a stage's hours are reckoned:** its share of its span's CPU, times that span's ETA.
- **Finalise:** its ETA is its per-byte CPU over 4 cores, plus a fixed ~32 s.
- **The split:** its ETA is its CPU over 4 cores.
- **The pipeline:** its ETA is the bench's wall time scaled up, which overstates the
  pipeline rows:
  - the bench keeps only 3.3 of the 4 cores busy (chunk 0's tail, then the group close
    running alone);
  - 4.4 s of the 24.2 s all-process CPU is outside the timed stages.

  A CPU-based pipeline ETA would read about 6h10m.

| cost | span | release (4 cores) | basis |
|---|---|---|---|
| `normalise_in_schema` (genson) | pipeline | 3h04m | bench share × ETA |
| `compact._write_file_parallel` (arrow-rs, the Hub's settings) | finalise | 2h22m | bench share × ETA |
| `write_table`, the claims (arrow-rs, through genson) | pipeline | 2h14m | bench share × ETA |
| lbzip2's decode | split | 1h42m | bench (decoding alone, 1h42m25s) |
| `partition_parquet` | pipeline | 1h22m | bench share × ETA |
| `_read_bucket` (before #112) | finalise | 1h15m | bench share × ETA |
| `_check_written` | finalise | 1h04m | bench share × ETA |
| the split besides the decode: reshape 30m, the chunk files' write ~20–25m, the rest ~10m | split | ~1h07m | bench; the write's share from a local profile |
| `_sorted` | finalise | 48m | bench share × ETA |
| `merge_group` | pipeline | 25m | bench share × ETA |
| finalise's idle at a release | finalise | ~20m | modelled (finalise-timeline) |
| finalise's disk-bound passes | finalise | 7–19m | modelled (finalise-timeline) |
| the md5 re-read after the download | split | ~1–2m on the path | modelled (below) |

- **arrow-rs:** its two Parquet writes are 4h36m, 27% of the release.
- **The md5 re-read:**
  - hashlib's md5 runs at about 570 MB/s on one core here. The bz2's 103.5 GB takes 3.0
    min CPU-bound, or 8.6 min at 200 MB/s of disk.
  - It starts once the download ends and is done well before the split. It costs the
    split at most one core of four, for a few minutes.
  - For gz from the mirror (156.6 GB, 4.6–13 min), it can reach the split's path only at
    the mirror's slowest rates (up to ~6 min).
- **Local profiles** (`process()`'s steps on the bench's c0, r50, s30 and s90, polars-genson
  1.5.0):
  - The claims call is about 90% of `normalise_in_schema`'s CPU; the labels,
    descriptions and aliases calls make up the rest (1.1 CPU-s).
  - Of the claims call's wall time:
    - genson's decode against `claims.avsc`: 3.75 s of 5.97 s;
    - the read: 0.96 s;
    - extraction: 0.78 s;
    - conforming: 0.48 s.

## The bz2 split: decoders

The bench's 77.3 MB dump slice decodes to 1,036.8 MB of JSON. Decoded on this 4-core VM,
warm:

| decoder | CPU | wall |
|---|---|---|
| lbzip2 2.5, 4 threads; 1 thread | 23.5 s; 25.0 s | 6.2 s; 25.2 s |
| 7-Zip 23.01, `-mmt1`; `-mmt4` | 23.5 s; 21.5 s | 24.8 s; 20.0 s |
| rapidgzip (bz2), `-P 4` | 24.5 s | 7.3 s |
| indexed_bzip2 1.7.0 (`ibzip2 -P 4`) | 26.2 s | 7.5 s |
| bzip2 1.0.8 | 30.9 s | 33.6 s |

- **lbzip2 is the floor:** no exact decoder beats its CPU per byte. 7-Zip matches it on one
  thread, but does not spread one stream across cores.
- **The rest of the split** (the per-chunk steps timed on the slice):
  - the compiled reshape: 5.4 CPU-s;
  - the chunk files' Polars write: 2.8–4.5 CPU-s, nearly all zstd over the JSON text;
  - `is_scholarly`: 0.5 CPU-s;
  - reading the decoder's lines: 0.55 CPU-s.

  Writing the chunk files at zstd 1, as lz4, without statistics, or with pyarrow and no
  dictionaries for the JSON columns all came out within noise of zstd 3.
- **So a bz2 split can't go much below 1h42m on 4 cores.** gz from a mirror (54m–1h00m) is
  the route. From Wikimedia, at its 3-connection ~14.5 MB/s, the 156.6 GB gz takes about
  3h to download.

## `_check_written` a column at a time

**What the check costs** (the bench's claims key, 1,175,112 rows in 7 row groups, written
with the Hub's settings):
- pyarrow's decode is about 80% of its CPU;
- `Table.equals` is about 20% (0.8 s).

**The change (#114):** each row group is decoded a column at a time on the calling
thread, and each column compared with `ChunkedArray.equals`. It is the same pyarrow
decode of every byte and the same values compared. The schema and row-count checks are
the same, and `scripts/test_sort.py` passes.

**Locally:** 3.0–3.9 CPU-s against 3.8–5.7 (fresh processes, 6 runs each, interleaved).

**On CI** (3 rounds, f09eeb2 against 95ad196, parity passed):

| | before | after |
|---|---|---|
| `_check_written`, per byte | 7.2 s | 6.7 s, within noise |
| CPU, finalise | 1m30s | 1m27s, within noise |
| ETA: finalise | 6h03m30s | 6h12m46s, within noise |
| ETA, a release | 15h42m15s | 15h41m58s, within noise |

**Not shipped:**
- **Why:** the local saving was the cost of faulting in each decoded row group's fresh
  memory. This VM overstates that cost; CI's runners don't.
- **What's left:** one independent decode of every written byte (~80% of the check) plus
  the comparison.
- **The alternatives:**
  - **A Rust-side comparison:** it could save only the comparison's share.
  - **A Rust-side decode:** it would be arrow-rs reading its own output, a weaker check.
  - **Polars as the decoder:** it took 8.4–9.3 CPU-s, about twice pyarrow.

## Missing

- **The arrow-rs prototypes are not kept anywhere.** These are the shared repetition
  levels and the u32 dictionary indices, described in polars-genson's
  `docs/journal/2026-10-22-write-path-comparison.md` (branch `docs/write-path-comparison`,
  not merged). They were a disposable patched copy of parquet 60.0.0 and need rewriting
  before any bench.
- **`_sorted`'s split on CI:** its sort, take and check have not been timed apart on CI.
  It is 13–14% of finalise's per-byte CPU. On this VM the threaded take of the claims key
  is 10–16 CPU-s and the sort 0.1–0.2 s. #54 measured the take at 0.6 s on CI on the
  in-memory path.
