# 2026-10-25: A bucket's direct row groups read by pq-glue (tried, not shipped)

## Current State

- `sort_by_id._read_bucket` still reads each planned row group with pyarrow's
  `ParquetFile.read_row_group`. Nothing here changed on master.
- lmmx/wikidata-pq#110 (not merged) switched those reads to
  `pq_glue.read_parquet(src, row_groups=[g])`. It took pq-glue from the wheel CI built
  for lmmx/pq-glue#7 (run 40118655266). #7 makes a read of some row groups read only
  their byte ranges.
- lmmx/wikidata-pq#112 is a separate change found while profiling: a row group whose
  rows all belong to the bucket is kept as read, not filtered. On the bench,
  `_read_bucket` −15.1% ±3.3% and finalise ETA −4.6% ±2.3%, with parity passing.

## Where `_read_bucket`'s time goes (local, fresh process)

The bench's claims key, at three times its rows (4 buckets), each bucket read in a
fresh process as in its worker:

| read | CPU | decode user | decode sys | filter user | filter sys |
|---|---|---|---|---|---|
| pyarrow | 22.3–25.1 s | 8.1 s | 6.9–8.6 s | 2.4 s | 4.2–4.5 s |
| pq-glue, one row group a call | 14.9–17.3 s | 7.6–7.9 s | 0.8 s | 2.3 s | 3.8–5.5 s |

- **Decode:** the two readers' user time is about the same. The difference is kernel
  time: pyarrow's read spends 7–8 s of system time taking memory on this machine, and
  pq-glue's spends under 1 s.
- **Filter:** the filter to `[lo, hi)` is mostly kernel time too, the copy's fresh
  pages. That is what #112 removes for row groups wholly in one bucket.
- **Values and output:** both readers give equal tables and schemas (with metadata), one
  chunk per row group. The sorted bucket files are byte-identical.
- **A batched read** (one call for each source's direct row groups, decoded in
  parallel) cost no less CPU and raised peak memory from 1.7 to 3.0 GB, because every
  row group is decoded before any is filtered.

## Measured on CI (#110, quick tier, 3 rounds, bench 7.5.0)

| | before | after |
|---|---|---|
| `_read_bucket` per byte | 8.3 s | 8.6 s (+4.4% ±2.1%) |
| `_read_bucket`, first run (2 buckets on 4 cores), CPU / wall | 2.4–3.8 s / 1.7–2.6 s | 2.1–3.2 s / 2.1–3.2 s |
| `_read_bucket`, 3× run (4 buckets on 4 cores), CPU | 8.5–13.6 s | 8.5–13.3 s |
| finalise ETA | 6h26m | 6h19m (within noise) |
| CPU, finalise | 1m33s | 1m31s (within noise) |
| peak RSS, all / one worker | 4.01 / 1.72 GB | 3.97 / 1.71 GB (within noise) |

Parity passed.

## Why it didn't pay

The local gain was kernel time spent taking memory, and the runners don't have that
cost. There, pyarrow's read is mostly user time, which arrow-rs matches but doesn't
beat.

- **With every core busy** (the 3× run): CPU is the same within 1–3%.
- **With cores free** (the first run's 2 buckets on 4 cores): pq-glue saves 10–20% CPU,
  probably pyarrow's threads' overhead, but takes longer in wall time, since it decodes
  a row group on one thread.

The ETA's per-byte part is the 3× run less the first, so the saving in the first run
alone reads as a rise per byte. Page faults cost far more under this shared machine's
kernel than on a runner, so its −31% CPU for the read didn't appear on CI at all.

## If it comes back

- pq-glue saves CPU only where cores are free while buckets sort, and it costs wall
  time per bucket there. Whether a release has free cores then depends on its machine
  and FINALISE_WORKERS, so it would need a full-tier or release-shaped measurement.
- The filter's copy of a partly kept row group could be a zero-copy slice. Its kept
  rows are nearly always one contiguous run. But a slice keeps the whole row group in
  memory until the bucket is sorted, and at a release most direct row groups are shared
  by two buckets (about 1.8 reads a row, bench CHANGELOG 7.4.0). That trade is untested.
