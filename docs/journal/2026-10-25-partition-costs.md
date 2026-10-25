# 2026-10-25: Where partitioning's CPU goes, and the sidecar from the frame's stats

Continues docs/journal/2026-10-22-sidecar-footers.md and
docs/journal/2026-10-24-genson-writer.md.

## Current State

- `partition_parquet` runs its sink and `key_stats` (each key's row count and the min
  and max of its ids) as one query with `pl.collect_all`, so the two share a scan.
  `write_frame_sidecar` then writes the sidecar from those stats: the key dirs are not
  listed and no footer is read, and only each file's size is taken.
- The sink writes one file per key whatever the key's size
  (`approximate_bytes_per_file=None`), so each key's file is the one the sidecar lists.
  The default only split a key past 4 GiB of in-memory data. A chunk's table is tens of
  MB.
- `place_unsplit` (claims) and `partition_parquet(arrow=True)` still write their
  sidecars from the footers (`write_sidecar`, `file_stats`).
- The partition files and sidecars are byte-identical to master's on the bench's four
  chunk files (c0, r50, s30, s90: 2,328, 856, 109 and 174 files).
- `scripts/test_sidecar.py` checks `partition_parquet`'s sidecar against
  `write_sidecar`'s on:
  - keys whose dirs sort differently as paths than as strings (`be` / `be-tarask`,
    `zh` / `zh-hans` / `zh_min_nan`);
  - ids long enough to be truncated;
  - multi-byte and null ids;
  - a `ref` column.

## Measured

**CI bench** (quick tier, 3 rounds, lmmx/wikidata-pq#89):

| | before | after | change |
|---|---|---|---|
| CPU, partition | 3.4 s | 3.0 s | -11.2% ±5.3% |
| CPU, all processes | 23.0 s | 22.4 s | -2.4% ±2.0% |
| `write_sidecar` + `file_stats` | 0.6 s | 0.0 s | |
| `partition_parquet` | 2.8 s | 3.0 s | +6.1% ±3.0% |

- **Peak RSS of one worker:** +33 MB (+1.9%).
- **Parity:** passed.

**What `partition_parquet` spends its time on:** files, not bytes.
- Chunk 0 writes 2,321 partition files: 564 labels, 439 descriptions, 489 aliases,
  827 links, 1 entities, and 1 claims as a hard link.
- Each table's rows written as one file take 0.06–0.16 s of CPU. Partitioned, they take
  0.57–1.18 s.
- Polars' sink costs about 0.9 ms of user CPU a file. About 40% of that is zstd, set
  up afresh for each column chunk. Written uncompressed, chunk 0's four tables took
  1.1–1.3 s against 2.1–2.3 s.

**What `write_table` spends its time on:** encoding the claims' 93 leaf columns (perf, a
pq-glue build with symbols, chunk 0):

| | share of the profile |
|---|---|
| nested levels, built (`LevelInfoBuilder`) | 24% |
| levels and dictionary indices, RLE encoded | 18% |
| zstd | 13% |
| statistics, min and max of every leaf, page by page | 7–9% |
| string dictionaries | 7% |

## Ruled out

- **pq-glue's `write_parquet` in place of polars-genson's:** the same bytes.
  - On a captured table read from Arrow IPC it took 30–40% less CPU. Most of that was
    system time from the page faults of genson's allocator (mimalloc).
  - Inside `process()` (the four chunks, twice each), the two took the same: 4.3–4.8 s
    of user CPU and 0.4–0.5 s of system time.
- **The claims' writer on one thread** (`engine="native"`): 8% less CPU, but chunk
  0's write takes more than twice the wall time. The quick tier's wall is chunk 0's.
- **Row groups of 32,768 or 1,048,576 rows** in place of 122,880: the same CPU.
- **zstd level 1 for the partition files** (Polars' default is 3): 5% less CPU for
  the four chunks' files, 3.6% larger.
- **The partition files written from Python** (`partition_by` with pq-glue or pyarrow
  per key, or `pyarrow.dataset.write_dataset`): as slow as the sink or slower. The
  per-file cost is in the writer's setup, not the data.
- **Sinking a collected frame** (`lf.collect().lazy().sink_parquet`): 2.8–2.9 s
  against 2.1 s. The exploded frame is in many chunks. Rechunked, it costs the same as
  the scan.

## Missing

- **Statistics on the claims' `id` alone:** pq-glue's writer has no option for which
  columns get statistics. A pq-glue build with statistics on `id` only took 12% less
  CPU for chunks 0 and 90's statements (3.71 s to 3.19 s), and one with chunk-level
  statistics only took 9% less. The sidecars read only `id`'s.
- **Fewer partition files:** a chunk writes a file per key. Writing one file per table,
  with a row group per key, would need a merge that glues chosen row groups (pq-glue's
  `concat` takes whole files) and sidecars that name row groups rather than files.
