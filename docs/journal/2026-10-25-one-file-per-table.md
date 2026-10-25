# 2026-10-25: One partition file a table per chunk, a row group per key (tried, not shipped as is)

Continues docs/journal/2026-10-25-partition-costs.md.

## Current State

- lmmx/wikidata-pq#106 is open, as a measurement, and is not to be merged as it stands.
  - It writes labels, descriptions, aliases and links as one file per chunk, at
    `results/<table>/chunk_N.by_<language|site>.parquet`. Each key has its own row group,
    written by pyarrow.
  - The group merge glues each key's row groups out of those files.
  - It takes pq-glue from the CI wheel of lmmx/pq-glue#10, through `bench/wheels`.
- lmmx/pq-glue#10 (draft) lets `concat` and `concat_many` take `(file, row_groups)`
  sources. Within one `concat_many` call, a file that several jobs copy from is opened, and
  its footer read, only once.
- Master is unchanged.

## What depended on a file per key

| what | how it changed |
|---|---|
| sidecar (`_write_sidecar_rows`) | Still one row per key, with the same key, rows and id bounds, in the same order. `path` is now the one file and `row_groups` is a new column. `file_size` is the key's row-group bytes; the first row also counts the footer, so the sizes still add up to the file sizes, which `chunk_partition_bytes` (the group threshold) relies on. |
| `validate_chunk_outputs`, partition resume | Not affected. They look at the processed files, and the one file sits beside them. |
| `merge_group` / `_glue_table` | Each key's sources are `(file, row_groups)`. The partition files are deleted only once every key of the table is staged. A key whose files are partly gone, but whose staged file exists, was merged before an interruption. |
| group ledger, build-branch layout | Unchanged: one group file per key per group. Each has the same rows, the same row groups (one per chunk) and the same Arrow schema as before; only the page encoding is new (pyarrow's). |
| compact (`_check_glued`, `_link_run`) | Unchanged: group files are still glued or linked byte for byte. |
| #93's planning and direct reads | Unchanged. They read compacted files, which have the same row groups as before. |
| `card_stats` | Reads the published copy only. |
| parity | Now compares the sidecars without their layout columns (`path`, `row_groups`, `file_size`); bench 8.0.0. |
| claims_labels (deduplicated), claims, entities | Partitioned as before. |

## Measured

**Correctness:**
- On the bench chunks (c0, r50, s30, s90), all 3,431 keys were checked against the
  file-per-key partition: the same rows, the same order, the same row-group count and the
  same Arrow schema per key.
- `scripts/test_sidecar.py`, `test_merge.py` and `test_sort.py` pass.
- In a local quick-tier run, all 2,324 published part files came out byte-identical
  (sha256), and parity passed. Parity also passed on CI.

**CI bench** (quick tier, 3 rounds, two heads of #106):

| | before | after, plain (`0f213f1`) | after, dictionaries but ids (`5432fd8`) |
|---|---|---|---|
| CPU, partition | 3.1 s / 3.8 s | -37.6% ±2.2% | -38.2% ±1.2% |
| CPU, group merges | 1.3 s / 1.8 s | -16.7% ±2.5% | -20.2% ±7.6% |
| CPU, all processes | 21.0 s / 25.4 s | -6.7% ±2.6% | -8.1% ±1.2% |
| wall (pipeline) | 6.63 s / 7.70 s | -5.7% ±4.5% | -2.9% (n.s.) |
| ETA: pipeline | 6h44m58s / 7h50m23s | -5.7% ±4.5% | -1.7% (n.s.) |
| finalise: sort | 33.17 s / 38.23 s | +6.6% ±5.2% | +6.1% ±2.7% |
| wall, with finalise | 40.95 s / 47.30 s | +4.3% (n.s.) | +4.2% ±2.6% |
| ETA, a release | 15h17m36s / 17h50m22s | -3.8% ±3.0% | n.s. |
| peak RSS, one worker | 1.758 / 1.711 GB | n.s. | +4.6% ±3.1% |
| disk, peak rise | 754 MB | -1.2% | -0.6% |

The two heads ran on runners of different speeds, so the "before" column gives both
heads' baselines.

**Why the sort is slower:** the stage CPU of finalise is the same, to within 2%, in every
round. The extra wall time is labels' "larger keys" step, which went from 20 keys to 29
(5 s to 8 s). That step takes keys by the bytes of their group files: those over
`SORT_IN_MEMORY_BYTES // 8`, scaled to the sample, about 73 KB. The pyarrow-encoded labels
group files are 8% larger (10.2 MB against 9.4 MB), which pushes 9 keys over the
threshold, and each such key gets a worker process to itself. The same 20 → 29 shift
showed up locally.

## Not shipped, and why

- **What's published can depend on the group files' bytes.** Several sort decisions are
  taken on group-file or compacted-file sizes:
  - which sort a key takes: in memory, or through buckets (`SORT_IN_MEMORY_BYTES`);
  - how many part files an in-memory key is written as (`ceil(size / COMPACT_FILE_BYTES)`);
  - how many buckets a bucketed key gets (`SORT_BUCKET_BYTES`);
  - compaction's runs (`COMPACT_FILE_BYTES`);
  - the row-group sample, which is taken from the largest file by size.

  Any writer other than Polars' sink encodes the same rows to other sizes. At a release's
  scale, a key near one of these thresholds would then be split into other part files.
  The rows and their order would be the same, but the files would not be byte-identical.
  On the bench they were identical because no key crossed a threshold that changes the
  output.
- **The end-to-end gain is within noise, or negative.** Partitioning and merging get
  substantially cheaper (partition CPU -38%, merge CPU -20%, all processes -8%). But the
  pipeline's wall and ETA don't move beyond noise on the second head, and the measured
  sort's wall goes up by 6%.
- **The remaining cost is per row group.** pyarrow's writer spends about 0.5 ms on each
  row group (each key), mostly setting up zstd for every page. One file per table saves
  the file system and Polars' partition machinery, not that set-up.

## Next

- To publish identical bytes, a key's row groups would have to be encoded exactly as
  Polars' sink encodes them. For example: Polars writes each key to memory, and pq-glue
  glues those buffers into the one file. That keeps the encoder's set-up for each key,
  and saves only the file-system share, roughly a fifth of `partition_parquet` locally.
- Alternatively, take the sort's thresholds from decoded or uncompressed sizes (from
  the footer) rather than file sizes. That would decouple what is published from how the
  group files are encoded. It changes the published layout once, and is a change of its
  own.
