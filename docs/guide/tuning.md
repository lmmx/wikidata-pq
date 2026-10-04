# Tuning

## How many workers

`WIKIDATA_WORKERS` sets how many chunks are processed at once (`CHUNK_WORKERS`, default 6):

```sh
WIKIDATA_WORKERS=8 just release 20260928 20260507
```

Most of a chunk's processing runs on one thread, with parallel stretches in Polars and in
polars-genson's inference. With one chunk at a time, about half of a 20-core machine was in
use. Running several chunks at once uses more of it. The gain from each added worker falls
as the workers compete for the cores in their parallel stretches and for disk.

`scripts/bench_workers.py` measures this on the release's own unprocessed chunks. Run it
while no release is running:

```sh
python scripts/bench_workers.py 20260928 --chunks 12 --workers 2,3,4,6,8
python scripts/bench_workers.py 20260928 --set main --chunks 12 --workers 3,6
```

It samples `--chunks` chunks spread across those left in the set and processes the same
sample once per worker count, in a scratch release under `releases/_bench_workers/`. The
chunks are hard-linked, so the originals are untouched. Groups are merged but not
uploaded. It prints, for each count, the wall time, chunks and source MB per hour, the
speedup over the first count, the whole machine's CPU use, and the trial's peak memory.

The `--chunks` value should be at least twice the largest worker count. With fewer, each
worker gets one or two chunks, the wall time cannot drop below the slowest single chunk,
and the larger counts measure lower than they would over a full run.

The measurements on 20260928 are in [Performance](performance.md#workers).

## Group size

Partitioned chunks are uploaded in groups, and each group adds one file per language to
each repo. The threshold at which a group closes is set so that a set comes to about
`GROUP_TARGET_COUNT` groups (30), within `GROUP_MIN_GB` (1 GB) and `GROUP_MAX_GB` (25 GB) of
partition files. The upper bound limits disk: a group's partitions and its staged copy
are on disk together at about the same size. More groups means more, smaller files on the
Hub before compaction. See [Push](../reference/push.md#group-size).

## Disk

- `CLEAN_UP_LOCAL` (on) deletes each file once the next stage no longer needs it. With it
  off, nothing is deleted.
- Finalise downloads one table at a time. The claims sort needs about three times the
  claims' size (the local copy, its buckets and the sorted files), so it runs first, before
  the other tables' copies are local.

## Compaction and sort sizes

| Constant | Default | Sets |
|---|---|---|
| `COMPACT_FILE_BYTES` | 500 MiB | target size of each compacted or sorted file |
| `COMPACT_ROW_GROUP_BYTES` | 128 MiB | in-memory size of a row group |
| `SORT_IN_MEMORY_BYTES` | 2 GiB | keys above this (claims) are sorted through buckets |
| `SORT_BUCKET_BYTES` | 64 MiB | source Parquet per sort bucket |
| `COMPACT_DOWNLOAD_WORKERS` | 32 | concurrent downloads from the Hub |

??? info "Documented against"
    Commit `6243a2c` (2026-10-04). See [About these docs](../about.md) to check for changes.

    | File | SHA-256 |
    |---|---|
    | `src/wikidata/config.py` | `99b048d1fd90ac8475cf1443d97a3e33f18c58c228630cf85467a85981f3503a` |
    | `scripts/bench_workers.py` | `76be82eec264b3351f42329187409f8e4d1de6cb2f2e92a8cfc2024b27ffaf00` |
