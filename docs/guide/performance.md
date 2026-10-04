# Performance

Measured timings of the release 20260928, on one machine: 20 cores, 125 GB of memory, the
working files on an encrypted home volume. The run was in progress when this page was
written; the stages not yet measured are listed at the end. The development journal
(`docs/journal/2026-10-03-release-20260928-run.md`) has the full record.

## The release

| Set | Chunks | Entities | Median rows per chunk | Source (Parquet, zstd 9) |
|---|--:|--:|--:|--:|
| scholarly | 10,853 | 46,383,002 | 3,643 | 49.5 GB |
| main | 12,182 | 75,432,640 | 7,128 | 44.7 GB |

The bz2 dump is 103 GB.

## Processing one chunk at a time

The scholarly set's first 2,476 chunks were processed one at a time, each in its own
process. Times between consecutive chunks' audit files:

| Rows | Source | Time |
|--:|--:|--:|
| 15 | 0.0 MB | 4.4 s |
| 290 | 0.6 MB | 5.3 s |
| 333 | 0.8 MB | 5.6 s |
| 1,119 | 2.0 MB | 7.4 s |
| 861 | 2.2 MB | 7.7 s |
| 1,452 | 3.7 MB | 10.5 s |
| 3,525 | 8.7 MB | 17.3 s |
| 2,827 | 9.1 MB | 17.9 s |

These fit about **4.4 s per chunk plus 1.5 s per MB** of source. The fixed part was the
process start-up and imports, and each chunk reading every chunk's state file about six
times (0.4 s a read over 13,000 files); the reads are now one chunk's file each.

Over an hour at the 2,400-chunk mark: 330 chunks per hour, 0.93 GB of source per hour.
From the start of the run to that point: 11.9 GB in 9.6 hours (1.24 GB/h), uploads
included. About 50% of the machine's CPU was in use.

## Group uploads

With one chunk at a time, processing paused while each group was merged, uploaded and
verified. The gap between a group's last chunk and the next chunk, for the first seven
groups: 326, 305, 316, 314, 307, 1,096 and 330 s. Groups now upload in a background thread
while chunks continue.

## Workers

`scripts/bench_workers.py 20260928 --chunks 12 --workers 2,3,4,5,6,7,8`, on 12 scholarly
chunks spread over chunks 2477 to 10155 (42,153 rows, 31 MB), with the run stopped:

| Workers | Wall | Chunks/h | MB/h | Speedup over 2 | CPU (machine) | Peak memory |
|--:|--:|--:|--:|--:|--:|--:|
| 2 | 56 s | 778 | 1,981 | 1.00× | 42% | 16.1 GB |
| 3 | 46 s | 930 | 2,367 | 1.20× | 50% | 16.6 GB |
| 4 | 44 s | 980 | 2,493 | 1.26× | 53% | 18.2 GB |
| 5 | 41 s | 1,053 | 2,680 | 1.35× | 58% | 16.0 GB |
| 6 | 40 s | 1,090 | 2,773 | 1.40× | 61% | 17.5 GB |
| 7 | 38 s | 1,136 | 2,891 | 1.46× | 63% | 19.7 GB |
| 8 | 38 s | 1,151 | 2,929 | 1.48× | 64% | 18.5 GB |

With 12 chunks, each worker at 6 or more gets one or two chunks, so the wall time is
bounded below by the slowest chunk plus start-up and the last group's merge. Peak memory
stays at 16 to 20 GB from 2 to 8 workers. The default is 6.

## Not yet measured

- Download, split and route times for 20260928.
- Processing rate with 6 workers over a full run, and on the main set's chunks.
- Finalise: compaction, the sort and claims_labels of each set, against the Hub.
- Promotion.
- The total time from dump to published release.
