# 2026-10-25: Where finalise's wall time goes, beside its CPU time

The bench's finalise ETA is finalise's CPU time per byte scaled to the release over the
cores. Any stretch where cores sit idle is hidden by that. This entry lays finalise's
wall time beside its CPU time, stage by stage (bench 7.5.0, lmmx/wikidata-pq#103), and
fixes the largest idle stretch it finds (lmmx/wikidata-pq#107).

## How it is measured

- A sampler process (`bench/timeline.py`) runs over finalise's two runs. Every 0.5 s it
  records the machine's CPU time (busy, iowait, idle), the CPU of the pipeline's process
  tree, and the run's disk. It is reaped after both runs, so its CPU is in none of their
  figures.
- Each timed call in phases.py also records its pid, start, depth and table/key.
- Each finalise stage gets its wall time, the cores busy over it, its idle core-seconds
  (4 × wall − CPU), cores in iowait, and its jobs.
- The machine's CPU and the process tree's agree to within 2%: the runner runs nothing
  else.

## Measured (#103: 3 rounds, 12 runs, mean)

| | wall | CPU | cores busy (of 4) | idle core-s |
|---|---|---|---|---|
| finalise | 36.3 s | 91.8 s | 2.54 | 53 |
| finalise, 3× the rows | 50.4 s | 139.2 s | 2.76 | 62 |
| per byte (3× less the first; what the ETA scales) | +14.1 s | +47.4 s | 3.35 | +9 |

The per-byte part runs at 84% of 4 cores. Its idle comes from three stages:

| stage, per byte | idle core-s | why |
|---|---|---|
| claims bucketing | +2.7 | 1 job: the key has 1 source file; runs at 1.8 cores |
| claims compaction | +2.1 | 1 glue job |
| claims packing | +0.7 | 1 job, then `_check_sorted` in the parent |
| the bucket sorts | −1 | 2 buckets at 3.16 cores busy; 4 buckets at 3.74 |

The bucket sorts are 76% of the per-byte CPU, and their cores are saturated. With 4
buckets on 4 workers, the stage reads 3.9-4.0 cores busy throughout, except:

- a ~1 s burst of iowait (1.8 cores) when the four buckets' writes land together;
- a ~2 s tail, as two buckets finish ~2 s after the others.

Most of the idle is in the fixed part, ~42 of the first run's 53 idle core-s: the in-memory
keys' write of descriptions, labels, links and aliases.

| table | wall | cores busy | idle core-s |
|---|---|---|---|
| descriptions | 11.3 s | 2.6 | 16 |
| labels (in memory) | 6.5 s | 2.7 | 8.5 |
| links | 5.0 s | 2.6 | 6.7 |
| aliases | 1.9 s | 2.4 | 3.1 |

Each table's ~20-40 larger keys run 2 at a time (`FINALISE_LARGE_WORKERS`), each in a
fresh spawned worker. On descriptions:

- 40 larger keys started in pairs every 0.6 s for 12 s, each key's own job ≤ 0.06 s;
- the time is the worker's spawn and its imports, with 2 cores idle;
- the 400 small keys followed in 0.7 s.

## At a release's shape (modelled from 7.4.0's figures)

The release's main-set claims: 44.7 GB in 30 source files, 667 buckets, about 90 part files.

- **4 cores.**
  - The bench's per-byte idle comes from single jobs (one source file, one compacted
    file, one part file). A release runs 30, 30 and ~90 jobs there.
  - The last wave of 30 jobs on 4 workers is 2 jobs, about 1/16 of the stage idle. These
    stages are ~10% of the per-byte CPU.
  - The bucket sorts are saturated, and their tail shrinks with 667 jobs.
  - So a release's CPU-bound part should run at about 95% of the cores: the ETA's
    assumption holds to about 5%, roughly 20 minutes of the ~5.9 h finalise ETA.
- **Disk, not measured by the bench.** The bench runs in page cache: iowait ≤ 0.1 core
  outside the burst above. A release's key (44.7 GB) is 3× the machine's RAM.
  - Its mostly-I/O passes read or write the whole key:
    - compaction's glue and its byte-for-byte check;
    - the copy's sha256 check;
    - packing's glue and its byte check.
  - That is ~90 GB written and ~135 GB read from disk, with cores largely idle. Each
    byte check re-reads files just written, so mostly from cache. It comes to ~7-8 min
    at 500 MB/s and ~19 min at 200 MB/s.
  - By the model, these are the release's largest idle stretches on 4 cores. The bench
    cannot size them.
- **The release machine (20 cores, `FINALISE_WORKERS` 6).**
  - A bucket-sort job uses 1.58 cores alongside one other job on 4 cores, and 0.94 when
    four share them.
  - So 6 jobs keep about 6-10 of 20 cores busy through the stage that is most of finalise.
  - This is the largest idle-core stretch of a real release. It is bounded by each
    bucket's memory, not by the stage structure.

## The fix (#107): the larger keys' next workers started ahead

`in_parallel(..., tasks_per_child=1, ahead=n)` starts n more workers than may run.

- Each takes its next job and imports its module, then waits on a semaphore for one of
  the `workers` slots.
- The larger keys take `ahead = FINALISE_WORKERS − FINALISE_LARGE_WORKERS`.
- Still at most 2 keys hold their rows at once.
- Each key still runs in a fresh worker.
- A waiting worker holds its imports: ~110 MB, measured.

CI, 3 rounds, parity passed:

| | before | after | change |
|---|---|---|---|
| wall, with finalise | 45.29 s | 39.63 s | −12.6% ±3.2% |
| finalise: sort | 36.45 s | 30.77 s | −15.7% ±3.0% |
| finalise: compact | 1.17 s | 1.17 s | |
| finalise, 3× the rows | 53.10 s | 47.07 s | −11.6% ±2.9% |
| ETA: finalise | 6h32m | 6h32m | no change beyond noise |
| CPU, finalise | 1m33s | 1m42s | +9.6% ±4.3% |
| peak RSS, all processes (the pipeline's; finalise's is not measured) | 3.779 GB | 3.774 GB | |
| disk used, peak rise | 756 MB | 756 MB | |

Finalise's idle on 3× the rows fell from 66 to 33 core-s. The stages:

| stage | wall, before → after | cores busy, before → after |
|---|---|---|
| descriptions | 12.2 s → 9.4 s | 2.5 → 3.9 |
| labels (in memory) | 7.0 s → 5.5 s | 2.7 → 3.8 |
| links | 5.3 s → 4.2 s | 2.6 → 3.6 |

The CPU rise is the same 44 worker starts per table run four at once on the runner's 4
vCPUs. The waste is not from extra workers: the processes seen were the same.

At a release, the change saves about a minute: ~40 larger keys a table, each worker
start overlapped with the keys before it.

## Not done, and next

- Finalise's peak RSS is not measured by the bench: the RSS samples stop with the
  pipeline. The timeline sampler could take it.
- The bench's claims key has one source file at either size. A compaction of the x3 run
  into several files would make bucketing, compaction and packing release-shaped. That
  changes what is measured, so it would be a major version.
- The release's I/O passes over the whole key are its likely largest idle stretches on 4
  cores, and a release-scale disk run would size them. Overlapping them with CPU-bound
  work (the next table's compaction and sort) changes the ledgers' order and restart
  points. Not tried.
- On the release machine, the bucket sorts' cores are bounded by `FINALISE_WORKERS` and
  a bucket's memory: measure a bucket job's peak RSS, and set the sort's workers to what
  memory allows.
