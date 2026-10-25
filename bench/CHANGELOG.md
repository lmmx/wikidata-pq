# Changelog

The benchmark harness (`bench/`, `.github/workflows/bench.yml`): how wikidata-pq is
measured, not what is measured. Changes to wikidata-pq's code or dependencies (a newer
polars-genson, say) are what the benchmark measures, and never change its version.

Its version is in `bench/pyproject.toml`, and every result records the version that
measured it. Results compare only within a major version and tier: a change to what a tier
measures or how (its workload, the timing, parity) is a new major version; a new tier,
or a change to how results are shown or stored, is a minor or patch version. Each version's entry says exactly what changed.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
harness follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [8.0.0] - 2026-10-25

### Changed

- finalise's run on three times the rows sorts the claims key from a release's files for
  its bytes. A release's claims group file is larger than the compaction target
  (`COMPACT_FILE_BYTES`, 500 MiB): 20260928's main-set claims were 44.7 GB in 30
  compacted files, one a group. So each group file is a compacted file of its own,
  linked rather than glued (`compact._link_run`), and each is a bucketing job. Its part
  files are 500 MiB each: 86, about 2.9 a group file. The run on three times the rows
  now sets the target (`WIKIDATA_COMPACT_FILE_BYTES`, read by config.py, and in the
  pipeline's process for compaction's and packing's plans) in that ratio to the claims
  group files' mean size: 73.0 MB × 524 MB / 1.49 GB = 25.7 MB on the quick tier
  (`x3_compact_file_mb`). The first run is as it was.
- Every other key's three times is far under that target (2.6 MB at most on the quick
  tier, `x3_other_key_max_mb`), so every other key is compacted and sorted into the same
  files as before.
- The run on three times the rows also takes the full tier's several groups: each
  group's file of a key is linked as three groups' (chunks first + j n to last + j n).
  Before, every copy was named as if the run had one group, so the full tier failed with
  `FileExistsError` from 7.4.0 on.

### Added

- finalise's peak RSS, in each of its two runs: the process tree's, its processes'
  resident memory summed at each sample (`finalise_peak_rss_mb`, as `peak_sum_rss_mb` is
  the pipeline's), and the largest worker's own peak (`finalise_worker_peak_mb`, VmHWM
  as last sampled), and `_x3` for three times the rows. timeline.py's sampler takes
  them, every 0.5 s. A worker that starts and ends between two samples is not seen; nor
  is its peak after its last sample. The comment shows them in the table.
- Each run's claims key's compacted source files and part files (`sort_sources`,
  `sort_parts`, and `_x3`), shown with the bucket path under the ETA.

### Fixed

- The pipeline's RSS sampler (`tree_rss_mb`) no longer dies when a thread exits between
  being listed and its `children` being read. Its samples stopped there before.

What finalise's figures on three times the rows measure has changed (and so its per-byte
part, the finalise ETA and `finalise, 3× the rows`), hence the major version.

### The claims key's shape on three times the rows, against a release's

Bench figures are measured (quick tier, 7.5.0's and this version's runs on three times
the rows). The release's are measured on 20260928's main-set claims
(docs/journal/2026-10-05-sort-speed.md: 30 compacted files, 44.7 GB, 667 buckets, 86
part files), except where marked modelled (7.4.0's model: about 780 million rows).

| | bench 7.5.0, 3× rows | bench 8.0.0, 3× rows | release, main-set claims |
|---|---|---|---|
| compaction target | 500 MiB | 25.7 MB | 500 MiB |
| group files, and their mean size | 3, 73.0 MB | 3, 73.0 MB | 30, 1.49 GB |
| compacted source files (bucketing jobs) | 1, 219 MB (3 glued) | 3 (each linked) | 30 (each linked) |
| compaction's claims worker jobs | 1 (the glue and its byte check) | 0 | 0 |
| buckets (bucket-sort jobs) | 4 | 4 | 667 |
| source files a bucket reads | 1 of 1 | 3 of 3 | about 16 of 30 |
| part files (packing jobs) | 1 | 4 | 86 |
| part files a source file | 1 | 1.33 (2.84 planned, one a bucket at most) | 2.87 |
| buckets a part file | 4 | 1 | 7.8 |
| rows a part file | 3.53M | 881k | about 9.1M (modelled) |
| jobs on 4 workers: bucketing, bucket sorts, packing | 1, 4, 1 | 3, 4, 4 | 30, 667, 86 |

The first run (the sample's rows) stays at 1 source file, 2 buckets and 1 part file.

What it changes in the part per byte (three times the rows less the first run):

- More like a release:
  - The claims key is no longer glued. A release links its group files, so its
    compaction does no glue and no byte check for claims. The bench's per-byte part
    carried one: a 219 MB copy and its check, in a worker of its own.
  - Bucketing runs 3 jobs at once, not 1, and packing 4, not 1. 7.5.0's timeline put
    2.7 and 0.7 idle core-seconds in those two stages per byte, against a release's last
    wave of jobs.
- Less like a release:
  - Each job starts a fresh worker (`tasks_per_child=1`), at about 0.6-0.9 CPU-s for
    its imports (0.9 measured on the shared development machine). The part per byte now
    has 5 more of these (2 bucketing, 3 packing) and 1 fewer (the glue), for 146 MB of
    claims added. Scaled to a release's bytes, that is about 1,500 worker starts for the
    claims key against its 116 (30 + 86). Modelled, it adds about 18-28 minutes to the
    finalise ETA (4 starts × 0.6-0.9 CPU-s × the scale of 3,667 / 2 runs' bytes / 4
    cores). The glue it no longer carries takes off some of that.
  - A part file is one bucket (a release's is 7.8), and each bucket reads every source
    file, as the three are copies of one.
- Not changed: the buckets (7.4.0's figures), and every other key.

## [7.5.0] - 2026-10-25

### Added

- Where finalise's wall time goes, beside its CPU time (`timeline.py`): over finalise's
  two runs, a process of its own samples every 0.5 s the machine's CPU time (busy,
  iowait, idle), the CPU time of the pipeline's process tree and of each of its
  processes, and the run's disk device's reads, writes and busy time
  (`<run>.cores.jsonl`, in the raw results). It is started before the first run and
  reaped after the second, so its CPU time is in none of their figures.
- Each call phases.py times also records its process, its start (`t0`) beside its end
  (`t`), its depth among the timed calls of its thread, and its leading string arguments
  (`of`: the table and key) (`<run>.phases.jsonl`, in the raw results).
- Each run's `timeline`: finalise's stages as the pipeline's process runs them (each
  table's compaction, the copy's check, the sort's write by table, and for a key sorted
  through buckets its bucketing, bucket sorts and packing, the in-memory keys after
  them), each with its wall time, the cores busy and waiting on I/O over it, the
  core-seconds left idle (the cores × its wall less the machine's CPU time over it), the
  device's busy share, and its worker jobs (how many, on how many processes, running at
  once, the longest, and the time the last ran alone). The comment shows the stages on
  three times the rows and per byte (that run less the first), before and after, the
  jobs in a fold.
- sort_by_id.sort_buckets and sort_by_id.pack_key as stages of phases.py, so the
  timeline can tell the bucket path's steps apart; their own CPU time (the parent's,
  between its workers' results) moves out of `write_table`'s in the table of where the
  CPU goes.

Every figure measured before is measured as it was, hence the minor version.

## [7.4.0] - 2026-10-25

### Added

- The claims key's bucket path as each finalise run took it, measured and recorded:
  its buckets (`sort_buckets`), rows a bucket (`sort_bucket_rows`), the share of its
  rows spilled to fragments rather than read from the sources directly
  (`sort_spilled_share`), and rows a spill flush (`sort_flush_rows`: rows spilled over
  the run's `_ipc_stream` calls), and the same with `_x3` for the run on three times the
  rows. The comment gives them under the ETA, for each commit that has them.
- What the bucket path is here, against a release's (below), in the README.

Every figure measured before is measured as it was, hence the minor version.

### The bucket path's shape, and why the bucket size stays 64 MiB

The bench's claims key is sorted through buckets of 64 MiB of source, as a release's:
2 buckets at the sample's rows, 4 at three times them. A finer bucket size was tried,
to bring the flushes to a release's size, and is not taken: it makes the bench less like
a release where most of the sort's cost is.

The bench's figures are measured (the bench's own claims group file, 12 row groups,
1,175,112 rows, 73.0 MB, through the direct-read plan of `_bucket_source`, #93, with its
fragments written; the 16- and 18-bucket shape also in a whole local run). The release's
bucket count is measured (20260928's main-set claims: 44.7 GB in 30 compacted files, 667
buckets; docs/journal/2026-10-05-sort-speed.md); the rest of its column is modelled, and
given as the range of two models: the dump as 8 shard passes over Q1 to Q140M, each chunk
85k Q numbers (the bench data: chunk 0 is Q8 to Q83751, its ids 99.9% in increasing
order; every routed chunk spans 80k to 95k Q numbers), 62% of its entities in the main
set, a property's 14 rows in each, a chunk's rows in row groups of up to 122,880, 667
equal-row buckets in string order, 30 source files, `_direct`'s plan and the flush rule;
rows per entity uniform (9.5), or heavier for low Q (62 below Q100k, as chunk 0 has, to
6.5 at Q140M, as r50 has). Both come to about 780 million rows, 57 bytes a row (the
sample's claims: 62). #93's own model gives 1.75 reads a row and 27% spilled.

| | bench, 64 MiB buckets (kept): sample; 3× rows | bench, 14 more buckets: sample; 3× rows | bench, 4.6 MB buckets: sample; 3× rows | release, main-set claims |
|---|---|---|---|---|
| bucket count | 2; 4 | 16; 18 | 16; 48 | 667 (measured) |
| rows per bucket | 588k; 881k | 73k; 196k | 73k; 73k | 1.17M |
| rows read directly; spilled | 100%, 0%; 90.2%, 9.8% | 69.5%, 30.5%; 58.8%, 41.2% | 69.5%, 30.5%; 7.0%, 93.0% | 70-80%, 20-30% (#93: 73%, 27%) |
| direct reads a row read directly | 1.21; 1.34 | 2.10; 2.25 | 2.10; 2.54 | 1.13 (#93: 1.21) |
| reads a row | 1.21; 1.61 | 2.66; 2.72 | 2.66; 2.99 | 2.30-2.39 (#93: 1.75) |
| rows per spill flush | none spilled; 26.6k | 16.3k; 21.1k | 16.3k; 18.7k | 12.6k-13.2k |
| source files per bucket | 1; 1 | 1; 1 | 1; 1 | about 16 of 30 |
| in the part per byte: spilled, reads a row, direct reads a row read directly | 14.7%, 1.81, 1.42 | 46.6%, 2.75, 2.35 | 124% (more spilled than the rows added), 3.15 | 20-30%, 1.75-2.39, 1.13-1.21 |
| buckets per row in the part per byte | 1 per 1.18M | 1 per 1.18M | 1 per 73k | 1 per 1.17M |

"The part per byte" is the run on three times the rows less the first (2,350,224 rows),
the part the ETA scales to the release; reads a row are `_direct`'s own accounting (each
bucket reading a row group reads it whole, a row group with rows left is read once more,
and a row spilled costs 2 reads).

Why: a release's row group is one chunk's rows, a narrow id range, so it falls into one
or two of 667 buckets and is read directly once or twice. The sample's key is four
chunks' rows, so a bucket finer than a quarter of the key cuts through a chunk, and its
row groups are read by several buckets each: at 16 buckets, 2.1 direct reads a row, a
release about 1.1-1.2. Its spilled rows, chunk 0's (Q8 to Q83751, half the key's rows,
spread over every bucket, as a release's earliest chunks are) and a property's, go from
15% to 47% in the part per byte, against a release's 20-30%. And it adds 14 buckets'
own costs (a worker process each, a file written with the Hub's settings and read back,
about 0.7 CPU-s a bucket, measured on this key through 2, 16 and 48 buckets) to every
run. A fixed bucket size small enough for 16 buckets at the sample's rows (4.6 MB, 48 at
three times them) spills nearly all the rows added and would also count those costs 16
times per byte: about 2.8 hours more in the finalise ETA. Before the direct reads (#93), when every row was spilled, the 14
more buckets did bring the flushes to a release's (84k and 65k rows a flush at 64 MiB,
33k and 31k with 14 more, a release 26k-33k, modelled); the direct reads take most rows
off that path, and what they cost is set by how many buckets a row group is in.

What the kept shape gets wrong:

- Understated: the spill. 15% of the rows in the part per byte against a release's
  20-30%, and in flushes of 26.6k rows against about 13k, so a cost per flush (an IPC
  stream's per-buffer codec calls, the concatenation of a flush's slices before it is
  written) weighs about half as much a row here. (Flushes of about 5k rows, the
  threshold, are those of a bucket gathering thin slices: a wide row group's, or a
  property's few rows; the model's release flushes average about 13k.)
- Understated: the costs per fragment. A release's bucket has rows from about 16 of its
  30 source files; the bench's key has one at either size (three groups glue into one
  219 MB compacted file), so a bucket reads and checks at most one fragment and one
  source.
- Overstated, a little: direct reads, 1.42 a row read directly in the part per byte,
  a release 1.13-1.21.
- Understated: a bucket's sort is of 588k and 881k rows, a release's of 1.17M.

## [7.3.0] - 2026-10-25

### Added

- sort_by_id._read_bucket, a bucket's rows read from its source row groups directly and
  from its fragments, as a stage (the fragments' IPC reads stay in sort_by_id._read_streams).

## [7.2.0] - 2026-10-25

### Added

- The split from the gz dump (`WIKIDATA_DUMP_FORMAT=gz`), for a commit whose split reads
  it: split-dump then route-release on a gz slice of the same lines (dumpslice.py writes
  it beside the bz2 slice, at level 9 as the gz dump is), run by `pipeline.py --split-gz`
  after each timed run, in a process of its own, so no figure of the timed run (its wall,
  CPU, memory, disk and device writes) takes it in. Its wall and CPU time, peak RSS and
  ETA: its CPU time per byte of the slice's gz, scaled to the gz dump
  (RELEASE_DUMP_GZ_BYTES) over the cores. Also each decoder alone on its slice (lbzip2,
  and the gz decoder the commit's split uses), its wall time scaled the same way. The
  comment shows them against the split from bz2, and the ETA and CPU from gz in the
  table when both commits have them.
- Its own parity: its chunks hold exactly the sample's rows, and its manifests are the
  split from bz2's, entry for entry. It is shown with its figures and kept apart from the
  run's parity (`gz_split_parity_ok` beside `parity_ok`, which is as it was); a commit
  failing it fails the job, as a commit failing the run's parity does.
- The runner installs isal (igzip) and pigz.

Every figure measured before is measured as it was, hence the minor version.

## [7.1.0] - 2026-10-24

### Changed

- The split's stages (dump.write_chunk, dump.reshape, dump.entity_row, dump.route_chunk)
  in the table of where the CPU goes are each the CPU time of the thread they run on,
  not of their process: the split can run its chunks in threads side by side, whose
  process times would each count the others'. What polars hands to its own threads in a
  chunk's write is in no stage, in either arm. The split's CPU and wall are as they were.

### Added

- dump.reshape, the compiled reshape, as a stage.

## [7.0.0] - 2026-10-24

### Added

- The split: the release's first stage, split-dump then route-release (dump.split,
  dump.route), run as the CLI runs them on a slice of a dump made of the sample's own
  entities (dumpslice.py: each chunk row turned back into its dump line, the lines
  bzip2-compressed, made once a job and untimed). Each step's wall and CPU time (lbzip2
  and the workers included), its stages in the table of where the CPU goes, and its ETA:
  its CPU time per byte of the slice's bz2, scaled to the release's dump
  (RELEASE_DUMP_BYTES) over the cores, added to the release's ETA. Its peak RSS: each
  step's process and its own (lbzip2, the workers), summed. The split runs before the
  pipeline, outside its wall, CPU and memory figures.
- Parity also requires each run's split to make exactly the sample's chunks: the
  scholarly works (the `s` files) in the scholarly set, the rest in the main one, row for
  row.
- A new stage in the ETA, hence the major version.

## [6.0.0] - 2026-10-24

### Added

- Where the CPU goes: each run times the pipeline's and finalise's stages (phases.py),
  each stage's own CPU time in its process, its worker's native threads included and the
  stages it calls apart, and the comment shows them by stage before and after, a stage's
  change in bold where it is beyond noise: the pipeline over every chunk, and finalise's
  per-byte part (the run on three times the rows less the first). The stages are wrapped
  by an import hook in the harness (in every worker through phases_site/sitecustomize.py),
  so the pipeline's code is unchanged and either commit is timed the same way.
- The timing is in the timed runs (a JSON line a call, about 7,000 calls a run), hence
  the major version.

## [5.1.1] - 2026-10-24

### Changed

- Every figure is still shown, before and after, but a change is given only when it is
  beyond noise, in bold; within noise its cell is blank (no "≈ ..." in every row). The
  ETA headline is one line, its method and its figures by runner CPU folded into a
  details block under it.

## [5.1.0] - 2026-10-24

### Changed

- Each change is the paired ratio, after over before on one runner, averaged over a
  round's two runners and then over the rounds, given as a percentage with its 95%
  confidence interval ("≈ +0.8% ±5.1%"). Before, changes and the noise test were on
  absolute differences, in which a slow runner's difference counts for more than a fast
  one's; the runners now drawn (EPYC 7763, 9V74, 9V45) differ in speed by up to about
  1.7×. The speed in the headline and the results store's are paired ratios too.
- The full-release ETA is also given by runner CPU, so its absolute figures compare
  like with like across pull requests: their mean over the rounds takes in which runners
  were drawn.

## [5.0.0] - 2026-10-24

### Changed

- Parity compares what a release publishes: finalise's sorted part files, each key's rows
  in order (row count and two sums of position-seeded row hashes) and its Arrow schema.
  They are kept from the first finalise run (KEEP=1) under `final/`. A run that publishes
  no files fails parity.
- The staged group files are compared as multisets: each key's rows in any order, and its
  schema. Their order is the intermediate layout, which a change to processing or to the
  group files may change on purpose (rows cut or reordered by id range), and the
  published files' order checks it wherever it shows: each key is stably sorted by id, so
  a change in the order of an id's rows changes them. Keys whose group files hold the
  same rows in another order are listed (`keys_reordered`), not failed.
- The ETA's finalise run on three times the rows triples the sort's in-memory threshold
  too, so every key is sorted the way the first run sorted it. Before, a third of the
  sample's keys moved to another path at three times the rows (82 → 251 sorted as larger
  keys, a fresh worker process each, its imports about 0.5 CPU-s; 8 more through
  buckets), and that cost, a fixed one per key in the release, was counted as cost per
  byte and scaled to the release's bytes. On CI's runners, worker start-up (imports) was
  half of finalise's CPU samples.
- A change to parity and to the ETA, hence the major version.

## [4.0.0] - 2026-10-24

### Changed

- The Full Wikidata ETA covers the whole run again, finalise included, now measured at
  the release's size rather than scaled by entities (3.0.0) or left out (3.1.1):
  - A key that would be over the bucket threshold in the release (claims) is sorted
    through buckets here too: `SORT_IN_MEMORY_BYTES` is scaled by the sample's fraction
    of the release's entities. Bucket and file sizes are not scaled, so their overheads
    stand as in the release. This changes what "wall, with finalise" measures, hence
    the major version.
  - finalise runs a second time, on the group's files three times over (three groups,
    the same keys, three times the rows). Its time at the release's size is a fixed
    part (per key, the release having the sample's keys) plus the CPU time per source
    byte, from the difference of the two runs, scaled to the release's bytes over the
    cores. The comment says how close the line through the two runs is to the second
    run's wall time.
  - The ETA is that plus the pipeline's stages scaled by entities, each shown. The table
    has the three ETAs and the second finalise run's wall time.

## [3.1.1] - 2026-10-24

### Fixed

- The Full Wikidata ETA extrapolates the pipeline alone, as before 3.0.0. 3.0.0 scaled
  finalise by entities too, which on the benchmark's sample its time does not follow:
  the quick tier's finalise is mostly per key (about 2,300 small keys, each a job of its
  own in compaction and the sort), so it put a release at 66 hours. finalise is given on
  a line of its own, as measured, before and after.

## [3.1.0] - 2026-10-24

### Added

- Prebuilt wheels: a commit whose `bench/wheels` lists `<owner/repo> <run id>
  <artifact>` lines has each of those CI artifacts downloaded into its `wheels/`
  before its environment is made, for a `pyproject.toml` that takes a dependency by
  path from there (its hash pinned in `uv.lock`). A measurement of another
  repository's unreleased change (polars-genson's, pq-glue's) then installs the wheel
  that repository's CI built, in place of building it from a git source in every job
  (9 of a round job's 12 minutes, for polars-genson). The job summary lists the wheels.
  What is measured, and how, is unchanged.

## [3.0.1] - 2026-10-24

### Fixed

- `CPU, finalise` counts the workers of the pool finalise's light jobs share
  (`wikidata.parallel`): each finalise step shuts that pool down when it ends, so its
  workers exit and their CPU time is counted. Without that they outlived the
  measurement, and their CPU went uncounted.

## [3.0.0] - 2026-10-24

2.x results do not compare with 3.0.0: the score now includes finalise.

### Added

- finalise's local stages run after the last group closes, as `main.finalise` runs
  them for a release, claims first: each table compacted (`compact_table`, the group
  files already where its download puts them) and sorted (`sort_by_id`'s sourced,
  written and done stages). Rows: `finalise: compact`, `finalise: sort`, `CPU, finalise`,
  and `wall, with finalise`.
- The Hub is never touched: `HF_HUB_OFFLINE=1`, and every Hub entry point of the
  finalise modules (`download`, `snapshot_download`, `HfApi`, `source_table`,
  `commit_table`, `verify_table`) replaced with one that raises. claims_labels, built
  from both sets' sorted labels, and the cards are not run.

### Changed

- The score and the Full Wikidata ETA are the whole run, finalise included (`wall,
  with finalise`); `wall` is the pipeline alone, as before.

## [2.1.1] - 2026-10-24

### Changed

- Work in progress is not benchmarked: a draft pull request, or one whose head commit's
  subject starts with `wip` or holds `[wip]`, runs only a gate job (seconds) and no
  rounds. Marking a draft ready for review runs the benchmark.

## [2.1.0] - 2026-10-24

The measurements are unchanged: 2.0 results compare with 2.1.

### Added

- The sample's size in every result (`workload.entities`, `workload.source_mb`), and in
  the comment's heading: the quick tier is 33,220 entities, 76 MB of source.
- A throughput row, entities per second.
- A full Wikidata ETA: the release 20260928's 121,815,642 entities through the
  benchmarked stages at each commit's rate on the runner, and the change (≈ when the
  wall time's change is noise). A straight extrapolation; it leaves out downloading,
  splitting and routing the dump, the uploads, and finalise.

## [2.0.1] - 2026-10-24

### Changed

- The results bar shows each job's progress: ⬜ until the job launches, its arrow (➡️
  forward, ⬅️ reverse) while it runs, 🟩 once its half is in. The launch bar is
  unchanged: each job's arrow turns 🟦 as it launches.

### Fixed

- Two comments on a new pull request: jobs starting together could each find no
  comment and each make one. A job that makes the comment now deletes any later ones;
  every job writes to the oldest.

## [2.0.0] - 2026-10-24

1.x results do not compare with 2.0.0: every round's order changed.

### Fixed

- A bias towards the commit after. Each round's order was shuffled from a fixed seed,
  and the three parallel rounds' seeds all came out before, then after: the commit after
  always ran second and gained from it. Comparing identical code showed it 1-3% faster in
  wall time with no change in CPU time (−2.8% ± 0.5% on #35).

### Changed

- A round is a pair of jobs on two runners: forward (the commit before first) and
  reverse (the commit after first), one run of each commit apiece. The round's figures
  are the mean of the two halves, so a change is the mean of the forward and reverse
  changes, and the effect of running first or second cancels. Three rounds are six
  parallel jobs; a round counts once both halves are in.
- The workflow no longer reuses a stored measurement of the commit before (from
  `cache/`, on the same CPU model): runners of one model still differ by 2–9% in wall
  time, against about 1% between runs paired in one job. Both commits always run.
- The pull request comment's bars have a pair of cells a round, its forward job (➡️
  until it happens) and its reverse job (⬅️): `🏁 🟩🟩 🟩⬅️ ➡️⬅️` is one round in and
  one half of another; launched jobs are 🟦. Once every round is in, ✅ takes the 🏁's
  place (❌ if a round failed).

## [1.1.1] - 2026-10-24

### Changed

- The pull request comment's status line has two bars, a cell a round. Launches:
  "⏳ 🟦⬜⬜ launching 1/3 rounds" until every round's pipeline runs have started (a
  `launch` write as each job reaches them), then "🚀 🟦🟦🟦 launched 3/3". Results:
  "🏁 🟩⬜⬜ running: 2/3 rounds remaining", then "🏁 🟩🟩🟩 ✅ done: 3 rounds in <the
  longest round, launch to result>" (🟥 for a round that failed).

## [1.1.0] - 2026-10-23

The quick tier measures exactly what 1.0.0 did, so 1.0.0 results compare with it.

### Added

- Tiers. `quick` is the pull request and master check: 4 chunks, 4 workers, one group.
  `full` is the campaign's 66-chunk workload: the 11 files cycled 6 times, 6 workers
  taking 8 chunks each, groups of 22 merged while chunks are still processing. It runs
  from the Actions tab, with a `before` input to set the commit to compare against.
- A progress bar on the pull request comment's status line, a cell a round, filled as
  rounds come in (not as they start): ⏳ ⬜⬜⬜ until the first, then 🏁 🟦⬜⬜, 🏁 🟦🟦⬜,
  and 🏁 🟩🟩🟩 ✅ once done (🟥 for a round that failed).
- The harness version (`bench/pyproject.toml`) and this changelog. Every result and
  record carries the version and the tier.

### Changed

- The results store keeps one append-only ledger, `records.jsonl`, of every tier and
  version; `RECORDS.md` is a view of it, one table per major version and tier. Raw
  results are in `runs/<tier>/<commit>.json`.
- The pull request comment's opening table takes the commit before's figures from the
  store only when a compatible harness version (the same major version) measured them.
- The cache key includes the harness version.

## [1.0.0] - 2026-10-23

Everything recorded before versioning, tagged 1.0.0 in the store.

### Added

- The benchmark in GitHub Actions: a commit against the commit before it, each on its
  own lockfile, on 4 chunks (`c0`, `r50`, `s30`, `s90`), 4 workers, one group.
- 3 rounds as parallel jobs, each round both commits on one runner, so each change is
  paired within a round.
- Output parity by fingerprint (row hashes per partition key, schemas, quarantine,
  sidecars but `file_size`).
- The pull request comment: one per pull request, edited in place; pending as a run
  starts, then the rounds in as they land; figures as mean ± standard deviation, changes
  paired per round, `=` for an unchanged figure and `≈` for a change inside the 95%
  confidence interval of its rounds' changes; times as durations and sizes in the largest
  unit the value before reaches.
- The results store on the `bench-results` branch (records, raw runs, a cache of
  commits' measurements by CPU model), appended to by master pushes.

[3.1.1]: https://github.com/lmmx/wikidata-pq/pull/43
[3.1.0]: https://github.com/lmmx/wikidata-pq/pull/48
[3.0.1]: https://github.com/lmmx/wikidata-pq/pull/42
[3.0.0]: https://github.com/lmmx/wikidata-pq/pull/41
[2.1.1]: https://github.com/lmmx/wikidata-pq/pull/40
[2.1.0]: https://github.com/lmmx/wikidata-pq/pull/38
[2.0.1]: https://github.com/lmmx/wikidata-pq/pull/37
[2.0.0]: https://github.com/lmmx/wikidata-pq/pull/36
[1.1.1]: https://github.com/lmmx/wikidata-pq/pull/35
[1.1.0]: https://github.com/lmmx/wikidata-pq/pull/34
[1.0.0]: https://github.com/lmmx/wikidata-pq/pull/33
