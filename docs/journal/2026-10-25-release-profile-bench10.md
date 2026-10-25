# 2026-10-25: Where a release's time goes under bench 10.0.0

Continues docs/journal/2026-10-25-release-profile.md (#116, bench 7.5.0) and
docs/journal/2026-10-25-claims-weights.md (#128, bench 10.0.0). Scripts are in
`2026-10-25-release-profile-bench10/`.

## Current State

- Docs only: nothing in `src/`, `reshape/` or `bench/` changed.
- **Master** is 4e58c15 (#127 merged).
- **The modelled release** (bench 10.0.0) is about **10h42m**:
  - the split 2h38m;
  - the pipeline 5h04m;
  - finalise 3h00m.
- #116's profile ranked costs under bench 7.x, which scaled the claims by entities. Its
  hours are stale. This entry re-ranks every timed stage under bench 10.0.0's scaling.
- **The judgement:** nothing untried and in bounds is clearly worth more than about 2–3%
  of a release (see Hypotheses). The rest of the large costs are:
  - floors (the bz2 decode, `_check_written`);
  - owned by running work (the claims built as Arrow in the split);
  - waiting on the user's two pending decisions;
  - or overstated by the model itself.

## Sources (measured)

Every figure below comes from the bench's own artifacts: each run's results JSON, phases
log and split-from-gz JSON. The comments' summaries hold only each stage's mean. The
artifacts give each call's process, table and enclosing claims call, which the
re-ranking needs.

| run | arms | runs | bench |
|---|---|--:|---|
| 40205337787 (#128) | fde78af, db7f2d7 | 12 | 10.0.0 |
| 40206252718 (#128's follow-up) | fde78af, 1b6dc06 | 12 | 10.0.0 |
| 40206699442 (#127's re-bench) | a0ca3d9, af58559b | 12 | 10.0.0 |

- **Same code in every arm:** outside `reshape/` all six arms run master's `src/`, so the
  pipeline's and finalise's rows pool all 36 runs.
- **The split:** its rows come from #127's "after" arm (af58559b, now master's
  reshape), 6 runs.
- **Intervals:** the ± figures are 95% intervals over the runners (EPYC 7763, 9V74,
  9V45). The weights' own systematic uncertainty is in CHANGELOG 10.0.0.

## How a stage's hours are reckoned (modelled)

The decomposition follows bench 10.0.0's model term by term (`bench/pipeline.py`,
`claims_model`, `release_model`; `bench/release.py`). For each run, the parts sum to the
harness's own pipeline, finalise and split ETAs to within 0.2%.

**The pipeline:** hours = a stage's own CPU × the run's wall/CPU × its scale.
- **The claims' calls:** a call inside `normalise_claims_direct` or `place_unsplit` (in
  its process, by start and end time) is scaled by its chunk's set's claims rows:
  1,172× for the main set and 1,960× for the scholarly set. Its chunk is the k-th one
  its worker finished, as `claims_measured` assigns them.
- **Everything else:** scaled by entities (3,667×).
- **CPU outside the timed stages:** `cpu_total` less the stages, mostly the workers'
  starts. Also scaled by entities.

**Finalise:**
- **A stage's per-byte part:** each stage's own CPU by table, using `phases.table_cpu`'s
  rule (a call's table is the one it names, or else that of the innermost named call
  enclosing it). The 3× run less the first is halved and divided by the 4 cores, then
  scaled by the claims' bytes (1,322×) for the claims and by entities for every other
  table.
- **The per-byte CPU outside the timed stages:** taken at the blended scale, as the
  model takes it.
- **Not scaled:** the fixed part (per key) and the release's claims workers' starts, as
  the harness has them.

**The split:** its CPU × the dump's bytes over the slice's (103.5 GB over 77.3 MB) over 4
cores. Its parts:
- lbzip2's CPU decoding the slice alone (`bz2_decode_cpu`, a separate run);
- `dump.reshape` and `dump.write_chunk` (each its thread's CPU);
- route-release;
- the rest: the split's CPU less those. That is Polars' write threads (zstd over the
  JSON), reading the decoder's lines, `is_scholarly` and the process's start.

## The ranking: modelled hours a release

Bench 10.0.0, master. The last column is a rough reading of what a release would see.
It is not a model, just the direction and size of what CHANGELOG 10.0.0 calls "still
uncertain", with this entry's own checks (below).

| # | stage | span | modelled | ± | share | how sure | at a release, likely |
|--:|---|---|--:|--:|--:|---|---|
| 1 | lbzip2's decode | split | 1h46m | 14m | 16.6% | sure: measured alone, by bytes | holds |
| 2 | CPU outside the timed stages (worker starts) | pipeline | 1h21m | 2m | 12.7% | **overstated:** entities stand for ~14,700 starts, a release has ~2,900 | ~16m |
| 3 | `partition_parquet` (labels, descriptions, aliases, links) | pipeline | 1h05m | 2m | 10.1% | fair: per file, and its files per entity hold (below) | ~1h |
| 4 | `normalise_in_schema`, claims (main 20m, scholarly 38m) | pipeline | 59m | 1m | 9.2% | sure: by rows, ±2 points (CHANGELOG 10.0.0) | holds |
| 5 | `write_table`, claims (main 18m, scholarly 27m) | pipeline | 45m | 1m | 7.2% | sure: by rows | holds |
| 6 | `_write_file_parallel`, claims | finalise | 40m | 1m | 6.3% | sure: by bytes | holds |
| 7 | the split's rest: Polars' write threads, line reads, `is_scholarly`, start-up | split | 36m | 6m | 5.6% | sure, by bytes, but for its start-up (~1 CPU-s, ~5m), which a release pays once | ~31m |
| 8 | `_check_written`, claims | finalise | 21m | <1m | 3.3% | sure: by bytes | holds |
| 9 | `_read_bucket`, claims | finalise | 20m | <1m | 3.3% | sure: by bytes | holds |
| 10 | `merge_group` | pipeline | 18m | 1m | 2.9% | fair: per file, but its claims glue (bytes) is scaled by entities | ≤18m |
| 11 | `normalise_in_schema`, labels, descriptions, aliases | pipeline | 15m | <1m | 2.4% | **overstated:** per row; the sample has ~3× a release's labels an entity | ~5–10m |
| 12 | `_check_glued`, every other table | finalise | 12m | <1m | 2.0% | **overstated 1.5×** by the 3× run's shape (below) | ~8m |
| 13 | `_sorted`, claims | finalise | 10m | <1m | 1.6% | sure: by bytes | holds |
| 14 | `dump.reshape` (#127's) | split | 10m | 1m | 1.6% | **understated:** the slice's lines are already in orjson's form | ~15m |
| 15 | `_read_key`, every other table | finalise | 10m | <1m | 1.6% | overstated: per row | ~4m |

The next rows are each 1.6% or less:
- `_write_file_parallel` of the other tables, 10m;
- `_write_run`, 8m;
- `atomic_sink_parquet`, 8m;
- `normalise_sitelinks`, 7m;
- finalise's per-byte CPU outside the stages, 6m;
- `_bucket_source`, 6m;
- `_check_written` and `_sorted` of the other tables, 6m and 5m;
- `_key_schema`, 4m;
- the release's claims workers' starts, 3m;
- route-release, 2m;
- `dump.write_chunk`, 2m.

The top 15 are 86.5% of the modelled release.

**By table:**
- The claims are about 1h44m of the pipeline and 1h44m of finalise's per-byte part.
- Finalise's other tables are 1h05m modelled. Scaled by their rows instead (the main
  set's from the 20260507 build, the scholarly set's per entity as in the sample), they
  would be about 27m. That is a lower bound, because their glue and byte check go per
  column chunk, not per row.

**Adjusted for rows 2, 11, 12, 14 and 15 and finalise's other tables:** a release would
take nearer **9½ h** than 10h42m. That is a rough reading, not a model; bench 11.0.0
(running) will model it properly.

### Checks behind the last column

- **Worker starts (row 2):**
  - **Measured:** the sample's CPU outside the stages is 4.39 s, for 4 workers.
  - **Modelled:** a release starts ~2,900 workers (23,035 chunks, 8 a worker). At
    1.1 CPU-s a start and the runs' wall/CPU of 0.305, that is about 16m.
- **Partition files (row 3):** each costs the sink about 1 ms (3.49 CPU-s for the
  sample's 3,431 files).
  - **Main set:** the bench's 5 routed main-set files average 851 partition files for
    4,217 entities. A release's main chunk has 6,192.
  - **Scholarly set:** 98 for 5,783.
  - **So a release** writes about 11–12M partition files (12,182 and 10,853 chunks),
    against the 12.6M that entity scaling stands for. Entity scaling holds for this row
    to within about 10%, unlike the per-row rows.
- **The 3× run's shape (rows 12, `_write_run`, `_key_schema`):** in the first run, every
  other key has one group file, which is linked. In the 3× run it has three, which are
  glued and checked. So the glue's per-byte figure is the whole of the 3× run's glue
  over twice the source, 1.5 times its cost a byte. At a release, every such key has
  ~30 group files and is glued.
- **Reshape (row 14):** on real dump lines (non-ASCII escaped, `/` as `\/`), #127's
  reshape runs at 0.52 of its predecessor's CPU, against 0.41 on the slice (#127's
  comment). So a release's reshape is about 15m, not 10m.

## What has been tried, by row

From lmmx/wikidata-pq#97 (Kept, Rejected, Merged) and `docs/journal/`.

| # | stage | tried, and the result |
|--:|---|---|
| 1 | lbzip2 decode | **Kept:** lbzip2 is the fastest exact decoder (7-Zip, rapidgzip, indexed_bzip2 and bzip2 measured, #116). **Merged:** the download over 3 connections (#109), the split following the download (`dump._feed`). **Option:** gz from a mirror, now ~44m for the whole split (58m before #127). |
| 2 | worker starts | **Merged:** persistent workers, 8 chunks a worker (2026-10-22). **Rejected:** #107, overlapped worker start-up (a bench-only gain). **Running:** bench 11.0.0 will count starts as fixed costs. |
| 3 | `partition_parquet` | **Merged:** the sidecar from the frame's stats (#89); sidecars from footers. **Rejected:** zstd 1 or uncompressed, partitions written from Python, sinking a collected frame. **Pending the user:** one file per table (#106, partition CPU −38%), blocked on decoupling the sort's thresholds. |
| 4 | `normalise_in_schema`, claims | **Merged:** known schemas (#57); genson 1.5.0 (#102). **Rejected:** genson's internals (genson #319: SIMD lexing, skipping pruned subtrees, interning keys, the integer double-parse, walking members); 64-bit offsets (genson #314); the decoder's columns grown to the batch's length (genson #317, #121). **Running:** the claims built as Arrow in the split (#129's study). |
| 5 | `write_table`, claims | **Merged:** genson's writer (2026-10-24). **Rejected:** id-only statistics (#94, pq-glue #9); a single-threaded write; row groups of 32k or 1M; pq-glue's writer in place of genson's. **Pending the user:** shared repetition levels upstream (−8.8%). |
| 6 | `_write_file_parallel`, claims | **Merged:** pq-glue's parallel writer (#45). **Rejected:** Hub files at zstd 1 (#86); the vendored arrow-rs fork (#120). **Pending the user:** shared repetition levels upstream (−6.0%). |
| 7 | the split's rest | **Rejected:** the chunk files' write settings: zstd 1, lz4, no statistics, pyarrow without dictionaries, all within noise (#116). |
| 8 | `_check_written` | **Kept:** it is at its floor, one independent decode of every written byte. **Rejected:** #114, a column at a time. **Merged:** a bucket's check read on its worker's thread (#122). |
| 9 | `_read_bucket` | **Merged:** direct reads (2026-10-25); a row group wholly the bucket's kept as read (#112, −15%). **Rejected:** pq-glue's read (#110); zero-copy slices (#115); one thread (#122). |
| 10 | `merge_group` | **Merged:** pq-glue's glue for every table (2026-10-23). **No change:** one aggregation per key (#55). **Pending the user:** one file per table (#106, merge CPU −17% to −20%). |
| 11 | `normalise_in_schema`, the maps | **Rejected:** "the label, description and alias calls". Known schemas (#58) made no measurable change. |
| 12 | `_check_glued` | **Merged:** a byte comparison without decoding (2026-10-24). Nothing since. |
| 13 | `_sorted` | **Merged:** a column at a time, rows freed as taken (#118, −20%), with its pages returned (#122). **Rejected:** run slices (#54). **Not tried:** a k-way merge (it would save at most the sort indices, 0.06 s a bucket). |
| 14 | `dump.reshape` | **Merged:** #127, one scan a line, copied in runs (−65% on the slice; the split ETA −19.6m ± 9.0m on the same runners, −10.8% ± 3.7%). |
| 15 | `_read_key` | **Merged:** one open a file; pq-glue for large keys. **Rejected:** Polars as the reader (2026-10-24). |

## What changed since #116's profile

| | #116 (bench 7.5.0) | now (bench 10.0.0) |
|---|---|---|
| `normalise_in_schema` (all tables) | 3h04m, 1st | 1h14m |
| `_write_file_parallel` (all tables) | 2h22m, 2nd | 50m |
| `write_table` | 2h14m, 3rd | 45m |
| lbzip2's decode | 1h42m, 4th | 1h46m, 1st |
| `partition_parquet` | 1h22m | 1h05m, 3rd (2nd once worker starts are corrected) |
| `_read_bucket` | 1h15m | 20m |
| `_check_written` (all tables) | 1h04m | 27m |
| `_sorted` (all tables) | 48m | 16m |
| `merge_group` | 25m | 18m |

- **The claims rows fell by about two thirds,** with the correction.
- **What did not fall:**
  - the split, which is scaled by bytes;
  - partitioning, which goes per file and was never claims;
  - the CPU outside the stages.
- **arrow-rs's two Parquet writes** are now 1h35m, 15% of a release, against 4h36m (27%)
  before.

## The pending decisions, valued under bench 10.0.0 (modelled)

- **Shared repetition levels upstream:** about 7m, 1.1% of a release, against #120's
  ~20m under bench 7.x.
  - **How it is reckoned:** 8.8% of `write_table` (claims only, 45m) is 4m; 6.0% of
    `_write_file_parallel` (all tables, 50m) is 3m.
  - **Its other value:** the −2.5% in finalise's peak RSS is unchanged.
- **Decoupling the sort's thresholds,** which would let one file per table ship: about
  28m, 4.4% of a release.
  - **How it is reckoned:** 38% of `partition_parquet` (65m) is 25m; 17–20% of
    `merge_group` (18m) is 3–4m. #106's extra in labels' larger keys is in finalise's
    fixed part, seconds at a release.
  - **Why it is the more robust figure:** these costs go per partition file, and entity
    scaling holds for them, unlike the per-row stages.
  - **So** it is now the largest saving that waits on a decision.

## Hypotheses (untried, in bounds)

Excluded:
- anything in Rejected;
- the pipeline overlapping the split (deferred);
- anything behind the pending decisions;
- the claims built as Arrow in the split (running).

### 1. Keys sorted in memory read straight from their group files: compaction's glue and byte check skipped

**Mechanism:**
- **Today:** for every other table, compaction glues each key's group files into
  compacted files (`_write_run`: `pq_glue.concat`). It then compares the copy with them
  byte for byte, column chunk by column chunk, in Python (`_check_glued`). The sort then
  reads those files whole (`_read_key`).
- **The cost at a release:** each such key has ~30 group files. Every byte is copied once
  and read back twice, only to be read a third time by the sort.
- **The change:** link each group file of a key the sort will sort in memory as its own
  compacted file (`_link_run`, as is done today for a key of one group file). The sort's
  `_read_key` already reads and checks the schemas of several files. The claims, and
  any key over `SORT_IN_MEMORY_BYTES`, keep today's path.

**Expected saving:** about 12–20m, 2–3% of a release.
- **Modelled:** under bench 10.0.0, `_write_run` (8m), `_check_glued` (12m) and
  `_key_schema` (4m) come to 24m. Without the 3× run's 1.5× (above), that is about 16m.
- **Cross-check:** at 33 µs a column chunk checked (1.68 CPU-s for the 3× run's ~51k),
  a release's ~12M partition files and ~5 columns each give ~8m checked and ~5m copied.
- **Less:** ~2m for `_read_key` opening ~30 files a key, not one.
- **Plus, not modelled:** the other tables' bytes are not written again or re-read
  twice. At a release that is at least ~15 GB (the main set's, from the 20260507 build;
  the scholarly set's are not on record), so about 1.5–6m of disk-bound wall at
  200–500 MB/s.
- **The bench would measure it:** the 3× run's glue goes, so finalise's per-byte part
  falls.

**Risk to the data contract: medium.**
- **The rows and their order** are the same.
- **Three sort decisions** read the compacted files:
  - the part count, `ceil(size / COMPACT_FILE_BYTES)`;
  - the in-memory threshold, `SORT_IN_MEMORY_BYTES`;
  - `_row_group_rows`' sample, the first row group of the largest file.
- **What changes with group files:** the size becomes the group files' sum (each with
  its own footer), and the sample can come from another row group.
- **The condition:** to keep the published files byte-identical, both must be computed as
  the glued files would have them:
  - the runs from `_runs`, and the sample from the largest run's first file;
  - the glued size exactly. That probably needs pq-glue to give a glue's footer size
    without writing it.
- **Otherwise** it falls under the pending threshold decision, and should wait for it.

**Checks and restart safety:**
- **Restart:** linking is an existing compaction path (`files.jsonl`).
- **`check_compacted_copy`** covers the linked files as it does now.
- **`_check_glued`** stays for everything still glued (claims packing). A link is the
  same bytes by construction, as `_link_run`'s docstring argues.

**Files:**
- `src/wikidata/compact.py` (`rewrite_table`'s planning);
- `src/wikidata/sort_by_id.py` (`sort_in_memory`, `_read_key`: the size and sample as
  the glued files');
- possibly pq-glue (a glue's size);
- `scripts/test_sort.py`, `scripts/test_finalise.py`.

### 2. Pipeline workers that take more chunks

**Mechanism:** raise `CHUNKS_PER_WORKER` from 8 to 32, or never replace a worker. A
release's ~2,900 worker starts become ~720, each about 1.1 CPU-s of imports and first
touches.

**Expected saving:** about 1–10m, at most 1.5% of a release.
- **Upper bound:** ~2,200 starts × 1.1 CPU-s at the pipeline's wall/CPU, about 12m.
- **Measured:** on 66 chunks, workers never replaced against 8 a worker came out at
  0.2–1.6% of the pipeline's wall (docs/journal/2026-10-22-persistent-workers.md). That
  is 1–5m at a release.
- **The bench can't see it:** the quick tier's 4 chunks start the same 4 workers either
  way. Bench 11.0.0's start-up as fixed cost would model it.

**Risk to the data contract:** none. Outputs were byte-identical at 1 and 8 a worker.
- **Memory:** a worker holds its recent chunks' peak (2.5–2.9 GB after a c0-like
  chunk). There was no growth over 16 chunks.

**Files:** `src/wikidata/config.py`.

No third hypothesis clears 0.5% of a release (below).

## Considered, and not proposed

- **A shared row group's other rows not decoded** (`_read_bucket`, a page-pruned
  selection over [lo, hi)): about 1–2m.
  - **Why so little:** at a release a directly read row is read 1.13–1.21 times. Only
    the pages wholly outside the bucket could be skipped, and pq-glue's reader was no
    cheaper than pyarrow's (#110).
- **Fewer spilled rows** (smaller claims row groups for short-id chunks): unmodelled,
  but they change `_row_group_rows`' sample, which belongs with the pending threshold
  decision.
  - **Why it may matter:** the release's spill is modelled at 20–30%, against the 3×
    run's 10%, so `_bucket_source` (6m) is likely understated.
- **The claims' packing pass removed** (one part file a bucket): it removes a copy and
  byte check of the claims (~96 GB at a release), disk-bound, maybe 2–7m. But it changes
  the published layout (667 files of ~65 MB against 86 of 500 MiB), and it belongs to
  the queued disk-bound passes, which need a release-scale disk run.
- **One file per table with Polars' own row groups glued into it:** this keeps the
  group files' bytes, so it would not need the threshold decision.
  - **Why not:** it saves only the file-system share of `partition_parquet` (about a
    fifth, ~13m). Writing each key from Python cost as much as the sink or more
    (docs/journal/2026-10-25-partition-costs.md), so the net is likely ~0.
- **The split's start-up and route-release** (~1 CPU-s on the slice, ~5–7m as modelled)
  are fixed costs scaled by bytes. Removing them is a model correction for bench/, not a
  speed-up.

## Judgement

- **Nothing left in bounds is clearly worth more than 2–3% of a release.**
- **The one hypothesis near the bar** (1, 2–3%) is worth it only if the glued files'
  sizes can be reproduced exactly. Otherwise it waits on the same decision as one file
  per table.
- **Where the large costs sit:**
  - **The bz2 decode (1h46m):** at its floor. gz from a mirror is the route.
  - **The claims' JSON decode and the pipeline's Parquet write (1h44m):** what the
    running split work targets.
  - **The pending decisions:** decoupling the sort's thresholds is worth about 28m and
    shared repetition levels about 7m, modelled.
  - **The model itself:** about 1h05m of overstated worker starts, and roughly another
    half hour of per-row overstatement in the other tables. Bench 11.0.0 (running) will
    move the modelled release more than any code change left here.

## Missing

- **No release-scale disk run:** finalise's disk-bound passes are not sized. For the
  claims, at a release:
  - the compacted copy's two hashes;
  - packing's copy and byte check.
- **Not on record:** the scholarly set's other-table rows and bytes. The rows-weighted
  figure for finalise's other tables assumes the sample's rows an entity.
- **The bench's 3× run** links every other key in its first run and glues it in its
  second. So it measures compaction's glue at 1.5 times its cost a byte. That belongs
  with bench 11.0.0's other-table weights.
