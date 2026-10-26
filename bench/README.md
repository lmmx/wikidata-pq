# Benchmark

A fixed workload of the publication pipeline, run in GitHub Actions: a commit against
the commit before it, on the same runner in the same job.

| run | before | after | result |
|---|---|---|---|
| a pull request that changes the pipeline | trunk: its merge base with master | its head | a comment on the pull request, updated by each push |
| a push to master | the previous master commit | the new one | a row appended to the `bench-results` branch (see [RECORDS.md](RECORDS.md)) |
| the Benchmark workflow, from the Actions tab | the commit's parent | any commit, branch or tag | the run's summary |

## The score

Speed is the commit before's wall time over the commit after's, one run of each:
1.10× (−9.1%) means the change took 9.1% off the wall time. A change that does not touch
the pipeline's speed reads about 1.00×, within the runner's noise. Absolute times move
with the runner's hardware, so they are recorded but not scored.

## Two tiers

- **quick** (every pull request and master push): the split's parts of 4 chunk files (5
  chunks: 2 of the main set, 3 of the scholarly set), 4 workers, two groups. A fast
  regression signal. It runs the real pipeline on real data, but not at the scale of a
  release: each worker takes one chunk, and the one group merges after processing, so
  worker reuse and merges overlapping processing are not exercised. It is not, by
  itself, evidence of full-pipeline throughput.
- **full** (from the Actions tab, tier `full`): the configuration of the campaign's
  66-chunk A/Bs: the 11 files cycled 6 times (since bench 12.0.0, the split's 7 parts of
  them cycled 6 times: 42 chunks), 6 workers taking 8 chunks each, groups of 22 merged
  while chunks are still processing. Its `before` input sets the commit to
  compare against: the campaign's reference is `2d25be6`, the last commit before #25.
  Its records are tagged with its tier, as every record is with its tier and harness
  version (`pyproject.toml`, `CHANGELOG.md`): results compare only within a tier and a
  major version.

### Does quick predict full?

Both tiers measured the campaign's two big changes, three rounds each (bench 1.1.0,
2026-10-24; the full runs are in the results store):

| change | quick | full |
|---|---|---|
| #25, `2d25be6` → `c69dcac` (merges, contention) | 1.77× ± 0.04, group merges ÷17.2 | 1.87× ± 0.11, group merges ÷23.7 |
| #26, `c69dcac` → `6090f4f` (partition writer, worker memory) | 1.36× ± 0.06, partition CPU ÷5.4, peak RSS −46% | 1.54× ± 0.01, partition CPU ÷5.1, peak RSS −44% |

Quick gets the direction and the mechanism of both right: the stage each change speeds
up, by about the same factor, and the same memory effect. It understates the end-to-end
speed-up, by 5% for #25 and 12% for #26. At full scale, merges and partitioning are a
larger share of the wall time, and the overlap of merges with processing and worker
reuse are exercised. So a quick result is a regression signal and a lower bound on
a change to those stages; a claim about release throughput needs a full run.

## The workload

- **Input:** 4 chunks, one each of the benchmark data's `c0` (a main-set chunk of 10,000
  entities), `r50` (a routed main-set chunk), `s30` and `s90` (scholarly-article chunks).
  The list is `FILES` in `ab.py`: as the pipeline gets faster, more of the 11 files make
  a bigger workload, and a new series of results.
- **Pipeline:** `pool.process_chunks` with the real `process` and `partition_chunk` for
  each chunk, 4 workers (the runner's cores), as `process-wikidata` runs a release.
- **Group:** the 4 chunks (the split's 5 parts) are one group, closed as `close_group`
  does it: the ledger's stages, `merge_group`, and the staging files' clean-up. The
  tier's group is a share of its chunks, applied to the parts (since bench 17.0.0).
- **finalise:** after the last group closes, its local stages as `main.finalise` runs
  them for a release, claims first: each table compacted (`compact.compact_table`, the
  group files moved, as each group closes, to where its download puts them) and sorted
  (`sort_by_id`'s check, write and metadata stages). Timed as `finalise: compact` and
  `finalise: sort`; the score is the whole run, `wall, with finalise`.
  - **The claims key's buckets:** a key the release would sort through buckets (claims)
    is sorted through buckets here too: the in-memory threshold is scaled by the
    sample's fraction of the release's entities, and the bucket size is the release's,
    64 MiB of source (2 buckets at the sample's rows, 4 at three times them). The part of
    finalise that grows with the rows has the release's buckets per row, and about its
    reads per row (each source row group read directly by the one or two buckets its
    rows are in: 1.81 reads a row against 1.75-2.39 at a release, modelled). It spills
    less than a release (15% of its rows in that part against 20-30%) and in larger
    flushes (27k rows against about 13k), and its buckets each have one source file, a
    release's about 16. Finer buckets would cut through the sample's four chunks, each
    a large share of its key, and read each row group several times. The figures, and
    how the release's are modelled, are in CHANGELOG.md (7.4.0). Each run records the
    shape it measured (`sort_sources`, `sort_parts`, `sort_buckets`, `sort_bucket_rows`,
    `sort_spilled_share`, `sort_flush_rows`, and `_x3` for three times the rows), shown
    under the ETA.
  - **The claims key's files, on three times the rows:** a release's claims group file
    is larger than the compaction target (44.7 GB in 30, against 500 MiB), so each is a
    compacted file of its own, linked, and a bucketing job; its part files are 500 MiB
    each, about 2.9 a group file. The run on three times the rows sets the target
    (`WIKIDATA_COMPACT_FILE_BYTES`, `x3_compact_file_mb`) in that ratio to the claims
    group files' mean size, so its claims key is 3 linked source files (3 bucketing jobs)
    and 4 part files (4 packing jobs, one a bucket: a part file is whole buckets), where
    it was one glued file and one part file. Every other key's three times is far under
    the target (`x3_other_key_max_mb`), so they are compacted and sorted into files as
    before; the first run is as before. The shape against a release's is in CHANGELOG.md
    (8.0.0).
- **The Hub is never touched:** `HF_HUB_OFFLINE=1`, and every Hub entry point of the
  finalise modules (`download`, `snapshot_download`, `HfApi`, `source_table`,
  `commit_table`, `verify_table`) is replaced with one that raises; the workflow holds no
  `HF_TOKEN`.
- **The split:** before the pipeline, `split-dump` then `route-release` on a slice of a
  dump made of the sample's own entities (`dumpslice.py`), its lines written as the dumps
  write theirs (every non-ASCII character as `\uXXXX`, `/` as `\/`, floats in PHP's
  form; since bench 11.0.0, before which they were in orjson's form), checked to give
  back exactly the sample's rows. After the timed run, and outside its figures, the split
  from the gz dump, and the split on the slice in orjson's form (bench 10.0.0's), so the
  change shows on the same runs (`orjson_split_*`, `split_eta_10`).
- **claims_labels' refs stage** (since bench 14.1.0): `collect_refs_stage`, as a release
  runs it once the claims are sorted, after each finalise run, on its sorted claims (with
  the split's refs sidecars, for a commit that writes them). It is timed apart: its wall,
  CPU and peak RSS, its time at a release (modelled as finalise's part per byte, by the
  claims bytes), and the end to end and ETA with it are figures of their own, and every
  figure before it is as it was.
- **The split and both sets' pipelines, as a release runs them** (since bench 17.0.0,
  `overlap.py`): after the timed run, and outside its figures, on its dump slice with a
  stand-in Hub on disk and a group a chunk: the split, then each set's `main.run`, one
  after another; and, for a commit whose runs follow the split (`dump.SetManifest`), all
  three at once as `just overlap-release` runs them (nice 10, half the workers each).
  Measured: each mode's wall and the peak RSS of the whole process tree. Modelled, apart
  from them: the groups' bytes at a release (scaled by each set's chunk-file bytes)
  uploaded at 7 MB/s, and how much of that a
  release's split could hide (CHANGELOG.md 17.0.0).
- **Over an emulated link** (since bench 17.2.0, `link.py`): after those, the stand-in
  Hub's transfers sleep for their bytes at a release's size over a release's rate, in
  the sample's time, one link each way, so the sample's transfers are about the share of
  its wall a release's are of a release's. Measured on the sample: the overlapped runs
  again, their pushes over a 7 MB/s link; and `main.finalise`'s three runs (main,
  scholarly, main), as `just release` runs them, on the overlapped runs' output through
  the stand-in, without the link and over a 7 and a 60 MB/s link: the group files
  downloaded from the build branch, compaction, the sort, each table's commit and
  verification, claims_labels' build (not the cards). Whatever order the commit's
  finalise runs its transfers in, the stage runs them in, so a change to it (its commits
  under its compute) shows as a measured wall (CHANGELOG.md 17.2.0).
- **A release's wall, modelled** (since bench 17.1.0, `overlap.release_wall`): from the
  run's stage ETAs and measured figures, nothing measured, a release as its recipes run
  it: the download (14.5 MB/s, the split following it), the split and both sets'
  pipelines at once for a commit whose runs follow the split (the split at its cores on
  the sample, its CPU time its ETA's, the pipeline at nice 10 in the cores it leaves and
  its CFS share of the rest, then on every core: since bench 17.2.0), the uploads at
  7 MB/s as a release's 60 groups close, and finalise and its claims refs once both
  runs' uploads are done. Since bench 17.2.0 also at 7, 30, 60 and 120 MB/s with
  finalise's own transfers: its download of the group files again and its published
  upload, as the commit orders them, and claims_labels' build. Shown beside the serial
  ETA, which it does not replace, with the critical path and by how much (CHANGELOG.md
  17.1.0, 17.2.0).
- **claims_labels' build, compaction and sort** (since bench 16.0.0):
  `build_claims_labels`, `compact_table` and the sort's local stages, as a release runs
  them once both sets' labels are sorted, after each run's refs stage, with this run's
  labels standing for both sets'. Planned with a release's settings (in-memory threshold,
  part size), not the sample's scaled ones. It is timed apart: its wall, CPU and peak RSS,
  and its published files' rows, MB, files and keys, are figures of their own, and every
  figure before it is as it was.
- **A chunk whose claims misfit claims.avsc** (since bench 17.3.0, `misfit.py`): after
  the timed run, and outside its figures. The sample's chunks all fit claims.avsc (it was
  inferred from them), but about 1 in 15 of a release's chunks has a claims member it may
  lack; such a chunk is reshaped twice in the split, its part's claims written as JSON,
  and decoded in the pipeline by genson's inference. Here `c0`'s lines, one reference
  given a globe-coordinate snak (a real dump's misfit), go through `dump.write_chunk` and
  the pipeline's `process` step on the misfit's part, in one process: their CPU and wall
  are measured (`misfit_*`), and the part's claims file must be what genson's inference
  makes of its claims (the misfit chunk's parity).
- **A new claims field found mid-run** (since bench 17.9.0, `evolve.py`): after the misfit
  chunk, and outside every other figure. `c0`'s lines, one statement given a mainsnak
  field no schema has, measure what the release schema's evolution costs
  (docs/journal/2026-10-26-schema-evolution.md), each step in a process of its own (CPU,
  wall, peak RSS): the process step that finds the field against the same step once it is
  known, the split with it in the release schema (the part built) and without, a version 0
  claims file written in version 1 (a merge's widening, and its transient disk), and the
  file read as the sort reads it, with and without the conform to version 1
  (`evolve_*`). Its parity: the value kept, and each widened file the later version's
  rows. A commit without a release schema halts at the field (`evolve_halts`).
- **The split reading the dump as synchronised sweeps** (since bench 17.10.0, `sweeps.py`):
  after the new field, and outside every other figure, for a commit with the sweep reader
  (`WIKIDATA_SPLIT_SWEEPS`; docs/journal/2026-10-26-progressive-finalise.md). The slice's
  entities, given new ids round-robin over 8 shards, are written as a dump of 8 shards one
  after another (`dumpslice.sharded`), as Wikimedia's is. On it, in byte order and as 8
  sweeps, each in a process of its own: the reader alone and the split (CPU, wall, peak
  RSS; `sweeps_*`), and the census's frontier each way. Its parity: every entity in one
  part, and the files the release publishes, after both sets' runs and finalise through
  the stand-in Hub, byte for byte the same from both splits.
- **Not included:** the upload to Hugging Face and its verification (network-bound, and
  it would publish), and, in the timed run, finalise's commits, verification and cards
  (the emulated link's finalise has the commits and verification, through the stand-in,
  not the cards).

## What the numbers mean

The quick tier is a sample: 33,220 entities (76 MB of source) of a dump of over 120
million. Its wall time is the time for those entities through the stages above, not a
release. Each result also gives the sample's throughput (entities per second) and a full
Wikidata ETA: the release 20260928's 121,815,642 entities (75,432,640 main, 46,383,002
scholarly, in 23,035 chunks; `docs/guide/performance.md`) through the same stages at
the same rate on the same runner. It is a straight extrapolation, for reading a change
at release scale ("−2h10m a release") rather than a prediction of a release's length:

- it leaves out downloading the dump and the uploads to the Hub, and runs the stages one
  after another, where a release overlaps the split and both sets' pipelines (the
  release wall, modelled, takes those in: above, and CHANGELOG.md 17.1.0);
- a release runs on a bigger machine (20 cores there, 4 on a runner) with more workers;
- the sample is 51% main-set and 49% scholarly entities, against 62% and 38% in the
  release, and costs per entity differ between the sets.

The claims are scaled by their own rows and bytes, not by entities (CHANGELOG.md
10.0.0). The sample's entities carry about 2.8 times a release's claims each: its chunk
`c0`, the dump's first 10,000 entities, has 62 statements an entity, against about 10 in
the release's main set (and in the bench's routed main-set chunks). The claims' CPU time
goes with their rows (over the 11 chunk files, about 9 µs a row, with no fixed part a
chunk to speak of), so:

- the pipeline's claims build (everything `normalise_claims_direct` calls: genson's
  decode, the statements and their write), for a part the split left as JSON, is scaled,
  in each set, by the release's claims rows over the sample's: the main set's 780 million
  (modelled; 774 million measured in the philippesaade build of an earlier dump), the
  scholarly set's modelled from its chunk files' 49.5 GB (measured) at the sample's
  scholarly chunk files' rows a byte (both route-release's one-file JSON at zstd 9: about
  999 million rows; bench 12.0.0 to 17.0.0 took the sample's side as its parts and the
  split's built files, about 760 million, kept beside as `_17`). Since the split builds the claims (bench 12.0.0), their
  build is in the split's CPU time, scaled with it by the dump's bytes, and the pipeline
  only places each chunk's claims file: those placements are scaled by entities with the
  pipeline's other parts (CHANGELOG.md 15.2.0). The pipeline ETA is its wall time scaled
  by its CPU time so reckoned over the sample's;
- finalise's part per byte is split by table (`phases.table_cpu`), and its claims share
  (about 85%) is scaled by the release's claims bytes over the sample's claims key: the
  main set's 44.7 GB (measured), the scholarly set's modelled likewise (each of the 11
  chunk files' claims partition is 0.91-1.11 times the chunk file). The other tables'
  shares are scaled by their own rows (below), and the rest by entities as before.

finalise's other tables are scaled by their own rows at a release (CHANGELOG.md 15.2.0):
each of the labels', descriptions', aliases' and links' stages but the per-file ones (the
decode, the sort and its check, the encode and the written file's check) by the
release's rows of that table over the sample's. The sample's main set has about 4 times
a release's labels and aliases an entity, and 18 times its sitelinks (chunk `c0`'s
well-known items), so entities overstated them. A release's rows are modelled: the main
set's the 20260507 build's (measured, `docs/dataset_cards_metadata.json`) grown by 2%,
the scholarly set's at the sample's rows a scholarly entity (not on record). Bench
15.1.0's ETAs are kept beside (`pipeline_eta_15_1`, `finalise_eta_15_1`, `eta_15_1`).

finalise's stages whose CPU time goes with the other tables' files, not their bytes
(CHANGELOG.md 15.1.0: compaction's plan and links, the local copy's check by sha256, the
sort's plans from the footers) are scaled by a release's files, not by entities. The run
on three times the rows has three times each key's group files (2,323 in the first run,
a file a key), and those stages' extra CPU time, less their extra bytes' sha256 at the
rate the claims' check hashed theirs, is about 0.3-0.7 ms a file. Scaled by entities, it
stood for about 3,700 times a key's files; a release has a group file a key a group. Its
files are modelled as the sample's keys times a release's groups in both sets (30 each,
`RELEASE_GROUPS`): 139,380, every key in every group, an upper bound (one main set, 2,757
keys in 31 groups, had 71,718: 26 a key). They run in finalise's process a file at a time,
so on one core: about 1-2 min of the finalise ETA, where entities made them about 20 min.
Bench 15.0.0's ETAs are kept beside (`finalise_eta_15`, `eta_15`).

The pipeline's worker starts are a fixed cost (CHANGELOG.md 11.0.0), as finalise's are
(9.0.0): `pool.process_chunks` spawns a worker process (about 0.7 CPU-s: the interpreter
and the package's imports) for every `CHUNKS_PER_WORKER` (8) chunks, so a release starts
about 2,900 for its 23,035 chunks, where scaling them by entities stood for about 14,700.
Each worker's CPU time outside its chunks is measured from the worker itself (its CPU
time at its exit less its chunks'), taken out of the part scaled by entities, and a
release's count (`pipeline.pool_starts`, from its chunks, measured, and the pool's
settings) added back at the bench's mean.

Bench 9.0.0's and 10.0.0's figures are kept beside them (`pipeline_eta_9`,
`finalise_eta_9`, `eta_9`; `pipeline_eta_10`, `split_eta_10`, `eta_10`) and shown under the
ETA. Still scaled by entities, and likely overstated by it (CHANGELOG.md 11.0.0): the
other tables' stages, whose CPU time goes mostly with their partition files (about
0.3-0.7 ms a file, against about 1 µs a row), and a release's partition files a chunk are
not on record; and `merge_group`, whose tables are glued at once in threads, so not
timed apart.

Beside the ETA, the comment gives the sample end to end, measured and not modelled: the
split's, the pipeline's and finalise's (first run's) CPU time and wall time summed
(`measured_cpu`, `measured_wall`), and the ETA by stage with each stage's (and each part
of the pipeline's) rate, the ETA's seconds a CPU-second of the sample's. Work moved from
one stage to another (the claims' decode from the pipeline into the split, say) leaves
the measured total as it was, and moves the ETA by the difference of the two rates.

## The data

The 11 chunk files are on the `bench-data` branch, which is data only and never merged.
The workflow reads it at a fixed commit (`DATA_COMMIT` in
`.github/workflows/bench.yml`) and checks every file against `SHA256SUMS`. A new version
of the data is a new commit, and a new series of scores.

- **Contents:** 60,000 entities of the 2026-10-12 Wikidata dump, as `split-dump` wrote
  them: 7 columns of JSON text (`id`, `labels`, `descriptions`, `aliases`, `sitelinks`,
  `claims`, `entity`).
- **What the pipeline reads** (since bench 12.0.0): not these files, but the split's own
  output. The split runs first, on a dump slice made of the tier's files, and the pipeline
  runs on the parts it wrote, each set's in its manifest's order, the main set's first,
  cycled as the files were. A split that writes a part's claims partition file (and the
  part without its claims column) hands the pipeline that file too, as a release's
  pipeline gets it; so too a split's built labels, descriptions, aliases and links (each
  a file with its index, since 14.0.0), and its entities file (since 15.0.0). The split's parity checks the parts against the files (below).
- **Files:** `c0` is chunk 0; `r10` to `r90` are routed chunks of the main set, and
  `s10` to `s90` are chunks of the scholarly-article set.
- **Size:** 96 MB. A default `git clone` fetches every branch, `bench-data` included; to
  skip it, clone with `--single-branch`.

## The commit before, from the store

The `bench-results` branch keeps each recorded commit's results in `runs/`. The pull
request comment's opening table takes the commit before's figures from there, as a
preview while the rounds run. The result itself always measures both commits, paired on
one runner: the same code's wall time differs by up to 1.5× across runners, and by 2–9%
across runners of one CPU model, against about 1% between two runs paired in one job.
(`ab.py --cache` can still reuse a stored measurement for local runs; the workflow does
not use it.)

## The pull request comment

Work in progress is not benchmarked: a draft pull request, or one whose head commit's
subject starts with `wip` or holds `[wip]`. Push such commits freely; mark the pull
request ready for review, or push a commit without the marker, to run it.

One comment per pull request, edited in place. As a run starts it says the result is
pending; once a pull request has results, a new push keeps them and marks them as being
recomputed, "(pending 🔁)" on the score and "(pending 🔁 <commit>)" beside the commit
they are for, until the new results replace them. Each row gives both commits' figures
and the change: absolute, and as a percentage, or as a multiple once a figure more than
doubles (2.3×) or falls below half (÷2.5). A figure that did not change shows as "=".
Times are durations: 1h23m45s, 10m23s, 11.42s (seconds keep their decimals only under a
minute). Sizes are in the largest unit in which the commit before's value is at least 1
(1234 MB shows as 1.234 GB), set by that value for the whole row, with the decimals that
keep the measurement's precision.

## Rules

- **Packages:** each commit runs on its own `uv.lock`, so a dependency change shows in
  the comparison.
- **Order:** a round is two jobs on two runners, in parallel with every other round's:
  forward runs the commit before first, reverse the commit after first. The round's
  figures are the mean of the two, and a change the geometric mean of the forward and
  reverse changes (their ratios' logs averaged; since bench 17.1.0, before which it was
  their arithmetic mean, which reads a noisy figure high), so what running first or
  second does falls on each commit once and cancels.
  A round shows in the comment once both its halves are in. `primer.py` touches 3 GB of
  fresh memory before each run, so each starts from the same memory history.
- **The split's parity:** each run's split must give the sample's rows exactly: the
  scholarly works in the scholarly set and the rest in the main one, every column as the
  files have it. A part with no claims column (its claims built by the split) must have a
  claims partition file with the content of the one the commit's own claims call
  (`process.normalise_claims_direct`) makes of the files' claims for its rows: the same
  rows, order, schema and row groups (since 13.0.0; 12.0.0 compared their bytes). A part
  with none of the labels, descriptions, aliases and sitelinks columns must have the four
  tables' files (or, for a table with no rows, its index alone), each with, key by key,
  the rows the commit's own processing of the files' column gives that key, in order, in
  the same columns and dtypes (since 14.0.0).
  Likewise a part with no `entity` column: its entities file must have the content of the
  entities the commit's pipeline decodes from the files' `entity` JSON (since 15.0.0).
- **Parity:** round 1 keeps both arms' outputs, and `parity.py` checks the second arm's
  against the first's. The check covers the group files' rows and their order, the
  Arrow schemas and metadata, the quarantine files, and every audit sidecar column but
  `file_size` (`path` without its run directory or its key's directory, since 14.0.0),
  and finalise's published files, claims_labels' among them since 16.0.0. A change whose output differs fails the job and makes no record.
  One difference passes, listed apart (since 17.5.0): a commit before the coordinate rule
  (a globe-coordinate's latitude, longitude and precision always in `__number`,
  src/wikidata/claims_numeric.py) against one after it, when their claims are the same by
  the rule, the commit after's already keep it, and nothing else differs.
- **Harness:** both commits run under the same harness, the one of the commit after (or
  of the pull request). The job summary gives its commit.

## Measuring another repository's unreleased change

A pull request can measure a dependency's unreleased build (a polars-genson or pq-glue
branch) without building it in every job, from the wheel that repository's CI built:

1. `bench/wheels`, one line per wheel: `<owner/repo> <run id> <artifact>`, e.g.
   `lmmx/polars-genson 40082865651 wheels-linux-x86_64` (the runners are Linux x86_64).
2. The wheel downloaded to `wheels/` (`gh run download <run id> -R <owner/repo> -n
   <artifact> -D wheels`), and the dependency taken from it in `pyproject.toml`:
   `[tool.uv.sources] polars-genson = { path = "wheels/<file>.whl" }`; `uv lock` pins its
   hash. `wheels/` is not committed.

The workflow downloads each listed artifact into that commit's `wheels/` before making
its environment, and lists them in the job summary. A git source works too, built from
source in every job. Such a pull request is a measurement, not for merging.

## What is recorded

Each run records:

- wall time;
- CPU time of all processes, and of process, partition and the group merges;
- peak RSS of all processes, and of the largest worker: the pipeline's (sampled every
  0.2 s), and each finalise run's (`finalise_peak_rss_mb`, `finalise_worker_peak_mb`, and
  `_x3` for three times the rows; from `timeline.py`'s samples, every 0.5 s: the process
  tree's RSS summed, and the largest worker's own peak, VmHWM, as last sampled);
- the peak rise in disk use, and the bytes the disk device wrote;
- temporary directories left behind;
- where finalise's wall time goes beside its CPU time (`timeline.py`): the cores busy
  over each of its stages, sampled every 0.5 s by a process outside the run's figures,
  with each stage's idle core-seconds and its worker jobs (how many at once, the longest,
  the last one's time alone). The comment shows it on three times the rows and per byte;
- finalise's worker processes, from the workers themselves (`phases.track_process`, a
  line as each starts and exits): by kind, how many, their jobs, their CPU time outside
  their jobs (their start, imports included, and their end; `claims_worker_cpu` for the
  claims key's) and their own peak RSS at exit;
- the sample end to end, measured (`measured_cpu`, `measured_wall`: the split, the
  pipeline and finalise's first run, summed), the first rows of the table;
- the pipeline's chunk workers, from the workers themselves (`pipeline.pipeline_workers`):
  how many (`pipeline_workers`; `pipeline_workers_counted`, what `pool_starts` counts for
  the bench), their mean CPU time outside their chunks (`pipeline_worker_cpu`, a row of
  the table) and to their first chunk (`pipeline_worker_startup_cpu`); and each chunk's
  start and end and its process's CPU time at its start (`chunks`: `t0`, `t`, `pcpu0`);
- the pipeline's CPU time by table and set, with each table's rows and partition files
  (`pipeline_tables`, raw results only; `pipeline.tables_measured`);
- the claims' work in each set, measured (`pipeline.claims_measured`): its chunks' claims
  CPU time, rows, bytes and chunk-file bytes (`pipeline_claims_{cpu,rows,bytes,source}_{main,scholar}`,
  `pipeline_claims_cpu` shown as a row), and the claims' share of finalise's part per byte
  (`finalise_claims_share`, from `phases.table_cpu`);
- modelled from these, and kept apart from the measured figures (`pipeline.claims_model`,
  CHANGELOG.md 10.0.0): the pipeline and finalise ETAs with the claims parts scaled by
  the claims' rows and bytes (`pipeline_eta`, `finalise_eta`, `eta`; bench 9.0.0's as
  `pipeline_eta_9`, `finalise_eta_9`, `eta_9`), and the release's claims they assume
  (`release_claims_rows_scholar`, `release_claims_mb`, and the scales);
- modelled likewise (`pipeline.starts_model`, CHANGELOG.md 11.0.0): the pipeline ETA with
  a release's count of worker starts (`release_pipeline_workers`, at
  `release_chunks_per_worker`; their CPU time `release_pipeline_workers_cpu`), bench
  10.0.0's kept as `pipeline_eta_10`, `release_pipeline_cpu_10` and, with the split on its
  slice, `eta_10`;
- the other tables' group files and keys in the first run (`finalise_files`,
  `finalise_keys`), and their per-file stages' CPU time on three times the rows less the
  first (`finalise_file_stages_cpu`, from `phases.table_cpu` by table and function);
- modelled from these (`pipeline.files_model`, CHANGELOG.md 15.1.0): those stages' bytes'
  sha256 (`finalise_file_hashed_cpu`), kept in the part per byte, their CPU time a file
  (`finalise_file_ms`), a release's files (`release_files`) and their time
  (`release_files_eta`), in the finalise ETA; bench 15.0.0's as `finalise_eta_15`, `eta_15`;
- the other tables' per-byte stages' CPU time on three times the rows less the first, by
  table (`finalise_{labels,descriptions,aliases,links}_bytes_cpu`), measured, and modelled
  from it (`pipeline.tables_model`, CHANGELOG.md 15.2.0): each table's rows at a release
  (`release_{table}_rows`) and scale (`{table}_rows_scale`), and the change they make to
  the finalise ETA (`release_tables_eta`); bench 15.1.0's as `pipeline_eta_15_1`,
  `finalise_eta_15_1`, `eta_15_1`;
- the pipeline's claims build CPU time (`pipeline_claims_build_cpu`: its
  `normalise_claims_direct` calls, a part the split left as JSON), measured;
- modelled likewise (`pipeline.release_model`, CHANGELOG.md 9.0.0): the finalise ETA with a release's count of worker starts
  (`finalise_eta`; bench 8.0.0's, every start in the part per byte, as
  `finalise_eta_8`), and finalise's peak RSS at a release's rows a bucket: a bucket-sort
  worker's (`release_bucket_worker_rss_mb`) and the process tree's with
  `FINALISE_WORKERS` (6) of them at their peak at once (`release_finalise_rss_mb`);
- the split and both sets' pipelines, apart from the timed run (`overlap.py`, CHANGELOG.md
  17.0.0): measured, each mode's wall (`overlap_seq_wall`, `overlap_wall`), the process
  tree's peak RSS (`overlap_seq_peak_rss_mb`, `overlap_peak_rss_mb`) and the groups'
  bytes uploaded (`overlap_seq_upload_mb`, `overlap_upload_mb`); modelled, those bytes at
  a release at 7 MB/s (`upload_eta_modelled`);
- over the emulated link (`link.py`, CHANGELOG.md 17.2.0): measured, the overlapped
  runs' wall with their pushes over a 7 MB/s link and the link's busy time
  (`overlap_link_7_wall`, `overlap_link_7_up_s`); finalise's three runs' wall without the
  link and over a 7 and a 60 MB/s link, and each run's (`finalise_link_{0,7,60}_wall`,
  `_run1_wall` to `_run3_wall`), the link's busy time each way (`_up_s`, `_down_s`), the
  share of it under the compute (`_hidden`), and the bytes published
  (`finalise_link_published_mb`; by set and table, `finalise_link_bytes`); modelled
  (`overlap.release_wall`), the release's wall at 7, 30, 60 and 120 MB/s with finalise's
  transfers (`release_wall_link_{r}_modelled`, its uploads' end and finalise's wall
  `release_uploads_end_link_{r}_modelled`, `release_finalise_link_{r}_modelled`), the
  build branch's and the published bytes (`release_group_gb_modelled`,
  `release_published_gb_modelled`) and finalise's download (`release_finalise_download_modelled`);
  and bench 17.1.0's wall model's figures, kept (`release_wall_modelled_17_1`, and its
  parts' `_17_1`).

Every 30 s each run writes its stage and the lowest memory available to the job's log
(`ab.Heartbeat`). A round job whose results do not come in (its runner lost, most likely
out of memory) is named in the summary and the comment, with what its log tells
(`lost.py`): the run it was in, its last stage and memory, and how it ended.

Each job also records the runner's CPU, cores, memory and image, the Python and package
versions, both commits, and the data's checksum. The job summary shows the score and a
table of ranges. The raw JSON, each run's per-chunk figures and RSS samples included,
is the `bench-results` artifact.

## Running it

**In GitHub Actions:** it runs on every pull request that changes `src/`,
`pyproject.toml`, `uv.lock`, the harness or the workflow, again on each push (a push
cancels the run in progress), and keeps one comment on the pull request up to date. On
every such push to master it appends the record to the `bench-results` branch, a
data-only store that is never merged: nothing is committed to master. To run it on any
commit, branch or tag, use the Benchmark workflow in the Actions tab. Packages and data
are cached, so the runs start within a minute. A run takes a few minutes on a
standard runner (4 cores, 16 GB).

**Locally**, with the data checked out in `data/`:

```sh
git fetch origin bench-data
git worktree add --detach ../data origin/bench-data
(cd ../data && sha256sum -c SHA256SUMS)
git worktree add --detach ../before master   # or any commit
(cd ../before && uv sync --frozen)
uv sync --frozen
.venv/bin/python bench/ab.py --data ../data --out bench-results \
  --arm before=../before --arm after=.
```
