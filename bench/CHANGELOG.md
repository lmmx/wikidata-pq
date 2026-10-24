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

## [7.0.0] - 2026-10-24

### Added

- The split: the release's first stage, split-dump then route-release (dump.split,
  dump.route), run as the CLI runs them on a slice of a dump made of the sample's own
  entities (dumpslice.py: each chunk row turned back into its dump line, the lines
  bzip2-compressed, made once a job and untimed). Each step's wall and CPU time (lbzip2
  and the workers included), its stages in the table of where the CPU goes, and its ETA:
  its CPU time per byte of the slice's bz2, scaled to the release's dump
  (RELEASE_DUMP_BYTES) over the cores, added to the release's ETA.
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
