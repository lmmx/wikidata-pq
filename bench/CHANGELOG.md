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
- The pull request comment's launch bar has a cell per job (6), its results bar a cell
  per round (🟨 for a round with one half in).

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

[2.0.0]: https://github.com/lmmx/wikidata-pq/pull/36
[1.1.1]: https://github.com/lmmx/wikidata-pq/pull/35
[1.1.0]: https://github.com/lmmx/wikidata-pq/pull/34
[1.0.0]: https://github.com/lmmx/wikidata-pq/pull/33
