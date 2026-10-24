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

[1.1.0]: https://github.com/lmmx/wikidata-pq/pull/34
[1.0.0]: https://github.com/lmmx/wikidata-pq/pull/33
