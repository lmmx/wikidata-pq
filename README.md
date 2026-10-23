# Benchmark results

The record of wikidata-pq's benchmark (`bench/README.md` on master). Data only, never
merged.

- `records.jsonl`: one entry per run, appended by the Benchmark workflow and never
  edited: a quick-tier push to master (the new commit against the previous one), or a
  full-tier run from the Actions tab. Each entry carries `bench`, the version of the
  harness that measured it (`bench/pyproject.toml`, changes in `bench/CHANGELOG.md`),
  and `tier`. Entries compare only within a tier and a major version.
- `RECORDS.md`: a view of `records.jsonl` (`bench/record.py --view`), one table per
  major version and tier, newest first.
- `runs/<tier>/<commit>.json`: each entry's raw results, one per round.
- `cache/`: commits' measurements by cache key (pipeline, harness, data, workload) and
  CPU model, so a pull request's commit before need not run again.

Everything recorded before the harness had a version was tagged `1.0.0`, `quick`, in one
commit; its raw results moved to `runs/quick/`.
