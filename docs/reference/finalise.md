# Finalise

`finalise-wikidata` (`main.finalise`) turns a set's uploaded groups into the published
layout, once every chunk is `COMPLETE` and no group is unfinished. It refuses to start
otherwise, saying to run `process-wikidata`: `just run-release {release} {set}` for a
release, `just run` without one. Each stage is recorded in a ledger, so a rerun skips what is done.

## Order for a release

1. **Claims first:** [compact](compact.md), then [sort](sort.md). The claims sort is the
   largest local job (the local copy, its buckets and the sorted files, about three times
   the claims). Doing it first means no other table's local copy is on disk yet.
2. **Collect claims_labels' refs** from the sorted claims (`collect_refs_stage`), then
   delete the claims' local copy.
3. **Every other table but claims_labels:** compact, then sort. Each sort downloads the
   table's local copy into `hub/{table}`.
4. **claims_labels.** It needs both sets' labels sorted. If the other set's labels sort is
   not `done`, finalise prints that it is waiting and returns without writing
   `finalise.done`. Otherwise it [builds claims_labels](claims-labels.md) and uploads it to
   the branch, then compacts and sorts it like the other tables.
5. **Card figures** (`update_stats`), recomputed where stale from the local copies.
6. **Dataset cards**, rendered and pushed where they differ from the Hub's
   ([Dataset cards](cards.md)).
7. `state/finalise.done`, which promotion requires.

The `release` recipe calls `finalise-release` for the main set, then the scholarly set,
then the main set again. The first call stops at step 4 because the scholarly labels are
not sorted yet. The scholarly call completes, with the main set's labels sorted by then.
The second main call completes the main set.

## Before finalise

Until a set is finalised, each repo has the uploaded groups,
`{key}/chunks-{first}-{last}.parquet`: one file per key per group, not sorted by id across
files, so they can be read but a filter on id reads every file. For a release they are on
the build branch, and `main` keeps the previous release until promotion, which requires
`finalise.done`. The philippesaade build uploads to `main` directly.

## Without a release

All six tables are compacted, then sorted, in `Table` order; claims_labels comes from the
chunks like the others. Then the card figures and cards. `run` calls `finalise` itself at
the end. The sort expects a local copy in `hub/`, which `download-wikidata` fetches.

??? info "Documented against"
    Commit `b8ac85a` (2026-10-04). See [About these docs](../about.md) to check for changes.

    | File | SHA-256 |
    |---|---|
    | `src/wikidata/main.py` | `578eeb24bd587fe45e004c8423ff02648c8235b58ddd35714351f9dd2e1ec03c` |
