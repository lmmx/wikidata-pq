# Finalise

`finalise-wikidata` (`main.finalise`) turns a set's uploaded groups into the published layout, once every chunk is `COMPLETE` and no group is unfinished. It refuses to start otherwise. Each stage is recorded in a ledger, so a rerun skips what is done.

## Order for a release

1. **Claims first:** [compact](https://lmmx.github.io/wikidata-pq/pr-preview/pr-152/reference/compact/index.md), then [sort](https://lmmx.github.io/wikidata-pq/pr-preview/pr-152/reference/sort/index.md). The claims sort is the largest local job (the local copy, its buckets and the sorted files, about three times the claims). Doing it first means no other table's local copy is on disk yet.
1. **Collect claims_labels' refs** from the sorted claims (`collect_refs_stage`), then delete the claims' local copy.
1. **Every other table but claims_labels:** compact, then sort. Compaction leaves the table's local copy in `hub/{table}` (its compacted files), which the sort reads, and the sort's commit is the table's one upload after the groups.
1. **claims_labels.** It needs both sets' labels sorted. If the other set's labels sort is not `done`, finalise prints that it is waiting and returns without writing `finalise.done`. Otherwise it [builds claims_labels](https://lmmx.github.io/wikidata-pq/pr-preview/pr-152/reference/claims-labels/index.md) into compaction's source directory, then compacts and sorts it like the other tables.
1. **Card figures** (`update_stats`), recomputed where stale from the local copies.
1. **Dataset cards**, rendered and pushed where they differ from the Hub's ([Dataset cards](https://lmmx.github.io/wikidata-pq/pr-preview/pr-152/reference/cards/index.md)).
1. `state/finalise.done`, which promotion requires.

The `release` recipe calls `finalise-release` for the main set, then the scholarly set, then the main set again. The first call stops at step 4 because the scholarly labels are not sorted yet. The scholarly call completes, with the main set's labels sorted by then. The second main call completes the main set.

## Without a release

All six tables are compacted, then sorted, in `Table` order; claims_labels comes from the chunks like the others. Then the card figures and cards. `run` calls `finalise` itself at the end. The sort expects a local copy in `hub/`, which `download-wikidata` fetches.

Documented against

Commit `6243a2c` (2026-10-04). See [About these docs](https://lmmx.github.io/wikidata-pq/pr-preview/pr-152/about/index.md) to check for changes.

| File                   | SHA-256                                                            |
| ---------------------- | ------------------------------------------------------------------ |
| `src/wikidata/main.py` | `578eeb24bd587fe45e004c8423ff02648c8235b58ddd35714351f9dd2e1ec03c` |
