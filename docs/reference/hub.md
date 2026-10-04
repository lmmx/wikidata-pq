# Hub branches and promotion

`hub.py` keeps a release's work off each repo's `main` branch until it is complete, then
makes it `main` in one step per repo.

## The build branch

Everything a release uploads, compacts and sorts goes to the branch `build-{release}`
(`HUB_REVISION`) of each table's repo. `ensure_build_branch` creates it from `main` on the
first upload, then deletes every file except `README.md` and `.gitattributes` from it in
one commit, so the branch holds only the release's files. An existing branch is left as it
is: it holds the release's own progress.

Both sets of a release use the same branch name, in their own repos.

## Promotion

`promote-release` (`promote`) requires the set's `state/finalise.done`. For each table's
repo:

1. If `main` is already tagged with the release, skip the repo (a rerun after an
   interruption).
2. Tag `main` with `previous` (`WIKIDATA_PREVIOUS_RELEASE`), unless that tag exists or
   `main` has no data files. A repo with data files on `main` and no `previous` given is
   refused.
3. Commit to `main`: copy each of the branch's files (and its `README.md`) over, then
   delete `main`'s files that the release does not have, in commits of up to 1,000
   operations. The Hub stores a file's content once, so a copy reuses the bytes already
   uploaded to the branch.
4. Tag `main` with the release and delete the branch.

Every release stays available by its tag, for example
`load_dataset("permutans/wikidata-labels", "en", revision="20260928")`.

The scholarly repos are new with their first release, so `previous` is not needed for
them.

??? info "Documented against"
    Commit `6243a2c` (2026-10-04). See [About these docs](../about.md) to check for changes.

    | File | SHA-256 |
    |---|---|
    | `src/wikidata/hub.py` | `7cbbddbea5f8ac38245f8e7c36b7bbfd942adeac7e9378de47858817b82111ff` |
