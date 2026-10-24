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

To read a release before promotion, give the branch as the revision:
`revision="build-20260928"` for `load_dataset` and `snapshot_download`, or
`hf://datasets/permutans/wikidata-labels@build-20260928/en/*.parquet`. Until the set is
finalised, the branch holds the uploaded groups (`{key}/chunks-*.parquet`), neither
compacted nor sorted.

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
them. For the main set, `promote-release` always requires `WIKIDATA_PREVIOUS_RELEASE`, and
the `release` recipe always passes it. A repo whose `main` has no data files is not tagged
with it, so for a first release into new repos any value will do.

## If promotion stops

Repos are promoted one at a time, so a failure leaves the repos before it at the release
and the rest at the previous one. Within a repo, `main` is tagged `previous` before the
first commit, so the previous files stay available by that tag. A failure between a repo's
commits leaves its `main` with part of the release's files and part of the previous
ones. Rerunning `promote-release` skips the repos already tagged with the release and
promotes the rest from their branch, which is deleted only after the last commit. There
is no command that puts the previous release back on `main`.

??? info "Documented against"
    Commit `b8ac85a` (2026-10-04). See [About these docs](../about.md) to check for changes.

    | File | SHA-256 |
    |---|---|
    | `src/wikidata/hub.py` | `7cbbddbea5f8ac38245f8e7c36b7bbfd942adeac7e9378de47858817b82111ff` |
