"""A release's branch in each table's repo, and its promotion to main.

A release (see config.RELEASE) is uploaded, compacted and sorted on the branch
`build-{release}` (config.HUB_REVISION) of each table's repo, so `main` keeps the previous
release while it builds. The branch starts from `main` emptied of its data files, so it
holds only the release's.

`promote-release` then tags `main` with the release it holds (once: a tag that exists is
left), commits the branch's files to `main` in place of its own (the Hub stores a file's
content once, so the commit re-uses the branch's uploaded bytes), tags `main` with the new
release, and deletes the branch.
"""

from huggingface_hub import CommitOperationCopy, CommitOperationDelete, HfApi

from .config import HF_USER, HUB_REVISION, RELEASE, REPO_TARGET, Table

# Files a repo keeps whatever its release: its card and attributes
KEEP = {"README.md", ".gitattributes"}
COMMIT_OPS = 1000


def _data_files(api: HfApi, repo_id: str, revision: str | None) -> list[str]:
    return [
        f
        for f in api.list_repo_files(repo_id, repo_type="dataset", revision=revision)
        if f not in KEEP
    ]


def ensure_build_branch(repo_id: str, api: HfApi) -> None:
    """Create the release's branch from `main`, without main's data files, if it is not
    there yet (an existing branch is the release's own progress, so it is left as it is)."""
    if not HUB_REVISION:
        return
    refs = api.list_repo_refs(repo_id, repo_type="dataset")
    if any(b.name == HUB_REVISION for b in refs.branches):
        return
    api.create_branch(repo_id, branch=HUB_REVISION, repo_type="dataset")
    if old := _data_files(api, repo_id, HUB_REVISION):
        api.create_commit(
            repo_id,
            repo_type="dataset",
            revision=HUB_REVISION,
            operations=[CommitOperationDelete(path_in_repo=f) for f in old],
            commit_message=f"Start release {RELEASE}: the previous release's files removed",
        )
    print(f"[hub] {repo_id}: branch {HUB_REVISION} for release {RELEASE}", flush=True)


def promote(previous: str, hf_user: str = HF_USER, api: HfApi | None = None) -> None:
    """Make the release's branch each repo's `main` (see the module docstring), `main`'s
    current files first tagged `previous` (the release they are)."""
    if not RELEASE:
        raise SystemExit("Set WIKIDATA_RELEASE to the release to promote")
    api = api or HfApi()
    for table in Table:
        repo_id = REPO_TARGET.format(hf_user=hf_user, tbl=table)
        refs = api.list_repo_refs(repo_id, repo_type="dataset")
        tags = {t.name for t in refs.tags}
        if RELEASE in tags:
            print(f"[promote] {repo_id}: already at {RELEASE}", flush=True)
            continue
        if not any(b.name == HUB_REVISION for b in refs.branches):
            raise SystemExit(f"[promote] {repo_id} has no branch {HUB_REVISION}")
        if previous not in tags and _data_files(api, repo_id, None):
            api.create_tag(repo_id, tag=previous, repo_type="dataset", revision="main")
            print(f"[promote] {repo_id}: main tagged {previous}", flush=True)
        new = sorted(set(_data_files(api, repo_id, HUB_REVISION)) | {"README.md"})
        stale = [f for f in _data_files(api, repo_id, None) if f not in set(new)]
        # The release's files first (overwriting same-named ones), then the stale ones out,
        # in commits of COMMIT_OPS
        ops = [
            CommitOperationCopy(src_path_in_repo=f, path_in_repo=f, src_revision=HUB_REVISION)
            for f in new
        ] + [CommitOperationDelete(path_in_repo=f) for f in stale]
        for i in range(0, len(ops), COMMIT_OPS):
            api.create_commit(
                repo_id,
                repo_type="dataset",
                operations=ops[i : i + COMMIT_OPS],
                commit_message=f"Release {RELEASE} ({i // COMMIT_OPS + 1} of "
                f"{-(-len(ops) // COMMIT_OPS)})",
            )
        api.create_tag(repo_id, tag=RELEASE, repo_type="dataset", revision="main")
        api.delete_branch(repo_id, branch=HUB_REVISION, repo_type="dataset")
        print(f"[promote] {repo_id}: main is release {RELEASE}", flush=True)


def run_promote() -> None:
    import os

    previous = os.environ.get("WIKIDATA_PREVIOUS_RELEASE")
    if not previous:
        raise SystemExit(
            "Set WIKIDATA_PREVIOUS_RELEASE to the tag for main's current files "
            "(20260507 for the philippesaade build)"
        )
    promote(previous)
