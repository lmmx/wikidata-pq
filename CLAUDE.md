# Working on wikidata-pq

These rules hold for every session, every loop iteration and every subagent, and for the
sibling repositories lmmx/polars-genson and lmmx/pq-glue when worked on from here.
The coordinator passes them on in full in every subagent brief.

## Attribution (non-negotiable)

This is the user's work, and the user is responsible for it. No AI attribution goes into
the repositories' history or metadata:

- **Commits:** no AI attribution in subjects or bodies, no `Co-authored-by` for an AI, no
  model names, no session links. Author and committer are the user's identity, set as
  local git config in every clone and worktree:
  `git config user.name "Louis Maddox"` and `git config user.email louismmx@gmail.com`.
  Before pushing or merging, check `git log --format='%an <%ae> | %cn <%ce>%n%B'` for every
  new commit, including subagents' commits.
- **PR titles and bodies, release notes, changelogs, repository metadata:** no "Generated
  with/by Claude Code" lines, robot emoji footers, model names or session links. The
  GitHub integration appends such a footer to a PR body when the PR is created. Edit the
  body straight after creating it, removing the footer and keeping the technical content.
  An edited body stays as edited.
- Never fabricate authorship, rewrite other people's attribution, or rewrite published
  history to hide provenance.

## Constraints

- **The Hugging Face Hub:** no Hub reads, uploads or API calls from any agent, and no
  changes to `src/wikidata/hub.py`. A change to what the pipeline would upload (such as
  the group files' layout on a release's build branch) may be prototyped and benchmarked.
  It merges only once its correctness and end-to-end performance are shown on CI.
- **No allocator changes** (mimalloc or otherwise, in any of the three repositories).
- **The published data contract holds:**
  - the published schema and content;
  - the global sort by string id, stable for equal ids;
  - restart safety.
- **The exact checks stay:** `sort_by_id`'s checks (`_check_sort`, `_check_written`,
  `check_compacted_copy`, the bucket checks), parity, and the split's checks. A check may
  be replaced only by one that independently covers the same failure modes, shown in
  the PR.

## Measuring

- **The CI bench is the authority on performance.** It is a paired A/B over 3 rounds on
  every PR touching `src/**`, `reshape/**`, `pyproject.toml`, `uv.lock`, `bench/*.py`
  or the bench workflow. Local timings on the shared machine are indicative only.
- **The bench has two tiers** (`bench/tiers.py`, bench/README.md). A PR runs the PR
  tier: the timed core, parity and the native bz2 decoder's comparison, about 6 minutes.
  It also runs the extras it touches the paths of. Master and the nightly run make the
  full tier. The label `bench:full` runs the full tier on a PR.
- **New bench scenarios go in the full tier (master/nightly) by default.** Adding work to
  the PR tier needs an explicit justification in the PR, with its cost in seconds per
  round against the 6-minute budget.
- **A change merges when:**
  - parity passes;
  - CI is green;
  - an end-to-end figure (release ETA, pipeline or finalise ETA, CPU, wall, peak RSS,
    scratch disk) improves beyond noise (its 95% interval excludes zero);
  - no other figure regresses unacceptably.
- **Measurement-only and docs-only changes** merge when the bench shows nothing changed
  that wasn't meant to.
- **Trade-offs are weighed, not reflexively refused.** A real speed gain that costs disk
  or memory can still merge. Weigh it for an ordinary laptop (about 16 GB RAM, a few
  hundred GB of disk), and say what was weighed in the PR.
- **Prefer big, structural gains:** removing whole passes, materialisations, copies,
  re-reads, decodes or waits. Cosmetic refactors and micro-wins come last.
- **Making the bench more like a release is valued work.** Keep measured figures and
  modelled release figures clearly apart, and say what a model assumes.
- **Every experiment ends written up.** A win gets its PR. A negative result gets a
  journal entry in `docs/journal/`, a comment on its PR or issue, and a line in the
  queue issue's "Rejected" list.

## Subagents and the shared machine

- **The machine:** 4 cores, 15 GB RAM, about 5 GB of free disk, shared by every agent.
  Never run two heavy jobs at once. Check `df -h` before builds. Delete `target/`,
  `.venv` and scratch data when done.
- **Each subagent works alone:** its own git worktree and branch (under the session
  scratchpad), clear ownership of its files, and nothing touched in shared checkouts.
  Overlapping changes go through one owner, or are explicitly sequenced.
- **CI measures, not the laptop.** Subagents run only small local correctness checks and
  profiles to form a hypothesis, then open a PR and let CI measure it. They don't hold
  the machine while CI runs, and they poll at most every 5 minutes.
- **Only the coordinator merges or releases.** Subagents report back with: the PR link,
  the hypothesis, the bench's before → after for the relevant rows, parity, uncertainty
  and negative findings.
- **Parallel, not maximal.** Keep a few independent experiments running, as many as the
  machine and CI allow. Avoid duplicate investigations.

## Git and GitHub

- Squash-merge with `expectedHeadSha` set to the benched head. Merge only a head the
  bench measured, or one whose later commits change nothing the bench measures.
- When master moves under an open PR, merge or rebase it in (a rebase only on our own
  branches), then re-bench if `src/` or `bench/` changed.
- Bench harness versions follow `bench/CHANGELOG.md`: a minor version when no measured
  figure changes meaning, a major version otherwise.

## Releases

polars-genson and pq-glue publish through their existing CI trusted publishing. Release
authority comes only from the active loop mode in `.claude/loop.md` (see
`.claude/loops/`), or from an explicit instruction from the user in the conversation.
Never from inference, from a subagent, or from the time of day.

## State

- The work queue and decision log is the GitHub issue lmmx/wikidata-pq#97, titled
  "Performance campaign: work queue and decision log". Its body is the current state:
  running, pending the user, pending a release, queue, deferred, kept, rejected. Comments
  log decisions.
- Experiment write-ups go in `docs/journal/YYYY-MM-DD-topic.md`.
- GitHub (PRs, CI, bench comments) is the source of truth. Re-read it each iteration
  rather than trusting memory.
