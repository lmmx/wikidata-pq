# 2026-10-04: Docs fixes from the FAQ reachability audit

Follows `ux/faqs.md` (90 simulated user questions, plus J. 91-96 added here) and
`ux/reachability.md` (a verdict per question). Documentation only: no change to `src/`,
the `Justfile` or `docs/dataset_cards/`, while the run of release 20260928 is in progress.

## Current State

- The Hub's `permutans/wikidata-labels` had only a `main` branch and no tags on 2026-10-04,
  and `permutans/wikidata-scholar-labels` had `main` and `build-20260928`
  (`https://huggingface.co/api/datasets/permutans/{repo}/refs`) — `main` of the six
  `wikidata-*` repos holds the philippesaade build, and the tag `20260507` is created by
  the first `promote-release`.
- README.md states that `main` holds the philippesaade build until 20260928 is promoted, that
  `just run` ends by calling finalise (`main.run`, `if not RELEASE: finalise(...)`), and that
  CC0 applies, and links the docs site.
- `docs/using-the-data.md` (nav "Using the data", and in the llmstxt sections) covers which
  table for what, subsets and `all`, `load_dataset` with several languages through
  `data_files`, lookups by id and by other columns, `mul`, choosing by `rank`, `time`
  precision and BCE years, quantity units, exploding qualifiers and references, the
  20260507 build and a release compared, pinning by tag, Hub rate limits, license, and a
  list of terms.
- The examples in `docs/using-the-data.md` for rank, dates, quantities and qualifiers ran
  against `permutans/wikidata-claims@main` (Q42, 337 statements, 331 of the best rank), and
  the release `references` example ran on one file of
  `permutans/wikidata-scholar-claims@build-20260928` — the `hf://...@{revision}` form of a
  Polars path reads a branch.
- `docs/guide/setup.md` has a "Limits" section: no limiting of languages, tables or chunks
  (`main.run` lists every chunk below `PARTITION`, `Table` is fixed at import); no skipping
  the scholarly set (`main.finalise` waits for the other set's labels sort before
  claims_labels); no lock on a working directory (nothing in `src/` takes one); and
  `download-wikidata` fetches every table in full (`main.download`).
- `docs/reference/config.md` names the five `WIKIDATA_*` environment variables as the only
  settings read from the environment, and every other constant as an edit to `config.py`.
- `docs/reference/hub.md` covers reading `build-{release}` before promotion, what a promotion
  that stops leaves on `main` (`hub.promote` tags `previous`, then commits in batches of
  `COMMIT_OPS`, then tags the release and deletes the branch), and
  `WIKIDATA_PREVIOUS_RELEASE` for a first release (`hub.run_promote` requires it for the
  main set).
- `docs/reference/push.md` states that a verify mismatch stops the run and that a rerun
  verifies again without re-uploading (`close_group` resumes after `pushed`).
- `docs/reference/finalise.md` describes a set's repos before finalise, and what to run when
  finalise refuses.
- `docs/reference/dump.md` and `docs/guide/release.md` state that routing deletes the split
  chunks, and that a bz2 deleted before `split.done` is downloaded again (`dump.split` reads
  the dump from the start on a rerun).
- The "Documented against" blocks of the pages edited record `b8ac85a`; the files they list
  have the same checksums as at `6243a2c`, apart from the `Justfile` (the `mkdocs` recipe).

## Missing

- The dataset card templates (`docs/dataset_cards/`, `dump/`, `scholar/`) do not define
  `mul` where they use it (the aliases cards do not mention it), and do not link the terms
  list in `docs/using-the-data.md#terms` — left until the run of 20260928 is finished, as
  `cards.py` renders and pushes the templates during finalise.
- The aliases cards do not say that null aliases in the philippesaade copy were dropped
  (README.md "Notes on coverage" only).
- No page covers reading the tables with pandas, DuckDB or Spark, every `datatype` value,
  redirects and deleted entities, a minimum of memory, or an update policy for releases.
- No page states what a rerun of `promote-release` does when a repo's commits all landed
  but its release tag was not created (the copy operations are sent again; the Hub's
  answer to a commit that changes nothing was not checked).
- The comments in `demos/*.sh` give `DATA=wikidata` as an example, which reads
  `wikidata/{table}/{key}/` (`demos/item.py:36-41`, `demos/classes.py:43-49`) — the README's
  `snapshot_download` example writes `wikidata/wikidata-{table}/`, so the two layouts match
  only with `local_dir=f"wikidata/{table}"`.
