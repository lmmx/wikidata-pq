# 2026-10-26: Finalise's published upload overlapped with its compute (L1a)

L1 from `2026-10-25-release-critical-path.md` (#167). Before this change, finalise ran
each table's `write_table`, then its `commit_table` and `verify_table`, then the next
table. So the link sat idle while the CPU worked, and the CPU sat idle while the link
worked. Now one background committer commits and verifies each table, in order, while
finalise writes the next. The design reorders the calls only:
- **Unchanged:** `hub.py`, the Hub API calls, commit_table's batches, and the checks.
- **Changed:** when these calls run, and in which thread.

The model and the sample measure are in `2026-10-26-finalise-upload-overlap/`:
`model.py` and `measure.py`.

## Current State

- **Done:**
  - `sort_by_id.Committer`: one thread, fed by `sort_table` once each table's `written` is
    recorded;
  - `main.finalise`'s loop and its join points;
  - `WIKIDATA_FINALISE_PENDING_UPLOAD_GB`: the bound, default 64; `0` restores serial;
  - `WIKIDATA_FINALISE_CLAIMS_LAST=1`: opt-in, off by default;
  - the test, `scripts/test_finalise_overlap.py`.
- **The release-wall figures are modelled, not measured.** The CI bench's finalise has
  no link. It runs sort_by_id's local stages itself, not `sort_table` or
  `main.finalise`, so it can't see this change. B1 (bench 17.2.0's emulated link) is
  what would show it on CI.
- **The saving is smaller than #167's L1 row.** That row assumed the link streams from
  the first sorted part through all three finalise runs, which is effectively L1b,
  `preupload_lfs_files`. Reordering alone can start a table's upload only once the
  whole table is written. Each run (main, scholar, main) is its own process. Claims
  goes first and is ~83% of finalise's compute, so only the later tables' compute and
  downloads hide under claims' upload. Modelled:
  - **−36 min at 7 MB/s, 15 and 30 MB/s;**
  - **−32 min at 60 MB/s;**
  - **−17 min at 120 MB/s.**
- **L1b and cross-run overlap are not done.** Both are open decisions in #97.

## Design

- **The committer** (`sort_by_id.Committer`) is one daemon thread in finalise's own
  process. It takes tables in the order `sort_table` hands them over. For each table:
  - it runs `commit_table`, then records `committed`;
  - it runs `verify_table`, then records `verified`.

  These are the same calls, with the same batches. A table is handed over only once
  its `written` stage is recorded, that is, after `write_table` returned:
  - every key sorted, with `_check_sort` and `_check_written`;
  - every file's sha256 in the manifest.
- **The `done` stage** (`finish_table`) runs in finalise's own thread (`reap`), between
  tables and while it waits. It is `write_metadata`, `replace_local_copy` and the scratch
  removal, unchanged. It stays in that thread because compaction's own `write_metadata`
  rewrites the same `dataset_cards_metadata.json`. Two threads must not read, modify and
  write it at once.
- **Join points in `main.finalise`:**
  - claims' refs (`collect_refs_stage`) and the deletion of claims' local copy run as
    soon as claims is done;
  - claims_labels' build waits for the set's labels and claims to be done, since it reads
    their sorted local copies;
  - the committer is joined, with every table handed over committed, verified and done,
    before the stats, the cards and `FINALISE_DONE`;
  - it is also joined before the early return when the other set's labels aren't sorted
    yet.

  So nothing is reported done before every verify.
- **The ledger** (`sort.jsonl`) is appended to by both threads, each for its own tables,
  under a lock. Each table's stages stay in order. The only new thing is that one table's
  `committed` can follow the next table's `written`.
- **`=0`** builds no committer and runs the old serial path in `sort_table`, line for
  line. The `done` block moved into `finish_table` unchanged.
- **Under the overlap, the buckets go once `written` is recorded** (with
  `CLEAN_UP_LOCAL`), not at `done`. Only `write_table` reads them. This frees the claims'
  fragments (~25 GB at a release) while its parts wait for the link.

### Claims last (`WIKIDATA_FINALISE_CLAIMS_LAST=1`, off by default)

- **What it does:** a release's claims are sorted after the set's other tables, not
  first. Their commits run under the claims sort, and the claims' commit runs at the
  run's end.
- **Saving, modelled:**
  - **more below ~30 MB/s:** −1h36m at 7 MB/s, −45 min at 15;
  - **less above:** −22 min at 30, −11 min at 60, −6 min at 120.
- **Costs:**
  - the other tables' local copies (~16–24 GB a set) are on disk during the claims sort,
    which serial order avoids;
  - in the scholarly run, claims_labels' build waits for the claims' commit.
- **Same published content.** It changes only which repo is committed first. Each
  repo's commits are the same (tested).
- **The bound still holds sensibly.** The other tables are pending, ≤24 GB, when the
  claims sort starts. No table follows claims in a run except claims_labels, which waits
  for it anyway.

## Restart: every crash state and its recovery

Each was tested (`scripts/test_finalise_overlap.py`). The test kills finalise with
`os._exit` at the point, then runs it again to the end. The Hub's files, each repo's
commits, the local copies and the cards' metadata must all match a serial run.

| killed at | the ledger after | the rerun |
|---|---|---|
| mid-write of table N+1 while table N's commit is in flight | N `written`, its commit not landed; N+1 `sourced`, its manifest partial | N handed over again and committed; N+1's write resumes from its manifest (the sort's own resume) |
| mid-commit: a batch landed, the next not | N `written` | `commit_table` skips each key already on the Hub (`_key_done`). Batches are cut greedily in manifest order, so the rest are the same batches an uninterrupted run makes |
| after a commit landed, before `committed` was recorded | N `written` | every key on the Hub: no commit; `committed` recorded, then verify |
| between commit and verify | N `committed` | verify, then done |
| after verify, before `verified` was recorded | N `committed` | verify again (reads only), then done |
| mid-way through `done` (the local copy half replaced) | N `verified` | `finish_table` again: sources removed where present, sorted files moved where still in the out dir (as before) |
| the committer failing (a commit raises) | N `written`; the tables after it at `written` or earlier | finalise stops with an error (below); the rerun resumes as above |
| the writer failing, or Ctrl-C | the committer's commit in progress abandoned, as by a kill | as above |
| after `written`, before the handover | N `written` | handed over on the rerun |

- **A table's commits are the same with the committer as without.** There is one caveat,
  which predates this change. Which keys share a commit follows the manifest's order.
  That is the order the sort's jobs finished in, and it varies from run to run, serial
  or not. Each key's additions and deletions are always in one commit, and the batches'
  limits are the same. The test compares each key's operations, by sha256, and the
  number of commits per repo.
- **A kill leaves the sort's spawned workers orphaned**, as a SIGKILL of finalise always
  has. The test runs each finalise in a session of its own and kills that session.

## The disk bound

- **The rule:** `WIKIDATA_FINALISE_PENDING_UPLOAD_GB` (default **64**). Finalise starts
  no further table, neither its compaction (with its download) nor its sort, while more
  than this many sorted bytes are written and not yet committed.
- **It is checked between tables,** so the bytes held reach at most the bound plus one
  table.
- **`=0`** restores the serial order.
- **Why 64 GB:**
  - **Above a set's sorted output at a release.** The main set's claims are ~43 GB and
    its other tables ~16 GB. The bound never binds at a release, so the whole of each
    run overlaps.
  - **Below the claims sort's peak.** The extra disk under the overlap is each waiting
    table's sources, still in its local copy until `done` (≈ its sorted bytes). For
    claims, it is also its sorted parts, which serial order deletes right after its
    refs. That is ~86 GB while claims commits, against serial's ~0 at that moment. The
    claims' fragments go at `written` instead of `done`.

  So the overlap phase holds ~86 GB plus the next tables' working set (~30 GB), about the
  claims sort's own peak (sources, fragments and parts, ~110–130 GB). Finalise's peak on
  disk stays near #167's ~235–300 GB.
- **A laptop short of disk** can set 16 GB:
  - claims (43 GB) then commits before the next table starts;
  - the later tables still overlap one another;
  - modelled, −23 min at 7 MB/s.

## Errors

- **A failure in the committer:**
  - it prints `[sort] {table}: COMMIT FAILED, finalise stops` with the traceback at once,
    and stops the committer;
  - the writer raises `CommitterFailed` at its next handover, wait or reap. At the
    latest, that is once the table it is writing is written, and that work is kept;
  - no table after it is committed in that run;
  - `FINALISE_DONE` isn't written;
  - the failing table is left at `written` for the rerun.
- **A failure in the writer:** finalise stops at once. The committer's daemon thread is
  abandoned mid-commit, as a kill would leave it.

## Memory

- **What the committer holds:** a thread in finalise's parent. `commit_table`'s
  operations are file paths, and huggingface_hub hashes and uploads them streaming.
  It holds no table's data.
- **Measured** on the sample (below): finalise's own peak RSS, serial against overlapped.
- **The workers' memory is unchanged:** the sort's workers and its memory budget
  (`FINALISE_MEMORY_GB`) don't see the thread.

## Measured on the bench's quick-tier sample, with an emulated link

- **What runs:** `measure.py` runs `main.finalise` for real (compaction, the sort,
  `commit_table` and `verify_table`) on the group files the bench's pipeline kept for the
  quick tier (`KEEP=1 bench/pipeline.py`, data e8c97e1, files c0 r50 s30 s90). It runs
  against the test's stand-in Hub, whose commits sleep for their bytes at a stated rate.
- **The setup:** 4 workers, 3 rounds a case, medians. This was the shared 4-core machine,
  run while no other job ran.
- **The rate is scaled:** each case reproduces the ratio of upload time to compute that
  the main set's run has at the release rate named. At a release that run computes
  ~4,100 s and uploads ~59 GB.
- **The sample's tables aren't the release's mix,** so these are walls of this sample
  under an emulated link. They are not a release's figures.

MEASURED_TABLE

## Modelled: a release's wall

`model.py` takes #167's inputs (`critical_path.py`) and bench 17.0.0's on record:
- finalise's three runs, 2h22m36s;
- the claims ~82.6% of it, 84.2 GB over both sets;
- the main set's claims 43 GB;
- the other tables ~40 GB, 16.2 GB of them the main set's;
- claims_labels 5 GB and 5 min.

Each run is simulated table by table: download, compute, then commit, serial or
overlapped, with the join points above. Its serial row reproduces #167's "today" row
exactly.

MODEL_TABLE

- **L1a's saving is the later tables' compute and downloads** hidden under the
  claims' upload. It is capped by the compute that follows claims in each run.
- **At 120 MB/s the link no longer outlasts that compute:** claims commits in ~6 min.
- **With kept group files (L3)** the downloads are gone, and so is their share of the
  saving.

## The bench

- **No regression is expected, and parity should pass.** The CI bench's finalise
  (`bench/pipeline.finalise`) calls `write_table`, `write_metadata`,
  `replace_local_copy` and `record_stage` itself, and stubs the Hub calls. Those
  functions are unchanged, so the bench measures the same code, and the published files
  are the same.
- **It can't show the gain:** the overlap lives in `sort_table` and `main.finalise`,
  which it doesn't call, and it has no link.
- **For B1:**
  - the switch is `WIKIDATA_FINALISE_PENDING_UPLOAD_GB` (0 = serial);
  - the committer is `sort_by_id.Committer`;
  - the overlap is within a run, in table order, claims first.

## Missing

- **The Hub's real upload rate,** which sets the saving: between −36 and −17 min.
- **Whether two directions share the link.** The model takes the build branch's
  download (compaction of the next table) and the upload as independent.
- **A per-batch bound.** The bound counts a table's bytes until its last commit. For the
  ~600-key tables it could release batch by batch, but there's no need while it doesn't
  bind.
