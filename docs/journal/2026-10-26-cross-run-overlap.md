# 2026-10-26: A release's finalise runs overlapped across their processes (cross-run)

A release is finalised in three runs, each its own process: the main set (since L2, the
early finalise in `just overlap-release`), the scholarly set, then the main set again for
claims_labels, which needs both sets' labels sorted (`just release`).
- **L1a (#170)** overlaps a table's commit and verify with the next table's write, but
  only within a run. Each run still ends with its last commits exposed, and joins its
  committer before it exits.
- **The next run then starts from cold:** its compaction's downloads and its first sort,
  with the link idle until its first table is written.

This entry measures those gaps on master (L1a and L2 in) and models what removing them
saves. It then builds the least invasive version behind `WIKIDATA_FINALISE_CROSS_RUN`,
off by default: each run starts once the run before has written its last table, while its
commits are still on the link. The model is in `2026-10-26-cross-run-overlap/model.py`.
Run it with master's bench record (bench-results branch,
`runs/quick/296adf44ea33d5a65912af4e2f8d32829dc3b328.json`).

## Current State

- **Built, behind `WIKIDATA_FINALISE_CROSS_RUN=1`** (off by default; the default is for
  the coordinator to decide on the bench's evidence):
  - `src/wikidata/cross_run.py`: the orchestrator (`finalise-release-runs`), the set
    lock, the run's mark and the published unsent bytes;
  - `main.finalise`: five lines under the flag;
  - the `overlap-release` and `release` recipes;
  - the test, `scripts/test_finalise_cross_run.py`.
- **Bench 17.5.0** sees it when the flag is on, in both the emulated-link stage
  (measured) and the release wall (modelled).
- **Modelled saving, the default 32 GB gate, L3 off:** −47 min at 7 MB/s, −23 min at 15,
  −13 min at 30, −2 min at 60, −1 min at 120.
  - **Ungated (64 GB):** −78, −50, −17, −2 and −1 min, at up to ~100 GB more disk at
    7 MB/s.
- **Hub calls: only reordered, none new.** `hub.py` is unchanged, and so are the
  commits, their batches and the checks. The scholarly set's compaction downloads,
  commits and verifies start earlier, while the main set's last commits are in flight.
- **Measured on the CI sample:** see "Measured" below (this PR's bench comment).

## The gaps on master

From bench 17.4.0's link stage on master (296adf4, CI run on record, 6 runs of the "after"
arm, medians). The sample's finalise runs three times through the stand-in Hub:

| run | no link (s) | 7 MB/s scaled (s) | the link's addition (s) | 60 MB/s scaled (s) | the link's addition (s) |
|---|---|---|---|---|---|
| 1, main | 12.50 | 28.90 | **+16.4** | 14.57 | +2.1 |
| 2, scholar | 7.57 | 30.13 | **+22.6** | 10.29 | +2.7 |
| 3, main (claims_labels) | 2.71 | 3.69 | +1.0 | 2.90 | +0.2 |
| **the three** | 22.78 | 62.74 | +40.0 | 27.79 | +5.0 |

- **At 7 MB/s the link is busy 45.15 s up and 4.06 s down,** of a 62.74 s wall. Only 19%
  of the link's time is under the compute (`finalise_link_7_hidden`, 0.19).
- **Every run's addition is mostly its exposed tail:** its last commits after its writer
  is done, with the CPU idle, then a cold start for the next run with the link idle.

At a release, from the model (bench 17.4.0's `finalise_link`, reproduced to the minute).
The CPU idle is each run's committer's tail after its writer has written its last table.
The link idle runs from a run's last commit to the next run's first.

| gap, min | 7 MB/s | 15 MB/s | 30 MB/s | 60 MB/s | 120 MB/s |
|---|---|---|---|---|---|
| CPU idle, end of run 1 (main) | **124** | 48 | 15 | 1 | 1 |
| link idle, run 1 → run 2 (the scholarly set's downloads and claims sort) | **75** | 75 | 75 | 75 | 75 |
| CPU idle, end of run 2 (scholar) | 21 | 9 | 3 | 0 | 0 |
| link idle, run 2 → run 3 | 2 | 2 | 2 | 2 | 2 |
| finalise, from its start to its end | 7h22m | 4h39m | 3h27m | 2h55m | 2h52m |

- **The large one is the boundary between the main set's first run and the scholarly
  run.** The main set's ~60 GB of published files take 2h23m at 7 MB/s, against its ~80
  min of compute. Then the scholarly run computes its claims for ~75 min (download and
  sort) with the link idle.
- **The second boundary is small.** Run 3 is claims_labels' build (2.5 min) and its
  commit.
- **Above 30 MB/s, the CPU tails go:** the link keeps up with the compute.
- **The link-idle gap stays ~75 min at every rate,** but it costs the wall only while
  the link has work to fill it: the main set's commits. Above ~60 MB/s, those commits
  end with its compute.

## The model

`model.py` keeps bench 17.4.0's `release_wall` (`_phase`, `_uploads_end`,
`_main_uploads_end`) and `finalise_link`'s tables, bytes and compute. Its `today` row is
bench 17.4.0's `release_wall_link_{r}_modelled` to the minute at every rate. Under
`xrun`:
- **The next run's writer starts once the run before has written its last table.**
  - The scholarly run starts no sooner than its set's last push, as finalise's guard
    requires.
  - The main set's second run starts once its first has ended. They share a ledger, so
    the lock serialises them.
- **The runs' commits go on one link, FIFO.**
- **claims_labels' build** waits for the set's own claims and labels, and for the other
  set's labels, to be committed.
- **The gate:** a table to write is not started while the other set's written,
  uncommitted bytes are over it. This is on top of each run's own L1a bound.
- **One process with one committer,** the hypothesis's other form, has the same timeline
  at a release's scale. Each run's start-up and ledger reads are seconds. So it has no
  row of its own, and it isn't built: it would need the set-dependent config
  (`STATE_DIR`, `HUB_COPY_DIR`, `REPO_TARGET` and ~20 more, imported by name across the
  package) to become per-call.

Release walls, L3 off (the default):

| | 7 MB/s | 15 MB/s | 30 MB/s | 60 MB/s | 120 MB/s |
|---|---|---|---|---|---|
| **today** (master: L1a, L2) | 12h00m | 8h50m | 7h38m | 7h06m | 7h03m |
| cross-run, ungated (each run's own 64 GB bound only) | 10h42m | 7h59m | 7h21m | 7h04m | 7h02m |
| cross-run, gated at 64 GB | 10h42m | 7h59m | 7h21m | 7h04m | 7h02m |
| **cross-run, gated at 32 GB (the default)** | **11h13m** | 8h27m | 7h25m | 7h04m | 7h02m |
| cross-run, gated at 16 GB | 11h29m | 8h34m | 7h29m | 7h04m | 7h02m |
| cross-run, gated at 0 | 11h58m | 8h47m | 7h36m | 7h06m | 7h03m |
| cross-run in `release` only (runs 2 → 3), 32 GB | 11h58m | 8h47m | 7h36m | 7h06m | 7h03m |

Savings against today:

| | L3 | 7 MB/s | 15 MB/s | 30 MB/s | 60 MB/s | 120 MB/s |
|---|---|---|---|---|---|---|
| ungated / gated at 64 GB | off | −78 min | −50 min | −17 min | −2 min | −1 min |
| **gated at 32 GB (the default)** | **off** | **−47 min** | **−23 min** | **−13 min** | **−2 min** | **−1 min** |
| gated at 16 GB | off | −31 min | −16 min | −9 min | −2 min | −1 min |
| gated at 0 | off | −2 min | −2 min | −2 min | 0 | 0 |
| ungated / gated at 64 GB | on | −67 min | −55 min | −21 min | −2 min | −1 min |
| **gated at 32 GB (the default)** | **on** | **−47 min** | **−23 min** | **−13 min** | **−2 min** | **−1 min** |
| gated at 16 GB | on | −31 min | −16 min | −9 min | −2 min | −1 min |

With L3 on, the walls are 11h40m today and 10h53m with the 32 GB gate, at 7 MB/s.

- **Where the saving comes from:** at 7 MB/s, the scholarly run's downloads and its
  claims sort (~75 min) run under the main set's last commits. The scholarly commits then
  follow on a link that has never gone idle.
- **Why 30 MB/s still gains 13 min:** the 75 min link-idle gap is still there. At
  30 MB/s, the main set's commits outlast its compute by ~15 min, so that much of the
  scholarly compute hides.
- **Why it is ~0 above 60 MB/s:** the link keeps up with each run's compute, so a run
  ends within a minute of its last write.
- **Why gating at 0 gains ~2 min:** the next run then starts only once the last commit is
  in. It saves only the verify, the `done` stage and the process start-up.

## Design

The least invasive form keeps each run its own process, as today, so nothing about a set's
config, ledgers or restart changes. It adds an orchestrator and three signals between
processes.

- **The orchestrator** (`cross_run.run_release`; the command `finalise-release-runs`)
  starts the three runs (`finalise-wikidata`) in order. Each starts once the run before
  has written its `finalise.written`, or has ended.
  - **One run writes at a time.** The next starts only once the one before has handed
    its last table to its committer. So the CPU and memory are as before; only the commits
    overlap.
  - **The scholarly run waits for `--after FILE`,** when given: the scholarly set's run's
    exit code. It must be 0 for the scholarly finalise to start, as finalise's guards
    require its set's groups all pushed.
  - **A run that fails before its mark** stops the later runs from starting. A run that
    fails after it leaves the others going.
  - **Every run started is waited for,** and the exit code is 0 only if every run's was.
- **Each run, under the flag** (`main.finalise`):
  - **`hold`:** it takes its set's `state/finalise.lock` (flock, exclusive, held for the
    process's life) before anything else. A set's second run therefore waits for its first
    to end, since they share a ledger and a committer must not be doubled. The other set's
    run can also tell it is going (`alive`: a shared, non-blocking probe). The kernel drops
    the lock with the process, however it ends.
  - **`mark_written`:** it writes `state/finalise.written` once it has handed its last
    table to its committer. That is just before the join on either path: the early return
    when the other set's labels aren't sorted, or after claims_labels' `sort`.
  - **`cross_run.Committer`:** the L1a committer, unchanged, with a thread that publishes
    its written, uncommitted bytes to `state/finalise.pending` as they change.
    `wait_for_other(table)` starts no table to write (one not yet `written`) while the
    other set's running finalise publishes more than `WIKIDATA_FINALISE_CROSS_RUN_GB`
    (default 32). It finishes its own verified tables while it waits.
  - **`wait_labels`:** before claims_labels' build, it waits for the other set's labels
    to be `done` while that set's finalise is running, instead of stopping at once. With
    no such run it stops as before, and the next run of the recipe builds it.
- **The recipes:**
  - **`overlap-release`, with the flag:** once the main run ends with 0, it starts
    `finalise-release-runs --after FILE` in place of the early finalise, at nice 10 as L2's
    was. FILE gets the scholarly run's exit code when that run ends. The recipe waits for
    both and fails if either fails.
  - **`release`, with the flag:** `finalise-release-runs` in place of its three
    `finalise-release` calls. After `overlap-release` has run them all, each run finds its
    tables done and makes no Hub call but the cards' check, as a rerun always has.
  - Without the flag, both recipes run as before. `release` is now a bash recipe so it
    can branch, and each of its lines still fails it (`set -euo pipefail`).
- **What is not touched:** `sort_by_id.py`, `compact.py`, `process.py`, `hub.py` and the
  Committer's own code. The subclass overrides nothing of it; it adds a thread and a
  method.

**claims_labels still waits for both sets' labels.** Its build reads both sets' sorted
labels (`claims_labels.labels_dirs`). It is built only once `sort_stage(other, labels)`
is `done`, as before. Under the flag, the run waits for that instead of returning early,
for as long as the other set's run is going.

## Restart: every new state and its recovery

`scripts/test_finalise_cross_run.py` first runs the split and both sets' runs on a dump
slice (r50 and s30 here, chunks of 2,000 entities), with bench/link.py's stand-in Hub.
Each case then finalises a copy of that output, its uploads slept for their bytes on one
link (0.4 MB/s), so a run's last commits are still going when the next starts. Each case
must end as the reference does (the three runs one after another, the flag off):
- the build branches byte for byte, every repo's files, published and not;
- the local copies, by sha256;
- both sets' cards' metadata;
- both sets finalised.

A kill is `os._exit` at a hook, as by SIGKILL. The orchestrator goes on as it does, then
is run again, as `just release` would be, until it ends with 0.

| killed at | at the kill | what follows | ok |
|---|---|---|---|
| main's first run, before its mark (labels `written`) | main's labels written, not committed; no scholarly run started | the orchestrator starts nothing more and fails. The rerun: main resumes (L1a's states), then the rest | (results) |
| main's first run, after its mark, at a commit | the scholarly run writing; a main table `written` with part of its keys on the Hub | the scholarly run goes on. Main's second run takes the lock once the first is gone, resumes its commits (`commit_table` skips keys on the Hub), and builds main's claims_labels. The orchestrator fails (run 1's exit), and the rerun finds everything done | (results) |
| the scholarly run, claims `written`, main's commits going | scholar claims handed over or not | run 3 isn't started; run 1 ends normally. The rerun: main's run 1 has nothing to write, the scholarly run resumes, then run 3 | (results) |
| the scholarly run, claims_labels `written`, before its mark | scholar claims_labels written, not committed | as above | (results) |
| main's second run, claims_labels `written`, the scholarly commits going | main claims_labels written, not committed | the scholarly run ends normally. The rerun: main's first run builds nothing (claims_labels already written) and hands it to its committer | (results) |
| both: main after its mark mid-commit, the scholarly run mid-compaction | each at its own ledger's stage | each resumes its own on the rerun | (results) |

Also checked uninterrupted:
- **The overlap:** each next run starts before the one before has ended, and only once it
  has written its mark.
- **claims_labels** built for both sets.
- **The lock:** main's second run takes it only once its first has ended.
- **The gate at 0 GB:** the scholarly run's first table starts only after the main run's
  last commit.
- **The recipes,** with stand-ins for their commands:
  - `overlap-release` runs `finalise-release-runs --after FILE` after the main run, at
    nice 10, and FILE reads 0 once the scholarly run ends. With the scholarly run failing,
    FILE reads 1 and the recipe fails. Without the flag, the early finalise runs as before;
  - `release` runs `finalise-release-runs` once in place of three `finalise-wikidata`, and
    three without the flag.

Also run, all passing, with the flag off: `test_finalise_overlap.py`,
`test_finalise_early.py` and `test_finalise.py`.

## Disk and memory on a 16 GB laptop

- **Memory: unchanged in kind.** One run writes at a time, so the bucket sorts and
  claims_labels' jobs (11.2 GB at a release, the tightest point) never run beside another
  run's.
  - What runs beside a writer is the run before's tail: its committer thread (file paths,
    streamed hashing), its `done` stages (file moves), its claims refs (from the
    sidecars, seconds), and for the scholarly run its card figures (`update_stats`: a few
    bytes an id, ~0.1–0.5 GB) and cards.
  - The finalise parent is ~170 MB (#170's measure).
  - So add ~0.2–0.7 GB to the writer's peak.
  - The sort's memory budget is taken from the memory available when each step starts
    (`parallel.memory_budget`), so it leaves room for it.
- **Disk: the gate bounds it.** A table written and not yet committed holds its sorted
  files and its sources (the local copy, until `done`): about twice its published bytes.
  Under cross-run, those of the run before are on disk while the next run sorts.
  - The model's estimate of finalise's own peak: the sorting table's working set (claims
    3× its group bytes, the rest 2×), plus each written, uncommitted table at 2×, plus
    each done table's local copy (not claims, deleted after its refs). Today it is 174 GB
    at every rate, at the end of the scholarly claims sort (3 × 52 GB, plus the main set's
    done local copies). The dump and any kept group files are outside it.

    | finalise's peak on disk, GB | 7 MB/s | 15 MB/s | 30 MB/s | 60 MB/s | 120 MB/s |
    |---|---|---|---|---|---|
    | today | 174 | 174 | 174 | 174 | 174 |
    | cross-run, ungated or 64 GB | **276** | 174 | 174 | 174 | 174 |
    | **cross-run, 32 GB (the default)** | **174** | 174 | 174 | 174 | 174 |
    | cross-run, 16 GB | 174 | 174 | 174 | 174 | 174 |

  - **Why 32 GB by default:** the scholarly claims sort (~75 min at 7 MB/s) starts once
    the main set's unsent bytes are under 32 GB. They drain in about the same time (32 GB
    at 7 MB/s is 76 min), so by the sort's peak, at its end, they are about gone. The peak
    stays at today's, and it keeps 60% of the ungated saving at 7 MB/s.
  - **Ungated, or 64 GB,** saves 31 min more at 7 MB/s, for ~100 GB more at the peak:
    ~380 GB with the dump, against ~280 GB. That is more than a few-hundred-GB laptop
    should be assumed to have.

## The bench (17.5.0)

- **The link stage** (`link.finalise_stage`) runs the three runs through the commit's own
  orchestrator, when the commit's config has the flag on. Each run's process is the
  stage's FINALISE, as before.
  - `finalise_link_{r}_wall` is then the first run's start to the last run's end.
  - `finalise_link_{r}_run{1,2,3}_wall` is each process's own wall, and these now overlap.
- **The release wall** (`overlap.finalise_link`) takes the cross-run timeline when
  `finalise_link_bytes["cross_run"]` is set: the gate, in bytes, read from the commit's
  config.
  - On master's six runs it gives the journal model's rows to the minute: 12h00m → 11h13m
    at 7 MB/s with the 32 GB gate.
- **A commit without the flag** is measured and modelled as before, to the second. master
  has it off.
- **This PR is benched twice:**
  - first with the flag's default flipped on (the head that shows the change);
  - then with it off again (the head to merge as a prototype, which should show no change).

## Measured

(CI results below.)

## Uncertainty and negative findings

- **The rate decides it.** At 7 MB/s, the one rate on record, the saving is ~47 min (78
  ungated). Above 60 MB/s it is within a couple of minutes of nothing.
- **Not modelled:**
  - two sets' commits sharing the link's bandwidth at once (the model's link is FIFO);
  - the scholarly set's downloads competing with the main set's uploads, if the two
    directions share a link (#170's caveat).
- **The disk estimate is coarse:** a table's working set taken as a multiple of its bytes,
  and a table's held bytes released at its last commit, not at its verify and `done`.
- **The one-process form wasn't built:** same timeline, and a large refactor of the
  set-dependent config.
- **The run-2 → run-3 boundary is worth ~2 min on its own** (`release` alone, with L2's
  early finalise ending inside `overlap-release`: the table's last row). That is why
  `overlap-release` starts the orchestrator, not only `release`: the saving is at the
  boundary between L2's early finalise and the scholarly run.
