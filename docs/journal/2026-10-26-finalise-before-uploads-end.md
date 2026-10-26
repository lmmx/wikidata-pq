# 2026-10-26: The main set's first finalise started before the build uploads end (L2)

L2 of `2026-10-25-release-critical-path.md` (#167), re-modelled on master after L1a
(#170) and bench 17.2.0's release wall (#168), then built. Until now, `just release`'s
first finalise (the main set's) started once `just overlap-release` had ended: the split,
both sets' runs and every push to the build branches. At 7 MB/s the link is the
release's path until then, and the CPU sits idle for the last ~50 min of it.

Now, once the split has numbered every chunk, the scholarly run's pushes wait their turn
behind the main run's. The main set's uploads therefore end first, and `overlap-release`
starts the main set's first finalise as soon as the main run ends. Its compute then runs
under the scholarly set's last uploads.

The model is in `2026-10-26-finalise-before-uploads-end/model.py`. Run it with master's
bench record (bench-results branch, `runs/quick/26e9259e13911801ec8984be45f5b81b281b74bb.json`).

## Current State

- **Built, behind `WIKIDATA_FINALISE_EARLY`** (on by default; `=0` restores the old order):
  - **The turn** (`src/wikidata/push/turn.py`, `pool.process_chunks`' new `hold`, a
    `wait_turn` call in `close_group`). Once the split has numbered every chunk, the
    main run holds an flock on its state dir's `push.turn` while it closes groups back
    to back. The scholarly run waits for it before each push.
  - **The recipe.** `overlap-release` runs `finalise-wikidata` for the main set, at
    nice 10, once the main run exits 0. It does this before waiting for the scholarly
    run.
  - **Bench 17.4.0.** The release wall's model sees the change. The bench's measured
    stages can't: they don't run finalise beside a run.
- **Modelled saving, L3 off (the default):** −24 min at 7 MB/s, −3 min at 15 MB/s,
  −1 min at 30 and 60 MB/s, 0 at 120 MB/s.
  - **With L3 on:** −24, −3, −1, −1, 0 min.
  - **Bench 17.4.0's own model** (equal groups, a group pushed once ready): −22, −1,
    −1, 0 min at 7, 30, 60 and 120 MB/s (it has no 15 MB/s row).
- **Smaller than #167's −57 min.** That figure assumed L3, all of finalise's compute
  hidden, and a link with no turn. Two things cut it:
  - L1a now hides part of the same compute;
  - the run's merges sit in its upload thread (L5, merge ahead, was rejected in #173).
- **Hub calls: only reordered, none new.** Nothing in `hub.py` changes. The same calls
  run in a different order:
  - the scholarly run's `push_group` calls start later;
  - the main set's first finalise runs its calls earlier: compaction's
    `snapshot_download`, then `list_repo_tree`, `create_commit` and the verify. They go
    to the main set's repos, while the scholarly run still pushes to its own.
  - `release`'s first `finalise-release main` now finds that run done in its ledgers,
    and makes no Hub call.
- **Finalise's sources are unchanged.** So is the wait on the build branch's verify:
  `main.finalise`'s guards (`get_next_chunk`, `unfinished_group`) already read only the
  set's own state dir. They are that wait. The main set's groups are all verified before
  its finalise starts, so compaction downloads (or reads kept files) as before.

## What L2 had to change, and what it didn't

#167's L2 moved finalise's guard from the compute to `commit_table` and read the groups
still uploading from local files. That is the `local` row below. It isn't needed, for two
reasons:

1. **Finalise's guard is per set.** `main.finalise` refuses to start while *its own* set
   has a chunk not complete or a group not done. The main set's first finalise needs
   only the main run to have ended, so the gate needs no code change, only the recipe.
2. **The two sets' uploads end together on a shared link.** Each set's pool closes one
   group at a time: merge, push, verify, then the next (`pool._Upload`). Under FIFO the
   sets take turns, and the main set's last push ends only ~5 min before the
   scholarly set's (5h00m against 5h05m at 7 MB/s). The gate alone saves ~5 min.

So the main set has to get the link first. Ordering alone doesn't do that:
- **Priority while the main set is pushing** (the scholarly set waits only for a main
  push in progress) does nothing, because the scholarly set takes the link while the
  main set merges its next group. The main set's uploads still end at 5h00m.
- **The main run has to hold its turn through its merges** while it closes groups back
  to back. That idles the link for each main merge.
  - Held from the start, it gains most at a 1-min merge (−38 min). But the merges in
    the split run at nice 10 on the cores the split leaves, and #173 puts them at
    0.3–2.4 min, with a worst case of 2–7 min. At 5 min, holding from the start costs
    +67 min.
  - **Held once the split has ended** (built), the merges are on free cores (~0.5 min).
    It saves −24 min at every merge wall modelled, 0.3 to 7 min.

## The model

`model.py` keeps bench 17.2.0's model:
- `_phase`: the split and both runs, a group ready when the runs have done its share of
  their CPU time;
- one upload link;
- `finalise_link`'s tables, bytes and compute, with L1a's committer and its 64 GB bound;
- the three finalise runs, one process after another.

It adds:
- **groups by set**, 30 each, at each set's bytes (main ~59 GB, scholar ~65 GB);
- **the coupled close:**
  - merges of 1 min during the split and 0.5 min after it;
  - 0.75 min for the scholarly set while the main set's finalise computes beside it at
    nice 10;
- **the turn;**
- **finalise's commits** filling the link's idle time between the scholarly pushes.

Its `today` row is 12h29m at 7 MB/s, against 17.2.0's own 12h24m on the same runs. The
gap is the merges and the sets' unequal bytes. Savings below are against `today` under
the same group rule.

The phase, L3 either way:

| | 7 MB/s | 15 MB/s | 30 MB/s | 60 MB/s | 120 MB/s |
|---|---|---|---|---|---|
| the runs' compute ends | 4h12m | 4h12m | 4h12m | 4h12m | 4h12m |
| every push ends (FIFO, today) | 5h05m | 4h19m | 4h15m | 4h14m | 4h13m |
| the main set's pushes end (FIFO) | 5h00m | 4h16m | 4h14m | 4h13m | 4h13m |
| the same, main first (not held) | 5h00m | 4h16m | 4h14m | 4h13m | 4h13m |
| the same, held from the start | 4h20m | 4h15m | 4h14m | 4h13m | 4h13m |
| **the same, held after the split (built)** | **4h39m** | 4h15m | 4h14m | 4h13m | 4h13m |
| every push ends, held after the split | 5h11m | 4h19m | 4h15m | 4h14m | 4h13m |

Release walls with finalise's transfers, L3 off (the default):

| | 7 MB/s | 15 MB/s | 30 MB/s | 60 MB/s | 120 MB/s |
|---|---|---|---|---|---|
| bench 17.2.0's `release_wall_link` (master) | 12h24m | 8h59m | 7h44m | 7h11m | 7h07m |
| **today** | 12h29m | 9h01m | 7h46m | 7h12m | 7h08m |
| gate (the recipe alone) | 12h24m | 8h59m | 7h45m | 7h12m | 7h08m |
| gate, main first (not held) | 12h24m | 8h59m | 7h45m | 7h12m | 7h08m |
| gate, held from the start | 11h51m | 8h58m | 7h45m | 7h12m | 7h08m |
| **gate, held after the split (built)** | **12h05m** | 8h58m | 7h45m | 7h12m | 7h08m |
| the same, finalise at nice 0 | 12h06m | 8h58m | 7h45m | 7h12m | 7h08m |
| `local`, held after the split (not built) | 12h00m | 8h55m | 7h43m | 7h11m | 7h07m |

Savings:

| | L3 | 7 MB/s | 15 MB/s | 30 MB/s | 60 MB/s | 120 MB/s |
|---|---|---|---|---|---|---|
| gate (the recipe alone) | off | −5 min | −2 min | −1 min | −1 min | 0 |
| gate, main first (not held) | off | −5 min | −2 min | −1 min | −1 min | 0 |
| gate, held from the start | off | −38 min | −3 min | −1 min | −1 min | 0 |
| **built: held after the split** | **off** | **−24 min** | **−3 min** | **−1 min** | **−1 min** | **0** |
| **built: held after the split** | **on** | **−24 min** | **−3 min** | **−1 min** | **−1 min** | **0** |
| built, under 17.2.0's rule (a group pushed once ready) | off | −26 min | −4 min | −1 min | −1 min | 0 |
| `local` (not built) | off | −29 min | −6 min | −3 min | −2 min | −1 min |
| `local` (not built) | on | −29 min | −6 min | −3 min | −1 min | −1 min |

With L3 on, the walls are 12h08m today and 11h44m built at 7 MB/s.

By a group's merge wall during the split (L3 off), held from the start / held after the
split:

| a merge in the split | 7 MB/s | 15 MB/s | 30 MB/s | 60 MB/s | 120 MB/s |
|---|---|---|---|---|---|
| 0.3 min | −42 / −24 min | −3 / −3 | −1 / −1 | −1 / −1 | 0 / 0 |
| 1 min (#173's central) | −38 / −24 min | −3 / −3 | −1 / −1 | −1 / −1 | 0 / 0 |
| 2.4 min | −25 / −24 min | −3 / −3 | −1 / −1 | −1 / −1 | 0 / 0 |
| 5 min | **+67** / −29 min | −2 / −2 | −2 / −2 | −1 / −1 | 0 / 0 |
| 7 min | **+50** / −47 min | **+19** / −10 | −1 / −1 | −1 / −1 | 0 / 0 |

**Where the saving comes from.** At 7 MB/s, the turn after the split moves the main set's
last push from 5h00m to 4h39m. The main set's first finalise then computes its claims
(~51 min) while the scholarly set's last ~32 min of pushes run. It commits once they're
done, its commits queued behind them on the link.

**Where it doesn't.**
- **At ≥30 MB/s**, the uploads end within ~2 min of the compute, so there is nothing to
  hide.
- **The turn costs the scholarly run** ~6 min of idle link while the main set merges
  (5h11m against 5h05m). That costs nothing on the wall: the early finalise's first run
  outlasts it.
- **`local`** would add 5 min at 7 MB/s, by compacting the main set's last group from
  disk before its push ends. The cost is a new source path in compaction, and commits
  gated on a verify still running. Not worth it.
- **`nice 0`** for the early finalise would starve the scholarly merges (CFS gives a
  nice-10 thread ~7% of a core against 6 nice-0 workers): −23 against −24 min. nice 10
  costs finalise ~nothing: the scholarly merges are ~0.75 min out of every ~6 min, on
  one core of four.

## Design

- **The turn** (`push/turn.py`).
  - The main run's pool calls `hold(True)` when a group's close starts with no more
    chunks to come. That is the case once `more()` has said the split numbered the last
    chunk, or for a run that doesn't follow the split.
  - It calls `hold(False)` when no close is running and none can start, while chunks
    are still to come, and at its end (also in `finally`).
  - While an upload runs, the pool now polls every `FOLLOW_POLL` (2 s). Before, once
    the split had ended, it waited for the next chunk's end. So a held turn is let go
    within 2 s of a close's end.
  - `hold_turn` takes or drops an exclusive flock on `releases/{release}/state/push.turn`.
  - The scholarly run's `close_group` calls `wait_turn` before `push_group`. It tries a
    shared flock without blocking. If the main run holds the turn, it prints that it is
    waiting, blocks until the turn is let go, and pushes at once.
  - **Neither touches a ledger.** A push already running isn't stopped: a main group
    that closes during a scholarly push shares the link with it.
- **The recipe** (`overlap-release`). After `wait $split`, it waits for the main run. If
  that run ended with exit code 0 and `WIKIDATA_FINALISE_EARLY` isn't 0, it runs
  `WIKIDATA_SCHOLAR= nice -n 10 env -u WIKIDATA_FOLLOW_SPLIT -u WIKIDATA_WORKERS finalise-wikidata`.
  Then it waits for the scholarly run.
  - A failed main run starts no finalise, and the recipe fails.
  - A failed early finalise fails the recipe once the scholarly run has ended.
  - `release` is unchanged. Its first `finalise-release main` finds the early run's
    tables done and stops at claims_labels' wait, as the third call always did.
- **`WIKIDATA_FINALISE_EARLY=0`** turns off both the turn and the recipe's finalise.

## Restart: every new state and its recovery

`scripts/test_finalise_early.py` runs `overlap-release` then `release` as the recipes
order them, with bench/link.py's stand-in Hub and its uploads slept on one link:
- the split and both runs at once;
- the early finalise once the main run exits;
- then finalise main, scholar, main.

The scholarly set's link is slowed 5× there, so its last pushes outlast the main run, as
at a release on a slow link. A killed process (os._exit) is run again until it completes,
the others left running. The build branches must end byte for byte as in the reference
run (`WIKIDATA_FINALISE_EARLY=0`, finalise after the runs), with both sets finalised.

| killed at | at the kill | the rerun |
|---|---|---|
| the main run, holding its turn, before its 2nd push | main's group `merged` and its turn held; the scholarly run waiting | the kernel drops the flock with the process, and the scholarly run pushes. The main run resumes its group from the ledger; the early finalise starts when it ends. ok |
| the scholarly run, after waiting, before its push | its group `merged` | the push resumes from the ledger. ok |
| the early finalise, claims `written`, the scholarly run pushing | L1a's states: claims handed to the committer or not | L1a's resume: claims committed, the rest written. ok |
| the early finalise, its first commit landed, the next not | a table `written`, part of its keys on the Hub | `commit_table` skips the keys on the Hub. ok |
| both: the scholarly run mid group, the early finalise mid compaction | each at its own ledger's stage | each resumes its own. ok |

Every kill fell while the scholarly run was still going, and the test checks the main
run held its turn when killed. In the uninterrupted run:
- no scholarly push started while the main run held its turn, and one waited for it;
- the early finalise started 6–8 s before the scholarly run ended;
- in the reference, 2 scholarly pushes started during a main push.

**The recipe itself** runs in the test with stand-ins for its commands on PATH:
- **default:** the early finalise runs after the main run and before the scholarly run
  ends, as the main set's, not following the split, with no `WIKIDATA_WORKERS`, at
  nice 10;
- **`WIKIDATA_FINALISE_EARLY=0`:** it doesn't run;
- **a failing main run:** it doesn't run, the scholarly run is still waited for, and the
  recipe fails.

Also run, all passing:
- `test_pool.py`, with a new test that every close with no chunks to come is held and
  none before;
- `test_overlap_restart.py`;
- `test_group_restart.py`, `test_keep_groups.py` and `test_finalise_overlap.py`.

## Disk and memory on a 16 GB laptop

- **Memory.** The early finalise overlaps only the scholarly run's tail: at 7 MB/s for
  ~32 min, at ≥15 MB/s for ~4 min or less.
  - The split has ended by then, and with it its 2.68 GB worker. So the scholarly run
    holds its parent, 2 workers of ~0.24 GB and a merge: about 1–1.5 GB.
  - Finalise's bucket sorts take their memory budget from the memory available when
    each step starts (`parallel.memory_budget`), so they leave room for it.
  - The claims_labels jobs (11.2 GB, the tightest point) are in runs 2 and 3, after
    every run has ended.
- **Disk.** When the early finalise starts at 7 MB/s, the scholarly groups not yet
  pushed are on disk: ~13 GB of partition files and a staged group, falling to 0 over
  ~32 min. Before this change, they were gone before finalise started.
  - The main set's claims sort reaches its peak (sources, fragments and parts,
    ~110–130 GB) near its end, ~50 min in, once they are gone. So the peak rises by
    ~0–13 GB, against #167's ~235 GB (L3 off) or ~300 GB (L3 on).
  - The turn moves backlog from the main set to the scholarly set but not its size: the
    link does the same work.
- **CPU.** The early finalise runs at nice 10, beside the scholarly run's nice-10
  merges. Both are nice 10, so CFS gives them equal shares.

## The bench

- **Bench 17.4.0** (`bench/CHANGELOG.md`). `overlap.release_wall` reads
  `config.FINALISE_EARLY` from the commit, as it reads the commit's other finalise
  defaults. For such a commit, the release wall with finalise's transfers takes the
  groups a set each in turn, the main set's first once the split has ended
  (`_main_uploads_end`). Finalise starts at the main set's last push, and its first
  run's commits queue behind the scholarly pushes (`finalise_link`'s `busy`).
  - A commit without the flag is modelled to the second as before (checked on master's
    six runs).
  - **On master's runs it gives** 12h24m → 12h01m at 7 MB/s, 7h44m → 7h43m at 30,
    7h11m → 7h10m at 60, and 7h07m at 120. With L3 on: 12h03m → 11h41m.
- **The measured stages don't see it.**
  - The link stage runs finalise's three runs after the overlapped runs, on a copy, as
    before.
  - The overlapped runs over the 7 MB/s link (`overlap_link_7_wall`) do run the turn.
    There the scholarly run waits behind the main run's merges after the split, so that
    row may move by a merge's length or so: the sample's merges are well under a second.
- **Parity:** the published files can't change. Finalise's code and inputs are the
  same, only its start is earlier, and the test checks the build branches byte for byte.

## Uncertainty and negative findings

- **The rate.** At 7 MB/s, the one rate on record, the saving is ~24 min. Above 15 MB/s
  it is within a few minutes of nothing.
- **The merge wall at a release** decides which turn pays. Holding from the start
  would be worth −38 min at #173's central 1-min merge, but +67 min at 5 min. The built
  turn is −24 min across 0.3–7 min, so it was chosen for robustness.
- **Not modelled:** two pushes sharing the link's bandwidth, where the model's link is
  non-preemptive and FIFO. The scholarly run's pushes don't stop for a main group that
  closes mid-push, so the main set's last pushes can end a push's length later
  (~5 min at 7 MB/s) than modelled.
- **Not measured:** none of this is a release's figure. The CI bench shows the model's
  rows, and only the restart test shows the orchestration working. The sample's runs
  are too short for the turn to move a measured row beyond noise.
- **Rejected along the way:**
  - the gate alone (−5 min);
  - main-first priority without holding (0 beyond the gate);
  - holding from the start (fragile);
  - `local` (+5 min over the built design, for a new compaction source path).
