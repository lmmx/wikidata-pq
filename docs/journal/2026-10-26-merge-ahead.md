# 2026-10-26: A group's merge taken out of its upload thread (L5, merge ahead)

L5 of `2026-10-25-release-critical-path.md` (#167), in lmmx/wikidata-pq#171.

**Outcome: not merged; #171 closed.** At a release a group's merge is ~1 min during the split, and L5 then saves ~0 at every link rate (the verdict below). The change stays on branch `pool/merge-ahead` in case a merge timed at a release is ever much slower than this estimate.

Until now, `pool._Upload` ran a group's whole close in one thread: merge, push, verify, clean-up. A set had one group at a time. So the set's next group couldn't merge until its last push had ended, and the link sat idle through every merge. At 7 MB/s the link is a release's critical path for its first ~5h, so that idle time lands on the release's wall.

## The change

- **The pool** (`pool._Uploads`) keeps the groups being closed in order. The first is pushed or verified, and up to `MERGE_AHEAD` (1) more come after it. The next group's close starts once the one before it has *merged*. Before, it started once that group was *done*.
- **The close** (`push.close_group`) calls two hooks, both no-ops outside a pool's upload thread:
  - `pool.merged()`, once its merge is done;
  - `pool.push_turn()`, before its push, which waits until the group before it is done and raises if that one failed.

  So merges run one at a time a set, and pushes and verifies run one at a time a set, in order. The uploads are the same files and commits, checked by the same `verify_group`; only their timing changes.
- **The pool wakes** on a pipe when a group merges or its close ends. Before, it noticed an upload's end only at the next chunk's end, so the next group could wait a chunk's time with the link idle.
- **The clean-up** deletes only its own group's staged files. Before, it removed each table's whole staging dir, which would now delete the next group's merged files. Its push and verify would then find no files and check nothing, with no error. It removes empty key dirs only while no merge is running in the process, because a merge lists the dirs once and makes only the ones it didn't find.
- **A close that doesn't call the hooks** (bench/pipeline.py's, which merges and moves the staged files itself) runs one group at a time, as before. Its next group's close starts only once its thread has ended.

## The ledger and restarts

A group merged ahead records `closed` and `merged` only when its turn to push comes. So the ledger still has one unfinished group at a time, and `unfinished_group`, main.py's resume and finalise's check are unchanged.

A run killed with a group merged ahead (N+1) resumes like this:
1. main resumes the group before it (N) from the ledger.
2. The pool forms N+1 again from its chunks. They are still at `PARTITION`, so `open_chunks` gives them, and the group rule is fixed by the chunks alone.
3. N+1's merge runs again. It resumes from its own files: each partitioned table's marker, and each glued key whose inputs are gone and whose staged file exists.

A different grouping isn't possible, but if one happened, it would fail loudly ("partition files missing") rather than upload something different.

The new kill points (`scripts/test_group_restart.py`) hold N's upload or verify until N+1 has merged ahead, so the kill finds both groups at once:

| kill point | at the kill | rerun |
|---|---|---|
| N+1's merge, mid table | N `merged` (its upload held); N+1 partly merged, unrecorded | ok |
| N+1's merge, mid table | N `pushed`, not verified | ok |
| N+1's claims glue | N `pushed` | ok |
| N's upload, mid table | N+1 merged ahead, waiting for its turn | ok |
| N's verify | N+1 merged ahead | ok |
| N's clean-up | N `verified`; staging holds N's files and N+1's | ok: N+1's survive |
| N+1's `closed`, at its turn | N `done`, N+1 `closed` with its files staged | ok |
| none (uninterrupted, held) | | the same 100 uploads and 41 published files |

A run where N+1 didn't merge ahead within a minute fails, and so does one where N+1 uploads before N is `done`. With `MERGE_AHEAD = 0` the held run fails ("the second group did not merge ahead"), so the test does see the overlap. All 22 kill points pass, the 14 before plus these 8.

`scripts/test_overlap_restart.py` passes every kill point. Its new one is "main run: mid upload, the next group merged ahead": the first upload is held until a second group has merged, then the run is killed mid-upload. `test_pool.py` has three new tests:
- the order and the one-ahead bound;
- no merge-ahead without the hooks, or with a bound of 0;
- a failed push stops the group merged ahead.

`test_keep_groups.py` and `test_merge.py` pass.

## Bounds

- **Disk:** at most one merged-but-unpushed group a set. A group merged ahead holds its staged files *in place of* its chunks' partition and processed files, which the merge deletes as it goes. Those inputs were on disk before this change too, while the group waited to close. So the only extra disk is one table's staged files that haven't yet replaced their inputs, while N+1 merges, during N's push.
- **Memory:** merges never overlap (N+1 starts once N has merged). What now overlaps a merge is the previous group's push, an upload bound by I/O. The bench's whole-tree peak for the overlapped runs is 3.178 → 3.186 GB (± 0.090), no change.

## Measured (CI bench, run 40260561553, 3 rounds, 718e028 → f37a1c9)

| | before | after | change |
|---|---|---|---|
| the split and both pipelines at once, pushes over a 7 MB/s link: wall | 28.86 s ± 1.77 | 28.30 s ± 1.75 | **−0.56 s (−2.0% ± 1.4%)** |
| of it, the link busy (up) | 17.43 s | 17.43 s | the same bytes |
| the link's busy fraction | 60.4% | 61.6% | +1.2 pt |
| the same, no link: wall | 13.28 s ± 1.60 | 12.80 s ± 1.53 | not beyond noise |
| peak RSS, at once, every process | 3.178 GB | 3.186 GB | not beyond noise |
| parity | passed | passed | |

A second run of the same code (40262409593, at 7450ea7, which adds only this entry) gave 29.00 s ± 2.00 → 28.54 s ± 2.05 over the link: −0.46 s (−1.6%), the same direction, but **not** beyond noise. So the gain on the sample is about 2%, at the edge of what three rounds can resolve.

Nothing else moved beyond noise except one row: finalise's three runs over a 60 MB/s link, −2.7% ± 1.9%. Finalise doesn't run the pool or a group close, so that row isn't this change.

The gain on the sample is small, and that is expected. The bench's groups are one chunk each, and their merges take well under a second of real time. Their pushes are a release's bytes compressed by ~1,020 into the sample's time. A sample merge is therefore much shorter, relative to its push, than a release's would be, so there is little idle link time to hide. What the bench does show is the mechanism working: the same bytes uploaded in less wall, with the link busy a larger share of it.

## Modelled: a release (not measured)

A small model of both sets' build-branch uploads, coupled (the close in one thread) against merged ahead. Its assumptions:
- 60 groups (30 a set, #167) of 124.3 GB / 60 each;
- the groups closable evenly through the split's 3h42m;
- one link shared by both sets, first come first served;
- a group's merge taking `m` during the split, on the ~0.3–0.6 core it leaves, and `m/4` after it;
- 0.2 min of verify and clean-up a group;
- with merge-ahead, one merge at a time a set and at most one group merged ahead.

What it gives for the uploads' end:

| link | a merge during the split | uploads' end, coupled | merged ahead | saving | link busy, coupled → ahead |
|---|---|---|---|---|---|
| 7 MB/s | 3 min | 5h07m | 5h07m | 0 | 97% → 97% |
| 7 MB/s | 6 min | 5h34m | 5h10m | −24 min | 89% → 96% |
| 7 MB/s | 9 min | 6h15m | 5h13m | −62 min | 79% → 95% |
| 30 MB/s | 3 min | 3h45m | 3h45m | 0 | 31% → 31% |
| 30 MB/s | 6 min | 3h46m | 3h46m | 0 | 31% → 31% |
| 30 MB/s | 9 min | 4h18m | 4h00m | −19 min | 27% → 29% |
| 60 MB/s | 3 min | 3h44m | 3h44m | 0 | 15% → 15% |
| 60 MB/s | 6 min | 3h45m | 3h45m | 0 | 15% → 15% |
| 60 MB/s | 9 min | 4h07m | 3h58m | −9 min | 14% → 14% |

#167's own model put this at 0 to −52 min at 7 MB/s and ~0 at 30 MB/s and above. This one agrees to within a few minutes on the coupled walls: 5h13m, 5h27m and 6h11m there.

**At 7 MB/s** the saving depends entirely on a group's merge wall at a release, which still isn't measured. Merges of 3 min or less are hidden either way, since the other set's push fills the link. At 6–9 min, merging ahead takes 24–62 min off the uploads' end, and that end is on the release's path.

**At 30 and 60 MB/s** the uploads' end (~3h45m) comes before the runs' own end (~4h11m, #167). So the saving reaches the release's wall only when merges are slow enough (9 min) that the coupled uploads run past the runs' end. Even then, merging ahead is limited by one merge at a time a set.

**Still unmeasured:** a group's merge wall at a release, under nice 10, during the split. That one figure decides L5's real size, between 0 and about an hour at 7 MB/s.

## A group's merge wall at a release (m), from evidence

The model above turns on `m`, a group's merge wall during the split. Below, `merge_group` is timed locally and scaled to a release group, by what its cost actually grows with.

### Measured: merge_group on the quick tier's files

**Method.** The quick tier's chunk files (c0, r50, s30, s90) were split into chunks of 2,000 and of 500 entities. Each set's pipeline then ran with the group close a no-op, leaving every chunk partitioned. `merge_group` was then timed alone, on an idle machine, on groups of k consecutive chunks, each sweep from a fresh hard-linked copy. CPU is the process's (its threads included); the data is in page cache.

| main set, as groups of | groups | wall | CPU | group bytes | keyed runs | keys (staged files) |
|---|---|---|---|---|---|---|
| 1 chunk (500 entities) | 34 | 29.0 s | 37.5 s | 98 MB | 32,442 | 32,510 |
| 4 chunks | 8 | 11.5 s | 15.5 s | 81 MB | 31,814 | 10,581 |
| all 34 (one group), mean of 2 | 1 | 4.2 s | 4.9 s | 75 MB | 32,442 | 2,324 |
| all 9 at 2,000 entities (one group), mean of 2 | 1 | 4.9 s | 3.3 s | 70 MB | 10,934 | 2,324 |

The scholarly set's whole group (33 chunks of 500) is 35 MB in 0.4–0.5 s, with 176 keys.

**Its cost is not linear in bytes.** It is per key and per run, where a run is a key's rows in one chunk: a sidecar row, one of the ~11–12M per-chunk partition files a release had before #140.

- **Per key: ~0.8 ms wall, ~1.1 ms CPU.** Same runs and bytes, keys 32,510 → 2,324: 37.5 → 4.9 CPU-s. A profile puts this in two `mkdir`s and an `rmdir` a key (~0.15 ms each on this file system), `os.replace`, `pq.read_metadata` and the key's footer flush.
- **Per run: ~75 µs CPU at the margin (20–130 µs across the passes).** Same keys and bytes, runs 10,934 → 32,442: 3.3 → 4.9 CPU-s. This is the Python footer parse of each run's row groups (`rowgroups.py`: `__init__` per run and per column chunk, `_uvarint`), under the GIL, so `_copy_built`'s threads don't spread it.
- **Bytes are cheap.** The claims glue (`pq_glue`) copies 47–48 MB in 0.11–0.26 s: 190–420 MB/s on one core, from page cache. The keyed tables' bytes are copied as whole row groups, a small cost beside their runs.
- **The per-group fixed cost** is ~0.1 s.

The bench's own record agrees in kind. `merge_group` is 1.4 CPU-s of the quick tier's pipeline (97%), on its fewer, larger chunks (a chunk a split part), so fewer runs.

### Scaled to a release group

A release group: 2.07 GB uploaded (124.3 GB / 60), ~5,400 keys (#167), and ~192k keyed runs (~11.5M per-chunk partition files at a release, #140, over 60 groups: ~406 chunks of 10,000 entities × ~470 keys a chunk).

| part | per unit | a release group |
|---|---|---|
| keys | 0.8–1.1 ms | 4–6 s |
| runs | 20–130 µs (75 central) | 4–25 s (14 central) |
| bytes, glued (claims, entities, ~1.3 GB) | 2.4–5 ms/MB | 3–7 s |
| bytes, keyed (~0.8 GB), I/O beyond page cache | | 0–10 s of wall, not CPU |
| **CPU, one core** | | **~12–40 s (≈25 central)** |

On a free core, then, a release group's merge takes about half a minute, plus up to ~10 s of I/O if its inputs have left the page cache.

During the split, a merge runs at nice 10 on what the split leaves: ~0.6 of a core (the split is 3.4 of 4 cores busy on the sample), shared by both sets' merges when they coincide, so 0.3–0.6 core each. Against a saturated split, the CFS share is 110/1024 of a nice-0 thread, about 0.1 core.

| a group's merge during the split | share of a core | m |
|---|---|---|
| central (25 CPU-s) | 0.45 | **~1 min** |
| range (12–40 CPU-s, + 0–10 s I/O) | 0.3–0.6 | **0.3–2.4 min** |
| worst case: CPU-bound split, CFS share only | ~0.1 | 2–7 min |
| after the split, a core each | ~1 | 0.3–0.8 min |

**What this means for #167's estimate.** #167 put a merge at 4.5–9 min during the split (2.7 CPU-min a group, scaled by entities, at 0.3–0.6 core). By runs and keys it is ~25 CPU-s a group, about a sixth of that. #167's "half that by bytes" pointed the same way, though bytes are not what the cost scales with.

### What the model gives at these m

The model above, with the merge after the split at ~1 min:

| link | m = 1.5 | 2 | 2.5 | 3 | 4 | 5 min |
|---|---|---|---|---|---|---|
| 7 MB/s | 0 | 0 | 0 | 0 | 0 | −6 min |
| 30 MB/s | 0 | 0 | 0 | 0 | 0 | 0 |
| 60 MB/s | 0 | 0 | 0 | 0 | 0 | 0 |

At 7 MB/s, a merge up to ~4 min is already hidden without L5: while one set merges, the other set's push keeps the link busy (97% busy either way).

### Verdict

**At a release, m is ~1 min (0.3–2.4 min) during the split, and L5 saves ~0 at 7, 30 and 60 MB/s.** It pays only if a merge gets less than ~0.15 core during the split (m ≥ ~5 min), which is this estimate's worst case: a CPU-bound split leaving the runs only their CFS share. There it saves minutes at 7 MB/s, up to −24 min at m = 6.

The change is correct, and its restart cases are tested. It costs ~nothing in disk or memory, and the bench's link stage shows the mechanism on the sample (−2.0% ± 1.4% in one run; −1.6% in a second, not beyond noise). But the release figure it was meant to move is modelled at ~0. It is insurance against slow merges, not a gain.

What would change this: a measured merge wall at a release during the split. The per-run cost (Python footer parsing, ~75 µs × ~192k runs) is the part that grows at a release, and also the one that could be cut if merges ever matter.
