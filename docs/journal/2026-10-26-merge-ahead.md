# 2026-10-26: A group's merge taken out of its upload thread (L5, merge ahead)

L5 of `2026-10-25-release-critical-path.md` (#167), in lmmx/wikidata-pq#171.

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
