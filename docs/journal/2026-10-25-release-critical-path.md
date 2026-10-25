# 2026-10-25: A release's critical path at master, with the transfers no model sees

A profile of the whole release as it now runs, `just overlap-release` and then
`just release`. It follows the old map, `2026-10-25-architecture-trace.md` (#136), which
predates fusion, the group files' new layout, the sort's planning, claims_labels' three
changes and the stage overlap. It asks which segment bounds each stretch of a release's
wall, how much of each is the contract's floor, and what is left to start. The model is
in `2026-10-25-release-critical-path/critical_path.py`.

## Current State

- **Docs only.** Nothing in `src/`, `reshape/` or `bench/` changed. Nothing was measured
  here: every input is a figure on record (Sources).
- **Master is a9424a6** (#164). Its own push bench (17.0.0) was still running. The
  figures are #164's and #159's bench comments, and #165's recomputation of #159's run
  under bench 17.1.0's release wall (not merged, model only).
- **The findings:**
  1. **At the one upload rate on record (7 MB/s), a release is bound by the network, not
     the CPU.** The build branch's 124 GB and the published 129 GB need about 10h09m of
     link time. That is the release's floor at that rate, against 6h38m of compute.
  2. **About 5h30m of a release's wall is in no model.** Bench 17.1.0 puts a release at
     7h42m. It leaves out finalise's transfers. Finalise downloads the build branch again
     (unless the group files are kept), and it uploads each table's sorted files only
     after writing them, with the CPU idle. With them, a release at master is about
     **13h13m** (13h06m–13h27m over the download rate).
  3. **The largest item left is when finalise uploads, not what it computes.**
     `commit_table` runs after each table's `write_table`, so finalise's 2h27m of compute
     and its 5h08m of upload run one after the other. Overlapping them (L1) saves about
     **2h32m** at 7 MB/s, and still 36 min at 60 MB/s.
  4. **Every CPU item left is under about 10 min at a release.** At 7 MB/s it is worth
     nothing on the wall, since the split phase is under the link. The split is ~90% the
     floor, finalise ~85–90%, and claims_labels is ~5 min since #164.
- **The verdict: one thing is worth starting, and it isn't CPU.** Finalise's transfers
  can be scheduled under its compute (L1, with L3 and L5). The bench can't see any of
  it, so a model term (B1) comes first or alongside. The user's measure of the Hub's
  upload rate decides how large it is: L1 alone is ~20 min at 120 MB/s and ~2h30m at 7 MB/s. Nothing CPU-side is worth a new
  experiment beyond what is running (#163, `claims.avsc`).

## Sources

- **Bench 17.0.0, #164's comment** (run 40239408023; before a4d4937, after 8bf17d5, 3
  rounds):
  - **the ETA by stage:** split 3h26m59s, pipeline 1h23m11s (CPU at a release 2h42m27s),
    finalise 1h56m50s;
  - **decoding alone:** 1h51m43s;
  - **the split's CPU by stage, finalise's per-byte CPU by stage, and finalise's
    timeline.**
- **#159's comment** (run 40238218967): at once against one after another, 15.52 s
  against 17.49 s; peak RSS at once 3,064 MB; uploads at a release 4h55m59s at 7 MB/s.
- **#165's body** (bench 17.1.0, recomputed from #159's run, after = 0c61bcc):
  - the download ends at 1h58m57s, and the split at 3h42m30s;
  - 72% of the runs' CPU is done during the split, and the compute ends at 4h11m02s;
  - the uploads end at 5h19m12s;
  - finalise is 2h22m36s on the scholarly anchor;
  - **the release wall is 7h42m16s.**
  - "finalise's own transfers and claims_labels' build are in neither model."
- **The upload rate:** 7 MB/s, the main set's 45 GB of sorted claims committed in about
  1.8 h on the release machine (`2026-10-05-sort-speed.md`). It is the only Hub transfer
  rate on record.
- **The download rate:** the build branch's has none on record. 50–100 MB/s is assumed,
  E1's range in #136, so 75 MB/s is the central figure.
- **Sizes:**
  - **G**, the build branch's bytes, is 4h55m59s × 7 MB/s = **124.3 GB**.
  - **P**, the published bytes, is taken as G + 5 GB.
    - The sorted parts are about the group files' bytes: 44.7 GB of groups gives ~43
      GB of parts (#150).
    - claims_labels adds 3.19 GB for the main set (`docs/dataset_cards_metadata.json`)
      and ~1–2 GB for the scholarly set, which is not on record.
- **claims_labels at a release:** the main set's build is 8.7 CPU-min, and both sets'
  13–22 CPU-min (#164). On 4 cores that is ~5 min.
- **Code read:**
  - the orchestration: `main.finalise`, `sort_by_id.sort_table` and `commit_table`;
  - compaction: `compact.source_group_files`;
  - the upload thread: `pool._Upload` and `_next_upload`;
  - the group close: `push.core.close_group`, `merge_group` and `_copy_built`;
  - the recipes: `overlap-release` and `release` in the Justfile.

## The critical path at master

`critical_path.py` simulates the recipes:
- **The runs:** 60 groups (30 a set) close on 17.1.0's rule.
- **The uploads:** one shared FIFO link.
- **Finalise:** its compute, then its transfers as the code orders them.

It reproduces the uploads' end to within 9 min of 17.1.0's: 5h10m against 5h19m.

| # | segment | wall at master (7 MB/s) | CPU at a release | bound by | the contract's floor in it |
|--:|---|---|---|---|---|
| A | **download** of the bz2 (103.5 GB) | 0 → 1h59m, under the split | md5 ~3 CPU-min | the network in, 14.5 MB/s (3 connections) | all of it. Not on the path: the split is longer |
| B | **the split** | 0 → 3h42m (3h27m on #164's runners) | ~13.8–14.8 core-h | CPU: 4 cores, 3.4 busy on the sample | **~90%:** the lbzip2 decode 53% (1h52m of the 4 cores); write_table's one encode 22%; the parse (`write_chunk`) 6%; one Arrow build ~9%. **Above it, ~20 min:** the claims' second parse (~6 min, #163), routing and per-part fixed costs (`route_parts`, ~8 min, ~4 min of it per part), ~6 min other |
| C | **both sets' runs**, at nice 10 | 72% under B, then to 4h11m (+29 min) | 2h42m core-h, modelled | the cores B leaves idle (~0.6) | **none:** `merge_group` copies the split's encoded pages into group files. On the path only through its 29 min tail, and through C ↔ D below |
| D | **the build branch's uploads** (124 GB) | first close ~5 min → 5h10m–5h19m | ~0 | **the network up, 7 MB/s.** Busy 4h56m of the span, waiting on closes ~9 min | all of it (kept by decision) |
| E1 | **finalise: G downloaded** again (unless `WIKIDATA_KEEP_GROUP_FILES`) | +20–41 min | ~0 | the network in, 50–100 MB/s (unknown) | none: those bytes were verified on this machine before deletion |
| E2 | **finalise: compute,** three runs (main, scholar, main) | +2h22m modelled (real ~1h50m) | ~9.5 core-h | CPU, ~95% of 4 cores at a release (#103); memory budget (3 bucket sorts in ~12 GB) | **~85–90%:** the claims are ~80% of it, and 76 of their 88.6 min is the sort's decode, the sort, the encode and the kept decode (#150). **Above it:** bucketing ~10 min, read amplification ~2 min, the other tables' per-file stages ~1–3 min |
| E3 | **claims_labels'** build, in the 2nd and 3rd runs | +~5 min | 13–22 CPU-min | CPU, memory (jobs' peak 11.2 GB) | ~all: one join, written as its published files (#160, #164) |
| E4 | **finalise: the published upload** (129 GB), each table's after its write | **+5h08m**, CPU idle | ~0 | the network up, 7 MB/s | all of it. **But it runs serially with E2, which it needn't** |
| | **the release** | **~13h13m** (13h06m–13h27m) | | | **floor: ~10h09m of link at 7 MB/s; 6h38m of compute** |

- **Through 5h10m** the path is the link (D), at 7 MB/s.
  - Nothing in B or C shortens it, except C's place in the upload thread (L5).
  - The 17.1.0 model agrees: the uploads end 1h03m past the compute and the last
    group's own upload.
- **After it,** the path is finalise as a strict sequence: download, compute, upload,
  per table. That sequence is the 5h30m no model sees.
- **At a fast link** (≥30 MB/s) the path is the compute instead:
  - B (3h42m), C's tail (29 min), E2 and E3 (2h27m), then E4, still serial (43 min at
    60 MB/s);
  - B and E2 are each ~90% floor.

### Peak memory and disk on a 16 GB, few-hundred-GB laptop

- **RSS:**
  - **the overlap phase:** 3.06 GB on the bench. At a release, about 4–5 GB: the split
    worker's 2.68 GB, the runs' workers 0.24 GB each, decoders and parents.
  - **finalise:** the bucket sorts are 3.31 GB each, three at once under the memory
    budget (~10–12 GB). The bench's "20.1 GB, all processes" assumes 6 workers, without
    the budget.
  - **claims_labels' jobs:** 11.2 GB at a release (#164). That is the closest to a 16 GB
    laptop's limit, beside the bucket sorts.
- **Disk:**
  - **The dump** (103.5 GB) stays for the whole release. Nothing deletes it.
  - **With kept group files,** they accumulate to G (124 GB) by the uploads' end. The
    main set's claims sort then adds ~25 GB of fragments and 43 GB of sorted parts:
    **~300 GB peak**.
  - **Without kept files,** it is ~235 GB.

### Where I trust the model, and where I don't

| figure | trust | why |
|---|---|---|
| the split's ETA | **±10%, and the runner mix** | CPU per bz2 byte, scaled by bytes, over 4 cores. Its two runs read 3h27m and 3h42m on different runners. It assumes 4 busy cores where the sample shows 3.4, so the split alone could be up to +18%. Under the overlap, the runs fill that gap, and the measured at-once ratio (0.89) puts the compute's end at 4h11m–4h42m |
| decoding alone | **good** | lbzip2 by itself on the slice, its wall scaled by bytes |
| the download | fair | one record (#109), 14.5 MB/s. Off the path either way |
| the runs' CPU (pipeline ETA) | **poor** | scaled by entities and the sample's wall/CPU. It is now nearly all `merge_group` (1.4 of 1.5 CPU-s), a byte copy that entity scaling overstates (the other tables ~3×, #131). It matters only for C's 29 min tail and for L5 |
| G (the build branch's bytes) | good | each set scaled by its chunk files' bytes. 4.9 h at 7 MB/s, as on record |
| **the 7 MB/s link** | **weak, and it decides the ranking** | one commit of sorted claims on the release machine, three weeks ago. Unknown: whether two sets' uploads share it or each get it; the per-file overhead of the build branch's ~140–160k small files; Xet's deduplication. The published parts are written with content-defined chunking (`compact.py`), so a second sorted release may upload much less than P. A first release under the new layout cannot |
| finalise's compute | **good for the claims** (within 3%, #150); fair for the rest | the other tables are now scaled by their own rows (#155) and their per-file stages by files (#152). CI runs in page cache. On NVMe the bucket path stays CPU-bound (#150), and on a disk under ~150 MB/s it doesn't |
| G's download, the published upload, claims_labels, cards | **not modelled** | 17.1.0 says so. They are E1, E3 and E4 above |

## What is left above the floor, ranked

The release's minutes are from `critical_path.py`, against today's ~13h13m.

| orchestration | 7 MB/s | 15 MB/s | 30 MB/s | 60 MB/s | 120 MB/s |
|---|---|---|---|---|---|
| bench 17.1.0 (no finalise transfers) | 7h33m | 6h45m | 6h35m | 6h34m | 6h34m |
| **today**: G downloaded, commits after each table's write | **13h13m** | 9h42m | 8h19m | 7h43m | 7h24m |
| L3: kept group files | 12h46m | 9h14m | 7h51m | 7h15m | 6h57m |
| L1: parts uploaded as written | 10h41m | 7h19m | 7h08m | 7h07m | 7h06m |
| **L1 + L3** | **10h28m** | 6h56m | 6h40m | 6h39m | 6h39m |
| L2 + L3: finalise's compute before the uploads end | 11h49m | 9h02m | 7h50m | 7h14m | 6h56m |
| L1 + L2 + L3 | 10h18m | 6h46m | 6h39m | 6h39m | 6h39m |
| floor: the link alone (first close + G + P) | 10h09m | 4h46m | 2h26m | 1h15m | 0h40m |
| floor: the compute alone | 6h38m | 6h38m | 6h38m | 6h38m | 6h38m |

### L1. The published upload overlapped with finalise's compute

- **Saving:**
  - **−2h32m at 7 MB/s;** −2h23m at 15 MB/s, −1h11m at 30 MB/s, −36 min at 60 MB/s;
  - with L3, **−2h45m** at 7 MB/s.
  - It is the only item worth over an hour at any upload rate under ~30 MB/s.
- **Mechanism:** `sort_table` runs `write_table` → `commit_table` → `verify_table` per
  table, and `main.finalise` runs the tables in turn, so the link and the CPU take turns.
  Two forms:
  - **L1a:** each table's `commit_table` and `verify_table` go to one background
    committer, in order, while finalise goes on to the next table and the next set. The
    cards and the run's end join it.
  - **L1b:** each part is uploaded as soon as `_check_written` passes
    (`HfApi.preupload_lfs_files`, content-addressed, so a rerun skips what is up). The
    commits stay as today: the same files, the same batches, a key's additions and
    deletions together.
  - L1b starts the link ~45 min sooner: at the first claims part, not after the main
    set's whole claims sort. With L2 the two are equal at 7 MB/s.
- **Risk: medium.**
  - **The ledger:** a table's `committed` can now follow the next table's `written`.
    Each table's stages stay in order and resumable (`_key_done` against the Hub).
  - **The disk:** sorted parts wait for the link, up to ~70–90 GB at 7 MB/s. They replace
    sources deleted as each table is written, so the peak stays near today's ~300 GB. It
    needs a bound: write no further while more than N GB is unsent.
  - **No contract change:** the same published bytes and commits, and the checks
    untouched.
- **Files:**
  - `src/wikidata/sort_by_id.py` (`sort_table`, `commit_table`, the parts' upload);
  - `src/wikidata/main.py` (`finalise`'s loop; the committer joined before the cards and
    `FINALISE_DONE`);
  - `src/wikidata/hub.py`, `src/wikidata/config.py` (the unsent-bytes bound);
  - `scripts/test_finalise.py`.
- **Depends on:** nothing in code. The CI bench can't show it (no transfer), so it needs
  B1, or a stated model, to merge under the campaign's rules.

### L3. The verified group files kept by default (`WIKIDATA_KEEP_GROUP_FILES=auto`)

- **Saving: −20 to −41 min at every rate** (G at 100 to 50 MB/s). The re-download is
  serial in `compact.source_group_files` today.
- **Cost:**
  - +62 GB a set held from its pipeline to its finalise, ~124 GB at the uploads' end;
  - `auto` keeps a group only while 150 GB stays free, so a short disk falls back to the
    download.
- **Risk: low.** #143 merged it opt-in. It is a default, and a disk trade for the user to
  weigh.
- **Files:** `src/wikidata/config.py`, the README's release section.
- **Depends on:** the user's decision. It is a precondition for L2.

### L5. A group's merge taken out of its upload thread

- **Saving: 0 to −52 min at 7 MB/s; ~0 at ≥30 MB/s.**
- **Why:** `pool._Upload` runs a group's close (merge, push, verify) in one thread, one
  group a set at a time. So a set's next merge waits for its last push.
  - During the split, a merge runs at nice 10 on the ~0.6 core the split leaves, shared
    by two sets.
  - When both sets merge at once, the link idles.
- **How much it idles** depends on a merge's wall at a release, which nothing measures:

  | a group's merge during the split | uploads' end, coupled (today) | decoupled |
  |---|---|---|
  | 3 min | 5h13m | 5h13m |
  | 6 min | 5h27m | 5h16m |
  | 9 min | 6h11m | 5h19m |

  17.0.0's 2h42m CPU over 60 groups is 2.7 CPU-min a group: ~4.5–9 min at 0.3–0.6
  core. Scaled by bytes rather than entities, it would be about half that.
- **Mechanism:** one thread merges, one pushes and verifies, with one merged group ahead
  a set. The group ledger already records `merged` apart from `pushed`, so a restart is
  unchanged.
- **Risk: low.** The disk holds one more staged group a set (~2 GB).
- **Files:** `src/wikidata/pool.py` (`_Upload`, `_next_upload`), `src/wikidata/push/core.py`
  (`close_group` in two calls), `scripts/test_split_follow.py`.
- **Depends on:** nothing. Invisible to the bench without B1's emulated link.

### L2. Finalise's compute started before the build branch's uploads end

- **Saving:**
  - **−57 min at 7 MB/s alone** (with L3);
  - **only −10 min once L1 and L3 are in,** since the link is then the floor;
  - 0 at ≥30 MB/s, where the uploads end with the compute.
- **Can finalise start on groups already uploaded?** Not table by table:
  - the claims' sort is global;
  - every group holds every language and site key.
  - **What it can do** is start once a set's groups are all merged and kept, before they
    are uploaded. Only its commits wait for the set's groups to be verified:
    `_key_operations` deletes the group files in the publishing commit, so they must be
    on the Hub.
- **Mechanism:**
  - `main.finalise`'s guard (`unfinished_group`) moves from the compute to
    `commit_table`;
  - `overlap-release` runs each set's finalise compute once its run's merges are done,
    in the ~1h of idle cores between the compute's end and the uploads' end.
- **Risk: medium-high.**
  - The build branch is the restart point, and a set would be sorted before it is
    complete.
  - A finalise moved to another machine mid-way has no group files to fall back on.
  - claims_labels still needs both sets' labels.
- **Files:** `src/wikidata/main.py`, `src/wikidata/sort_by_id.py`, the Justfile,
  `src/wikidata/push/groups.py`.
- **Depends on:** L3, and on L1 to be worth more than ~10 min.

### B1. The bench: an emulated link and finalise's transfers (the bench's owner)

- **What it is:**
  - the overlap's stand-in Hub sleeps each push for its bytes at the release's scale over
    a stated rate, which shows L5 as measured;
  - `release_wall` adds G's download and P's upload in finalise as the code orders them,
    which shows L1, L2 and L3 as modelled.
- **Saving:** none itself. Without it, the release's largest items can't pass the
  campaign's "an end-to-end figure improves" rule except by a stated model.
- **Files:** `bench/overlap.py`, `bench/release.py`, `bench/CHANGELOG.md`.
- **Depends on:** #165 (17.1.0) merging. Its owner should take it.

### The rest: under ~10 min each, and ~0 at 7 MB/s

| item | release minutes (fast link; at 7 MB/s) | status |
|---|---|---|
| E2: the split writes the group files, so `merge_group` goes | −29 min (C's tail, a poor model); 0 | High risk: an appending writer for ~5,400 open keys, restart a group's chunks. Not worth it before L1 |
| the claims' second parse in the split | ~−6 min; 0 | **#163, running** |
| per-part fixed costs in the split (~40 ms × 23,035) | ~−4 min; 0 | queue 3; needs a release-shaped measure |
| the claims bucketed at write time (E5) | ≤ −10–12 min; 0 | It moves a decode and an encode of the claims into the group close to save bucketing's 10 min. Not worth it |
| finalise's other tables' per-file stages | −1–3 min (the model's 21 min is ~15–50× overstated, #150); 0 | Nothing to do |
| claims_labels' join build side (queue 4) | +1–1.5 min, for −20–30% of 11.2 GB | Memory only; worth it if a laptop runs short |
| the dump deleted once the split is done (L6) | 0 min; −103.5 GB of peak disk | A user option: the dump is only the split's restart point |
| gz from the mirror (not authorised as default) | −1h55m of the split at a fast link; 0 at 7 MB/s | A per-release option |

**The claims' bucket path after #158 and #164:** unchanged.
- Neither touches it: #158 changed the four built tables' group files, and #164
  claims_labels.
- Finalise's per-byte CPU at master is ~79% floor stages:
  - the encode, 15.6 CPU-s;
  - `_check_written`, 8.5;
  - `_read_bucket`, 6.8;
  - `_sorted`, 4.9.
- Bucketing (`_bucket_source` and the fragments' IPC) is 4.2 CPU-s, 9%, ~10 min at a
  release, as #150 found.

**claims_labels after #164:** about 5 min of a release's wall. Its refs are 22 s (#139),
its build is one join a batch of languages, and it is written as its published files
(#160). Its publish (~5 GB, ~12 min at 7 MB/s) is in P and is L1's to hide.

## The verdict

- **Something is worth starting:** L1, with L3's default (a user decision) and L5
  beside it. L1 and L3 are 2h45m of a release at 7 MB/s, and 45 min–1h40m at
  30–120 MB/s. L5 adds up to ~50 min at 7 MB/s.
- **Hold L2** until L1 is in. It adds ~10 min then.
- **B1 first or alongside:** none of these can show on today's bench.
- **The single most useful number is the Hub's upload rate.** The user's measure of the
  Hub's upload and download rates is already pending. With them, check whether Xet
  deduplicates a second sorted release's parts.
  - At 7 MB/s, L1 is ~2.5 h.
  - At 60 MB/s, it is ~36 min.
  - If deduplication cuts P severalfold, it shrinks with it.
- **The CPU side is done in all but name.** Every item left is under ~10 min at a release
  and worth nothing on the wall at 7 MB/s. The split is ~90% the floor (lbzip2, one
  parse, one build, one encode), finalise's claims ~86%, and claims_labels ~5 min.
  Beyond #163 and the `claims.avsc` work already running, no CPU experiment is worth
  starting.

## Missing

- **Hub rates:**
  - the upload and download rates on the release machine;
  - whether two uploads share the link;
  - the build branch's per-file overhead (~140–160k files);
  - Xet's deduplication of a second sorted release.
- **A group's merge wall at a release under nice 10,** which sizes L5.
- **The scholarly set's published bytes and claims_labels' size,** taken here as G + 5 GB.
- **Master's own 17.0.0 push run** (a9424a6), still running. #164 changed only
  claims_labels, which is in neither ETA, so nothing above should move.
