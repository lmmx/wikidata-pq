# 2026-10-26: Is "7 MB/s" the uplink, or how we upload?

Every slow-link figure in the campaign rests on a Hub upload rate of 7 MB/s, "the one rate
on record". At that rate a release is network-bound: bench 17.9.0 models master at
10h50m, against 6h25m–6h51m at 30–120 MB/s. If 7 MB/s were one connection's rate, or
latency-bound, then configuration alone could lift it: Xet's concurrency, more commits in
flight, the two sets' uploads truly concurrent, or larger batches.

This entry looks for:
- where the figure came from;
- how our uploads use concurrency today;
- what Xet's controller would have done during the measurement;
- what each kind of link would give the release.

**Nothing here called the Hub.** The model is `2026-10-26-upload-throughput/model.py`. The
measurement is `2026-10-26-upload-throughput/throughput_measure.py`. It is **the user's
to run**, and no agent has run it.

## Current State

- **Docs only.** Nothing in `src/`, `reshape/` or `bench/` changed. No Hub read, upload or
  API call was made.
- **The versions read:**
  - **huggingface_hub 1.32.0**, the wheel from PyPI, as `uv.lock` pins it. Below, `hub:`
    is a path in that package.
  - **hf_xet 1.6.0**, from its Rust source: huggingface/xet-core at tag `v1.6.0`
    (de71453), as #177 read it. Below, `xet:` is a path in that tree.
- **The findings:**
  1. **7 MB/s is one rough figure, not a timed measurement.** Its source is one line in
     `2026-10-05-sort-speed.md`: "claims took about 1.8 h each time (45 GB at about
     7 MB/s)".
     - **What it covers:** the main set's sorted claims of release 20260928, 86 parts of
       500 MiB (44.7 GB), on the user's release machine (20 cores, 126 GB RAM; its
       network isn't recorded), around 2026-10-04/05.
     - **How they went up:** `commit_table`'s two `create_commit` calls (50 parts, then
       36), each one Xet upload session at hf_xet's defaults. No HF_XET_* setting has ever
       been in the repo.
     - **What isn't recorded:** no timestamps, no log and no tool timed it.
     - **Everything since rests on it:** #167 took it as "the one rate on record", then
       `bench/link.py`, `bench/overlap.py` and every journal since.
  2. **That measurement already had all of Xet's default concurrency.**
     - **The ingest side:** 8 files were chunked at once, each cut into 64 MiB xorbs.
     - **The upload side:** an adaptive controller, from 2 up to 64 parallel uploads, each
       its own HTTP/1.1 connection.
     - **How it ramps:** it adds a stream every 0.5 s while a 64 MiB xorb at the next
       concurrency is predicted to take under 60 s.
     - **So a per-connection limit can't give 7 MB/s.** A stream faster than ~1.7 MB/s
       ramps to 64 streams, which is the uplink's rate or over 100 MB/s. A slower stream
       stays at 2, which is ≤3.4 MB/s.
     - **If the figure is right, 7 MB/s was an aggregate bound:** the uplink, or a cap on
       the Hub's side.
     - **Not a per-connection one,** so `HF_XET_HIGH_PERFORMANCE` wouldn't lift the bulk
       of the bytes.
  3. **A second, weak figure agrees.** The scholarly set's first 7 groups of the same
     release closed in 2,994 s in all, merges and verifies included (2026-10-04, in
     `2026-10-03-release-20260928-run.md`).
     - **The bytes:** about 14.8 GB of group files, estimated as the bench scales them.
     - **The rate:** ≥4.9 MB/s through the build branch's path, with its ~2,800 files a
       group.
  4. **What configuration and concurrency can still win is our overheads.** The bench
     doesn't model them:
     - a commit's fixed cost;
     - a round trip per small file (each new file's first chunk is a global-dedup query,
       8 at once);
     - small tables that have only 1–4 xorbs, so only 1–4 streams;
     - the serial structure: a group's 6 tables one after another, finalise's ~147
       commits one after another, and the turn lock after the split.

     At a 7 MB/s uplink, with plausible costs (c0 = 5 s, q = 0.2 s), the model puts
     master at **11h17m, against the bench's 10h50m**, and between 11h01m and 12h12m
     across the ranges tried. Hub-API-neutral changes recover most of that: **10h52m**,
     modelled.
  5. **If the user's sweep shows a per-connection limit after all,** the gain is large:
     - **the rates:** 14 MB/s at 2 streams and 28 at 4;
     - **the walls:** 8h01m and 6h56m, against 10h50m;
     - **where it would come from:** mostly the defaults already, not new settings
       (finding 2), so the record would be what's wrong.
- **The verdict:** don't plan on configuration collapsing the network-bound regime. The
  evidence points to the uplink, or a Hub-side cap. Two things are worth doing:
  - **the free, offline check first:** hf_xet's own log on the release machine records
    the concurrency the controller settled at and its predicted bandwidth;
  - **then the sweep,** which settles it.
  
  The low-risk changes below are worth 4–14 min each at 7 MB/s, ~25 min together,
  modelled, and about 10 min together at fast links. Two of them can wait for the sweep: high-performance mode for the pipeline's
  runs, and a group's tables pushed at once.

## 1. Where 7 MB/s came from

| what | where | what was measured | how | concurrency | machine |
|---|---|---|---|---|---|
| **"45 GB at about 7 MB/s"** | `2026-10-05-sort-speed.md:132-133`, added in 5ad5f96 (`git log -S "7 MB/s"`: its first appearance) | the main set's claims (86 × 500 MiB parts, 44.7 GB), uploaded twice: compaction's commit, then the sort's. "about 1.8 h each time" | not stated. No timestamps, log lines or ledger times are recorded. 44.7 GB / 1.8 h = 6.9 MB/s | `commit_table`: 2 `create_commit` calls, one after another (≤50 additions each, `COMPACT_COMMIT_MAX_ADDS`); each one Xet session at the defaults (below) | the release machine ("the host": 20 cores, 126 GB RAM, `sort-speed.md:7-21, 138`); link and location not recorded |
| #167's "the one upload rate on record" | `2026-10-25-release-critical-path.md:57-59, 136` | none: it cites the line above, rated **weak** | | | |
| `bench/link.py:57-59` `LINK_RATES = (7e6, 60e6)`, `bench/overlap.py:45-46` `UPLOAD_RATE = 7e6` | bench 17.0.0 onwards | none: "the one on record (2026-10-05-sort-speed.md)" | | | |
| every later journal's "at 7 MB/s" | `docs/journal/2026-10-26-*` | none: modelled at it | | | |
| the scholarly groups' closes, 2026-10-04 | `2026-10-03-release-20260928-run.md:31-32` | 7 groups' closes stopped processing for 326, 305, 316, 314, 307, 1,096 and 330 s (2,994 s), merge + push + verify inline. The bytes aren't recorded | chunk audit mtimes | one group at a time; a group's tables one after another; an `upload_folder` a table | the same machine |

- **The second line, as a rate:**
  - **The bytes:** 2,289 chunks of the scholarly set's 10,853, about 11.26 GB of its
    49.5 GB of source.
  - **Scaled:** the bench puts the scholarly set's group bytes at 65.2 GB for 49.5 GB of
    source (1.32 a source byte), so ~14.8 GB.
  - **The rate:** 14.8 GB in 2,994 s is **≥4.9 MB/s**, since merges are inside the
    closes.
  - **The caveats:** the bytes are estimated, and that release's group files were written
    by older code.
  - **What it rules out:** it is consistent with a ~7 MB/s link and ~2 min of merge and
    fixed costs a group. It can't rule out a faster link with slow merges.
- **hf_xet's version then:** not recorded. The first lock on record (5ad5f96,
  2026-10-07, two days later) pins huggingface_hub 1.32.0 and hf_xet 1.6.0, as today.
- **A correction:** `2026-10-26-schema-integrity.md` suggests a past release's push rate
  could come from "the group ledger [which] records when each group was merged and
  pushed".
  - **What the ledger holds:** `groups.jsonl` has `{"first", "last", "stage"}` and no time
    (`push/groups.py:84-89`). `sort_by_id`'s ledger likewise has none (`:129-134`).
  - **What's left:** only the file's mtime (its last append).
  - **The usable record is hf_xet's log** (§6).
- **Searched, nothing else found:**
  - `bench/CHANGELOG.md` and `bench/RECORDS.md`;
  - the README and `misc/pipeline_logs/` (2025 memory-debugging runs: no upload lines);
  - every `docs/journal/` entry;
  - `/home/user` (no state dirs, ledgers or xet logs of a release).

## 2. How our uploads use concurrency today

| path | where | process / thread | calls | files a call | Xet session | parallel uploads | serialised by |
|---|---|---|---|---|---|---|---|
| **a group's push** (build branch) | `push/core.py:429` `push_group`, from `close_group` (`:552`) on the pool's upload thread (`pool.py:55, 76`) | each set's run is its own process (`just overlap-release`). One upload thread a run, one group at a time (`_next_upload` returns the running upload while it's alive) | a table at a time: `create_repo`, `ensure_build_branch` (`list_repo_refs`), `_ensure_dataset_card` (`file_exists`), then `upload_folder`. Then `verify_group`: `get_paths_info` 100 paths a call (`:466`) | `upload_folder` with Xet is huggingface_hub's pipelined upload (`hub:hf_api.py:6073`): preupload 256 files a request, commits of 250 files at first, adapted 20–1000 to a 40 s target (`hub:_upload_pipeline.py:66-70`). A batch's commit overlaps the next batch's upload (queue of 1, `:355`), so ≤3 batches are in flight | **one a process** (`hub:utils/_xet.py:286`; `get_xet_session`), so its controller and its 8 ingestion slots are shared by every call in the run | a table's xorbs: a main group's claims file (~1.36 GB) has ~21; its labels (~220 MB in 621 files) ~4; descriptions, aliases and links ~1–3 each (small files share xorbs) | the tables one after another; then verify; then the next group's merge. **The turn lock** (#175, `push/turn.py`; `close_group` `:576`): after the split, the main run holds the turn while it closes groups back to back, and the scholarly run waits before each push. So after the split one set pushes at a time, and the link idles through that set's fixed costs and verifies. During the split both runs push at once, as two processes with two controllers |
| **finalise's commits** | `sort_by_id.commit_table` (`:1415`) on the `Committer` thread (#170, `:1533`) | one committer a finalise run. #179/#183's cross-run overlap: the next run's process starts writing once the one before has written its last table, so at run boundaries **two processes** commit at once, and the bench models them FIFO. The main set's first finalise also overlaps the scholarly run's last pushes (L2) | a table at a time: `_remote_files` (`list_repo_tree`), then `create_commit` per batch, one after another, then `verify_table` (another listing) | ≤50 additions (`config.COMPACT_COMMIT_MAX_ADDS`, `:253`) or ≤2,000 operations. At a release: claims 86 + ~99 parts → 2 + 2 commits; labels, descriptions, aliases and claims_labels ~600 keys → 12–13 commits each; links 955 → 20; ~147 commits in all | the run's one session; `create_commit` → `preupload_lfs_files` → one `new_upload_commit` for the batch (`hub:_commit_api.py:664`) | a claims batch has ~400 xorbs, so the controller decides (up to 64). A labels batch (~500 MB) has ~8, links (~60–280 MB) 1–4 | a batch's commit request and the next batch's upload don't overlap; the tables one after another |
| cards | `cards.py:284` `upload_file` | finalise | a table's README | 1 | | 1 | |

## 3. Xet's adaptive concurrency controller (hf_xet 1.6.0)

- **Start, minimum and maximum:**
  - **Uploads, by default:** start 2, 1–64 (`xet:xet_runtime/src/config/groups/client.rs:207, 215, 223`).
  - **High-performance mode:** start 16, 4–124, and 100 files ingested at once instead
    of 8 (`xet:xet_runtime/src/config/xet_config.rs:159-176`).
  - **Fixed:** `HF_XET_FIXED_UPLOAD_CONCURRENCY` sets the start, the minimum and the
    maximum together (`xet:.../config/aliases.rs:33-35`).
- **High-performance mode overrides your own settings.** `XetConfig::new` applies the
  environment's HF_XET_* values first, then `with_high_performance()` (`xet_config.rs:140-145`).
  So `HF_XET_HIGH_PERFORMANCE=1` overrides `HF_XET_DATA_MAX_CONCURRENT_FILE_INGESTION`
  and the `AC_*` values. A lower ingestion limit with HP's other values needs the knobs
  set one by one, without HP.
- **How it ramps** (`xet:xet_client/src/cas_client/adaptive_concurrency/controller.rs:440-600`):
  - **The two models:**
    - an RTT model: duration ≈ a + b × (bytes × concurrency), so the bandwidth is
      **shared** across streams (`rtt_prediction.rs`);
    - a success ratio: a transfer succeeds if it completes, in under 90 s, and below the
      95th percentile of its predicted RTT (`:462-488`).
  - **Up:** +1 stream when the success ratio is over 0.8, at least 20 MB sent and one
    transfer completed, at least 500 ms since the last change, and the predicted RTT of
    a reference transfer (the 95th percentile of sizes seen, 1–64 MB) at the new
    concurrency under 60 s (`:529-560`).
  - **Down:** −1 when a transfer fails, takes over 90 s (partial reports count), or the
    success ratio is under 0.5 (`:562-590`).
  - **What it learns:** with the uplink the limit, streams settle where a 64 MiB xorb
    takes ~60 s: N ≈ 60 s × U / 64 MiB, ~6 at 7 MB/s. With one connection the limit, a
    stream's RTT doesn't grow with N, the fitted slope flattens, and N climbs to the
    maximum.
  - **The edge:** from a start of s streams, the first step needs
    (64 MiB / r) × (s + 1)/s < 60 s. That is r > 1.7 MB/s at the default start of 2, and
    r > 1.2 MB/s at HP's 16.
- **Ramp time:** after the first completed xorb (~10–20 s at these rates), +1 per ≥0.5 s
  of progress reports, so 2 → 64 takes ~30 s. That is negligible against a release.
- **State across calls:** it persists for the process. Controllers are cached per Xet
  context and endpoint (`xet:.../adaptive_concurrency/cache.rs:1-6, 25-31`), and
  huggingface_hub keeps one `XetSession` a process. So a run's pushes, or a finalise
  run's commits, share one controller, and it doesn't restart at 2 each commit. Each
  process (the two sets' runs, each finalise run) has its own, and they compete on the
  link, each backing off on the other's RTT.
- **What bounds it:**
  - **Xorbs to send:** a commit of B bytes has ceil(B / 64 MiB) of them, as small files
    share xorbs (`file_upload_session.rs:391` takes the permit per xorb). This is the
    binding limit for our small tables, whatever the settings.
  - **Connections:** each stream is its own TCP connection (`xet:xet_client/src/common/http_client.rs:90, 123`,
    `http1_only`).
  - **Chunking threads:** blocking threads, beside ≤32 async threads
    (`xet:.../core/runtime/native.rs:33`). CDC, BLAKE3, sha256 and an lz4 probe are cheap
    against a link of ≤120 MB/s (not measured here).
  - **Memory per in-flight xorb:** a serialised xorb is held from its cut until its
    upload ends, and an ingesting file blocked on a permit holds one. Peak is about
    (streams + ingesting files) × 64 MiB, plus 8 MB read blocks:

    | | default | high performance |
    |---|---|---|
    | at a 7 MB/s uplink | ~6 streams + 8 files: ~0.9 GiB | starts at 16 streams, then falls to ~6; with finalise's 50-part claims batch, 50 files: ~4.5 GiB |
    | fast link, many big files | ≤64 + 8: ~4.5 GiB | ≤124 + 100: ~14 GiB |

    The push path has at most one big file a group (its claims, ~21 xorbs), so it stays
    ≤~1.5 GiB in either mode.
  - **HP's download side:** its download buffers are 16 GB + 2 GB a file, up to 64 GB,
    against 2 GB + 512 MB up to 8 GB by default (`xet_config.rs:170-174`;
    `groups/reconstruction.rs:31-47`). finalise downloads the group files, so **HP must
    not be set for finalise on a 16 GB laptop.**
- **Small files' own cost:** each file's first chunk is a global-dedup query, one round
  trip to the CAS (`xet:xet_data/src/deduplication/file_deduplication.rs:160-170`). The
  file's block waits for it (`:180`).
  - **Where it bites:** with 8 files ingested at once, a group's ~2,800 files take
    ~350 × q, whatever their bytes. HP's 100 files at once cut that 12-fold.
  - **The switch:** `HF_XET_DEDUPLICATION_GLOBAL_DEDUP_QUERY_ENABLED=0` would skip it.
    It isn't recommended: it also gives up cross-release deduplication.

## 4. What 7 MB/s can and can't be

- **What the measurement would read on each kind of link** (`model.py`'s
  `record_rate`: the 86 parts in two default commits):
  - **uplink 7 MB/s:** 7.0 MB/s. One stream may fill it, or several may be needed (at
    2 MB/s a stream, the controller ramps to 4–6);
  - **a stream 7 MB/s, uplink 30, 60 or 120:** 29.8, 59.2 or 116.9 MB/s. The controller
    would have ramped;
  - **a stream 1.5 MB/s:** 3.0 MB/s, stuck at 2 streams.
- **What that rules out:** only an aggregate limit near 7 MB/s reproduces the record.
  - **Unless the record is wrong:** "about 1.8 h" may have included other work.
  - **Or the controller misbehaved:** it's read from source, not observed.
  - **The sweep tests both.**
- **Not ruled out:** a cap on the Hub's side, per client, token or repo. The sweep's
  fixed-N points can't tell it from the uplink. A speed test of the machine's uplink to
  any other host can.

## 5. The throughput model, and the walls

`model.py` (every figure modelled):
- **The wall:** bench 17.9.0's `release_wall`, with the build branch's pushes and
  finalise's commits at rates of their own. With one rate it reproduces the bench's
  `release_wall_link_{7,30,60,120}_modelled` to the second, on all 6 after-arm runs of
  master's record (b110f78). Walls are medians over those runs.
- **A commit of B bytes in f files:** c0 + max(B / min(U, N r), ceil(f / F) q).
  - N = min(the controller's settled permits, ceil(B / 64 MiB));
  - U is the uplink, r one stream's rate, c0 a commit's fixed cost on the link, q a small
    file's dedup round trip, and F the files ingested at once.
- **The inventory:** a release's bytes by set and table are the record's, scaled as
  `finalise_link` scales them: 124.3 GB of group files and 128.9 GB published. The files
  are the main set's keys (`docs/dataset_cards_metadata.json`), taken for both sets: an
  upper bound of ~166k group files in ~900 pipeline commits (#167 estimated ~140–160k).
- **Both sets at once:** during the split, both sets' runs push at once, each filling the
  other's fixed costs. After it, under the turn lock, one at a time. The share pushed
  after the split comes from the bench's own timeline.

**If one stream is 7 MB/s and N streams add up to the uplink** (the bench model at
min(7N, U), no fixed costs):

| N streams | U = 30 MB/s | U = 60 MB/s | U = 120 MB/s |
|---|---|---|---|
| 1 | 7: 10h50m | 7: 10h50m | 7: 10h50m |
| 2 | 14: 8h01m | 14: 8h01m | 14: 8h01m |
| 4 | 28: 6h56m | 28: 6h56m | 28: 6h56m |
| 8 | 30: 6h51m | 56: 6h29m | 56: 6h29m |
| 16 | 30: 6h51m | 60: 6h28m | 112: 6h25m |
| 32 | 30: 6h51m | 60: 6h28m | 120: 6h25m |

The wall is flat above ~30 MB/s: the compute is the path there. So 4 streams would be
enough, if per-connection were the case. Finding 2 says the default would already have
used them.

**Our paths' effective rates and the wall** (c0 = 5 s, q = 0.2 s, an API round trip
0.3 s; selected rows, all in `model.py`'s output):

| link | the record would read | settings | push MB/s | finalise MB/s | wall | bench at U |
|---|---|---|---|---|---|---|
| uplink 7, one stream fills it | 7.0 | **today** | 6.2 | 6.7 | **11h17m** | 10h50m |
| | | `HF_XET_HIGH_PERFORMANCE=1` | 6.5 | 6.7 | 11h13m | |
| | | no turn lock (L2's early finalise lost) | 7.0 | 6.7 | 11h38m | |
| | | a group's tables at once | 6.9 | 6.7 | 11h03m | |
| | | finalise: 250 additions a commit | 6.2 | 6.9 | 11h09m | |
| | | finalise: 2 commits in flight | 6.2 | 7.0 | 11h06m | |
| | | HP + tables at once + 250 + 2 in flight (turn kept) | 6.9 | 7.0 | **10h52m** | |
| | | recommended: HP for the runs only and their tables at once; finalise 250, 2 in flight | 6.9 | 7.0 | **10h52m** | |
| uplink 7, a stream 2 MB/s | 7.0 | today | 5.0 | 6.6 | 12h06m | 10h50m |
| | | recommended | 6.9 | 7.0 | 10h52m | |
| a stream 7, uplink 30 | **29.8** | today | 23.5 | 24.9 | 7h04m | 6h51m |
| a stream 7, uplink 60 | **59.2** | today | 27.6 | 38.9 | 6h38m | 6h28m |
| | | recommended | 59.8 | 60.0 | 6h28m | |
| a stream 1.5, uplink 60 | **3.0** | today | 2.7 | 2.9 | 24h02m | 6h28m |
| | | `HF_XET_HIGH_PERFORMANCE=1` everywhere | 8.5 | 19.0 | 7h27m | |
| | | HP for the pipeline's runs only | 8.5 | 2.9 | 16h49m | |
| | | recommended | 32.5 | 6.0 | 10h58m | |

**The fixed costs' range** (uplink 7 MB/s, one stream fills it, today's settings):

| c0 (s) | q (s) | API (s) | push MB/s | finalise MB/s | wall | one set's pushes: link idle | finalise: link idle |
|---|---|---|---|---|---|---|---|
| 2 | 0.1 | 0.15 | 6.67 | 6.89 | 11h01m | 0h28m | 0h05m |
| 5 | 0.1 | 0.15 | 6.46 | 6.73 | 11h13m | 0h46m | 0h12m |
| 5 | 0.3 | 0.45 | 5.91 | 6.71 | 11h28m | 1h36m | 0h13m |
| 15 | 0.3 | 0.45 | 5.28 | 6.23 | 12h12m | 2h36m | 0h37m |

- **The bench is ~10–80 min optimistic at a 7 MB/s uplink.** It sends every byte at the
  link's rate, and our ~1,050 commits and ~166k small files aren't free. The 2026-10-04
  figure (≥4.9 MB/s through the push path, merges included) bounds how bad the overheads
  can be: about the c0 = 5 s, q = 0.2–0.3 s rows.
- **Where the turn lock caps parallelism:** after the split, one set pushes at a time,
  and its fixed costs and verifies idle the link.
  - **Removing it** lets both sets fill each other's gaps. But the main set's pushes then
    no longer end first, and L2's early finalise is lost: +21 min overall at 7 MB/s.
  - **Keep it.** The better fix is to cut the main set's own gaps: its tables at once.
- **Where the serial committers cap parallelism:** finalise's ~147 commits run one after
  another, each a fixed cost and a tail, and small tables get 1–4 streams.
  - **The cost:** 12 min at 7 MB/s (c0 = 5 s), but 10 min at 60 MB/s, where finalise's
    commits are the path.
  - **The fix:** larger batches, or two commits in flight on different repos.

## 6. What the user must run

`throughput_measure.py`, unexecuted. Each subcommand but `logs` uploads synthetic random
bytes to a private scratch dataset repo of the user's. Run them in order:

1. **Offline, first, on the release machine:** `python throughput_measure.py logs`.
   - **What it reads:** hf_xet 1.6.0 logs at INFO by default to
     `~/.cache/huggingface/xet/logs/` (`xet:xet_runtime/src/logging/constants.rs:3`;
     `xet_pkg/src/lib.rs:83`), kept two weeks and 250 MB (`groups/log.rs:51, 65`).
   - **What it shows:** the controller's lines every 10 s, its start, minimum and
     maximum, and how often it moved, for every upload in that window: the concurrency it
     settled at, its predicted bandwidth and the bytes sent.
   - **Reading it:**
     - concurrency ~4–8 with predicted bandwidth ~7 MB/s is the uplink;
     - concurrency at 64 with ~7 MB/s is a per-stream or Hub-side limit;
     - concurrency stuck at 2 is a slow stream.
   - **No Hub call at all.** release 20260928's own logs are three weeks old and most
     likely deleted.
2. `sweep --repo <you>/upload-scratch`:
   - **The points:** a fixed N = 1, 2, 4, 8, 16, 32, then adaptive, then
     `HF_XET_HIGH_PERFORMANCE=1`.
   - **The sizes:** 128 MiB and 512 MiB files (our parts are 500 MiB; our group claims
     files ~1.4 GB), 2 GB a point, a fresh process each.
   - **What it prints:** MB/s, MB sent, the 1 MiB first call's seconds (the session's
     start and a call's fixed cost), the controller's last and highest concurrency and
     its predicted bandwidth (from the point's own hf_xet log), and peak RSS.
   - **Time:** about 80 min at 7 MB/s.
   - **Reading it:**
     - flat from N = 1–2 is the uplink, or a Hub cap;
     - rising ×N up to a plateau is per-connection, with the plateau the uplink;
     - adaptive below the best fixed N would mean the controller doesn't find it.
3. `small --repo <you>/upload-scratch --hp`:
   - **The commits:** K = 1, 50 and 250 files of 64 KiB and 1 MiB, at the defaults and in
     HP mode.
   - **What it gives:** the intercept is c0 and the slope is our per-file cost, q / F.
     These are the two figures the model guesses.
   - **What it commits:** to branch `measure` of the scratch repo.
4. `parallel --repo <you>/upload-scratch`:
   - **The calls:** k = 1, 2 and 4 calls of 1 GiB at once, as threads of one process and
     as k processes;
   - **What it shows:** whether two sets' runs, or commits in flight, get more than one
     call does.

Then put the figures in `model.py`'s `Link` (U, r, c0, q, api) and run it again.

## 7. Changes we could make now, and their evidence (not implemented)

**Hub-API-neutral** (only environment variables, or the concurrency and size of calls we
already make):

| change | where | modelled effect (7 MB/s uplink; 60 MB/s) | risks | evidence the user needs first |
|---|---|---|---|---|
| **HP mode for the pipeline's runs only:** `HF_XET_HIGH_PERFORMANCE=${HF_XET_HIGH_PERFORMANCE:-1}` on the two `process-wikidata` lines of `overlap-release`, **not** for finalise | Justfile | −4 min; ~0. It mostly cuts small files' dedup round trips (8 → 100 at once). At a 1.5 MB/s stream (not the record's case), 24h02m → 16h49m | **memory:** the push path is ≤~1.5 GiB of xorbs either way (one big file a group). **Throttling:** up to 16 connections at the start, back to ~6 at 7 MB/s, 429s retried 5× up to 6 min, then the run stops (resumable) | `logs` or `sweep`: does the controller sit at 2? `small`: q, at 8 against 100 |
| **finalise: 250 additions a commit** (`COMPACT_COMMIT_MAX_ADDS` 50 → 250, ≤ upload_folder's own first batch) | `config.py:253` | −8 min; −8 min (6h38m → 6h30m) | bigger commits: a failure re-sends more, but Xet deduplicates a rerun (#177 §3). Memory unchanged: ingestion stays 8 files at once. The Hub's limit on a commit's files isn't in the client source; the pipeline already commits 250–1,000 | `small`: c0 against K |
| **finalise: two tables committed at once** (each its own repo: no parent-commit races) | `sort_by_id.Committer` | −11 min; −10 min (6h38m → 6h28m) | the commits' order across tables changes, and the ledger's order with it (each table's stages stay in order); a second Xet upload in the process shares the controller | `parallel`: k = 2 in one process |
| **a group's tables pushed at once** (a thread a table, then verify) | `push/core.py:push_group` | −14 min; ~0 | the ingestion slots are shared (modelled). A failure in one table fails the group, which resumes as today (`pushed` is recorded once all are up). ≤~1.5 GiB of xorbs | `small` (c0, q) and `parallel` |
| relax the turn lock | `push/turn.py` | **+21 min** (L2's early finalise lost) | | **don't** |
| `HF_XET_HIGH_PERFORMANCE=1` for the whole release, or for finalise | | ~0 at 7 MB/s | **finalise's peak:** a 50-part claims batch, 100 files at once: ~4.5 GiB of xorbs, and HP's download buffers up to 64 GB while finalise downloads the group files, on a 16 GB laptop | **don't**, unless the sweep shows a stream under ~1.7 MB/s; then per-knob (`HF_XET_CLIENT_AC_INITIAL_UPLOAD_CONCURRENCY=16`, ingestion left at 8), not HP |

- **Together** (`model.py`'s "recommended" row: HP for the runs only, a group's tables at
  once, 250 additions a commit, two tables in flight):
  - **at a 7 MB/s uplink:** 11h17m → 10h52m, −25 min;
  - **at 60 MB/s:** 6h38m → 6h28m, −10 min.
- **New Hub API use** (needs the user's authorisation):
  - L1b's `preupload_lfs_files`, decoupling upload from commit (#177);
  - `hf_xet`'s streaming upload.

  Neither changes the uplink.
- **Why none is built here:** each needs the user's `small` and `parallel` figures to
  size it, and each touches `src/` or the recipes. Those changes would be benched on CI
  as usual.

## Uncertainties and corrections

- **The record:** "about 1.8 h" is a rounded recollection of one table's commit. Its
  start and end aren't recorded. The ~10% rounding alone makes 6.3–7.7 MB/s.
- **The controller:** read from source, not observed. The sweep's adaptive points (and
  `logs`) show where it settles.
- **The model's c0, q and API round trip are guesses.** The file counts are an upper bound
  (the main set's keys for both sets). The push path's fixed costs are modelled; its
  merges stay in the bench's CPU model.
- **The brief's "~128 MiB parts":** our parts are 500 MiB (`COMPACT_FILE_BYTES`); 128 MiB
  is the row group (`COMPACT_ROW_GROUP_BYTES`). The sweep runs both sizes.
- **Deduplication:** content-defined chunking could cut a second release's bytes
  (#167). This entry doesn't model it: it changes the bytes, not the rate.
