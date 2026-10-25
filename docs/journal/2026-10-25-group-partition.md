# 2026-10-25: Bypassing the group files: what they are for, the designs, and a prototype

A prototype, authorised by the user as "bypassing group files", on three terms:
- restart/resume and the upload guarantees are kept;
- the Hub integration is not casually rewritten;
- consequential compatibility or recovery trade-offs are reported before they are committed to.

Two pull requests come out of it:
- **lmmx/wikidata-pq#140** (design D below): the tables split by key are partitioned once a group, at its close, not per chunk and then glued.
- **lmmx/wikidata-pq#143** (E1 of `2026-10-25-architecture-trace.md`): the verified group files are kept for finalise, rather than deleted and downloaded again.

Neither is merged. Each lists the trade-offs the user must accept.

## Current State

### What the group files are for

A release's chunks are processed by worker processes that finish out of order. Each chunk is
partitioned into a file per (table, key): about 11–12M files at a release. Once a contiguous
run of chunks reaches the threshold (about 1/30 of the set's projected partition bytes), the
run is closed as a group:
- `merge_group` glues each key's partition files into `staging/{table}/{key}/chunks-a-b.parquet`;
- `push_group` uploads them to `build-{release}` with `upload_folder`;
- `verify_group` compares each file's size and sha256 with the Hub's;
- the clean-up deletes them.

**What each guarantee needs from them:**

- **The Hub's file guidance** (<100k files a repo, <10k a folder): one file per key per group, about 30 a key. That is the only reason the rows are concatenated before upload.
- **The upload contract.** `push_group` uploads, and `verify_group` checks every staged file against the Hub's sha256 before the chunks are `COMPLETE` and their local data deleted. hub.py itself only makes the build branch (`ensure_build_branch`) and promotes it. Finalise expects these things on the build branch:
  - `{key}/chunks-{first}-{last}.parquet` (`compact.FILE_RE`), downloaded by `compact.download`;
  - its sort's `commit_table` deletes exactly a key's sources (compaction's manifest) and adds the part files, refusing any unknown file;
  - promotion copies whatever the branch then has.

  Nothing depends on a group file's internal layout, its row groups, except through sizes:
  - compaction's runs (`COMPACT_FILE_BYTES`);
  - the sort's in-memory or bucket choice;
  - its part count;
  - `_row_group_rows`' sample.
- **Restart and resume:**
  - *A chunk:* its state file; partitioned chunks wait in `open_chunks`, their partition files on disk.
  - *A group:* `groups.jsonl` (`closed`, `merged`, `pushed`, `verified`, `done`), each stage safe to repeat. The partition files are deleted only once their key's group file is checked against the sidecars, and the staging only once the Hub's copy is verified.
  - *Across machines:* the build branch holds every verified group, and finalise downloads from it. The state dir would still have to be carried over: the chunk states and the ledgers are local.

### The designs, at a release

Notation:
- **G** ≈ 115–135 GB: the group files, all tables, both sets.
- **C** ≈ 100 GB: the unsplit tables' share of G (claims 96.5 GB, plus entities).
- **S** = G − C ≈ 15–35 GB: the tables split by key (labels, descriptions, aliases, links, claims_labels).

Rates are bench 11.0.0's (#132's comment): the pipeline's "rest" is 1,104 ETA-s a sample
CPU-s; `merge_group` is 1.0 CPU-s on the sample (18 min) and `partition_parquet` 3.5 CPU-s (64 min).

| design | bytes written / read locally, beyond master | modelled saving at a release | Hub |
|---|---|---|---|
| master | partition files: S + C written, read by the merge; group files: G written, read by upload, verify and the download | — | ~30 files a key |
| **(a)** finalise reads partitions; partitions uploaded | − G written and read (no merge) | +18 min of merge; then 11–12M uploads, 11–12M downloads, and 5,750 sort commits of 2,000 deletes | **~11.5M files**, ~100× the per-repo guidance; big languages ~23k files a folder (>10k) |
| **(b)** chunk workers append to per-group, per-key files | − S + C partition write and read (no partition files, no glue) | 30–45 min: the merge (18) and the file system's share of partition (a fifth of 64, #106), less a lock and an append a (chunk, key) | unchanged; group files can stay byte-identical with an appending writer |
| **(c)** group concat fused into compaction's glue | − compaction's copy of small keys only (≤ S); claims are linked already | ≤ ~20 min (`_write_run` + `_check_glued`), which #135 already takes for in-memory keys; and it needs (a) or no upload at all | as (a), or nothing until the sort |
| **(d) group partition** (#140) | − S partition write and read; − ~11.5M file creates, opens and unlinks; the split tables' rows encoded once, at the close | up to ~70–80 min (the merge, and partition's per-file cost, ~1 ms a file, ~11.5M → ~160k files); the bench sees only its sample's share (below) | unchanged names and rows; split tables' row groups re-encoded |
| **E1** keep verified files (#143) | − G downloaded and written at finalise | 20–45 min of wall at 50–100 MB/s, unmodelled (the bench models no transfer) | unchanged |

**Restart, by where a crash lands:**

- **(a):**
  - *Pipeline:* as master's, with each upload a restart point.
  - *Machine lost:* the Hub has everything, but finalise must fetch 11.5M files.
- **(b):**
  - *Mid-append:* a torn row group sits in a file shared by up to 6 workers, so it cannot be truncated once another has appended after it. That needs a lock held across append and ledger, or redoing the whole open group.
  - *Redoing the group* means its chunks' inputs must be kept until the close. Their processed files are deleted at partition today, so that is (d)'s disk cost, plus (b)'s concurrency.
  - *Needs:* pq-glue's appending writer (not written), and per-file locks on ~2,700 files a set.
- **(c):**
  - *Pipeline crash:* fine.
  - *Machine or disk lost:* everything since the split is lost, since nothing is uploaded until the sort. That means the split and the pipeline again, ~6.5 h.
  - *Disk:* the whole set's partitions (and inodes: ~6M files a set) are held until finalise.
- **(d):** tested at 14 kill points (below). The inputs, the processed files, are kept until a table's group files are checked and its marker written.
- **E1:**
  - *Crash after deletion, or on another machine:* finalise falls back to the download.
  - *Crash mid-move:* the rerun's clean-up moves the rest.

**Memory and disk on a 16 GB laptop:**

- **master:**
  - *Memory:* the merge is a byte copy, megabytes.
  - *Disk peak:* the open group's partitions plus its staging, ≤ 2 × `GROUP_MAX_GB`.
- **(a) / (c):** disk holds the set's partitions (~60–70 GB main) until finalise.
- **(b):**
  - *Disk:* ≤ 1 × group, with no staging copy.
  - *Memory:* each worker holds a key's slice before appending.
- **(d):**
  - *Disk:* as master's. The processed files wait where partitions did, then the group files; a table of several batches briefly holds its batch files too.
  - *Memory:* the parent's sink holds about its batch's rows: 280 MB for 4M and 840 MB for 16M synthetic label rows (600 keys), measured. `GROUP_SINK_ROWS` (10M) bounds it at ~0.5–0.9 GB.
  - *Determinism:* the sink's output was byte-identical across three runs.
- **E1:** + the set's G (~60–70 GB main) from its pipeline to its finalise, and the other set's while one is finalised.

**Chosen:** (d), as the design that keeps every guarantee as it stands:
- the Hub layout;
- the upload contract;
- the restart semantics (inputs kept until outputs checked).

It needs nothing new upstream, and it removes both the merge's copy and the partition files'
per-file cost.

E1 is independent: it removes a transfer, not a pass, and is a separate PR.

### The prototype (#140)

- **`main.partition_chunk`:**
  - *Each table split by key* (`push.core.GROUP_PARTITIONED`) gets only its sidecar (`partitioning.write_stats_sidecar`, from `key_stats`), and its processed file is kept.
    - The sidecar rows are `partition_parquet`'s: keys, rows, id bounds and `path`, in Path order.
    - Its `file_size` shares out the processed file's bytes by rows, so the group threshold still has a size a chunk.
  - *Claims and entities* are placed as before.
- **`merge_group` → `_partition_table`, per split table:**
  1. The group's processed files are read in chunk order by one Polars sink (`PartitionBy` by key, the file named by the group), in batches of up to `GROUP_SINK_ROWS` rows.
  2. A key's batch files are glued (`pq_glue.concat_many`); claims_labels' are deduplicated by Polars.
  3. Rows are checked against the sidecars, keys must match exactly, and the files move into staging.
  4. Then the marker `staging/.merged/{group}/{table}` is written, and only then are the processed files deleted.

  Claims and entities are glued in a thread meanwhile.
- **Also fixed** (found by the restart test, present on master): `place_unsplit` left `claims/all/chunk_N.tmp` behind when a chunk was partitioned again over its own hard link, because a rename between two links to one inode does nothing.

### Restart tests

`scripts/test_group_restart.py` runs main.run's loop in a process of its own: resume an
unfinished group, then `pool.process_chunks` with `close_group` in the background.
- **The data:** synthetic chunks of all 7 tables.
- **The Hub:** a stand-in HfApi with the build branch on disk.
- **The groups:** 0-2 and 3-5, forced, as chunks 3-5 wait for the first close.
- **The kills:** `os._exit` at 14 points, then the run repeated until it completes. The points:
  - a chunk's partition, between tables, and before its state;
  - the merge: after a sink, before a marker, mid-deletion, in the glue thread;
  - after `merged`, `pushed`, `verified`, `done`;
  - mid-upload, mid-verify, mid-clean-up, and the last group's upload.

In every case:
- the same 100 files reach the stand-in build branch, byte for byte;
- finalise's local stages publish the same 41 sorted files, byte for byte;
- every chunk is `COMPLETE` and nothing is left on disk.

On master the same harness passes its merge-path points but leaves the `place_unsplit` link (above).

`scripts/test_merge.py` covers:
- rows, order and schema against per-chunk partitioning;
- deduplication;
- batches;
- an interrupted sink resumed byte-identical;
- an interruption after the marker;
- missing inputs and mismatched keys.

### Published files identical

The comparison is per key, on the published sorted part files: rows, order and Arrow schema
(parity's `final` fingerprint), and sha256 where it can be had.
- **Locally:** three bench chunks (r50, s30, r30) through process, partition, merge, compaction and sort. All 1,154 published part files are byte-identical to master's.
- **The group files:** rows and order identical for all 1,154 keys, but other bytes for the split tables (one row group a key, not one a chunk). Claims and entities are byte-identical.
- **With batches of one chunk:** the published files are still byte-identical.
- **The synthetic restart harness:** published files byte-identical to master's.
- **At a release**, a key near a size threshold could be cut into other part files: same rows and order, other bytes. Decoupling the sort's thresholds from encoded sizes, another agent's work, would take this away; the comparison to use then is parity's.

### Bench (12.0.0, quick tier, 3 rounds paired, #140 head `145299f` against master `fa959e4`)

**Measured:**

| | before | after | change |
|---|---|---|---|
| measured end to end: CPU (split + pipeline + finalise) | 2m07s ± 7.7s | 2m08s ± 9.3s | n.s. |
| measured end to end: wall | 47.37s ± 3.10s | 48.00s ± 3.50s | n.s. |
| CPU, all processes (the pipeline) | 8.8s ± 0.3s | 8.3s ± 0.3s | **−0.4s (−4.6% ±1.0%)** |
| CPU, partition | 3.1s | 0.6s | **÷4.9** |
| CPU, group merges | 1.2s | 3.3s | **2.7×** |
| wall (the pipeline) | 3.38s | 3.36s | n.s. |
| peak RSS, all processes | 2.187 GB | 2.157 GB | n.s. |
| disk, peak rise | 652 MB | 641 MB | **−1.6% ±0.4%** |
| finalise: compact / sort | 1.13s / 33.73s | 0.93s / 34.53s | n.s. |

**Parity:** passed in every round.
- All 2,324 published keys have the same rows, order and schema.
- The group files have the same rows; none is reordered.
- The sidecars are the same but for `file_size`.

**Modelled, the bench's way** (entities): the ETA is 7h59m23s → 7h56m26s, and the pipeline's 2h32m33s → 2h28m10s. Both are within noise; before's pipeline ETA has ±14m.

**Why the bench sees so little:** the bench scales the sample's per-file costs by entities.
- **Before:** the sample's 3,431 partition files stand for ~12.6M, close to a release's ~11.5M.
- **After:** its 2,324 group files (one group of all 4 chunks) stand for 8.5M, where a release has ~2,700 keys × ~30 groups × 2 sets ≈ 160k.

**A per-file model of a release,** from the sample's own figures (`release CPU × 0.40`, the runs' wall/CPU):

- **Measured locally, on Polars' sink:**
  - 0.27 µs of CPU a row;
  - 0.7–1.2 ms a file;
  - 4M rows to 1 key took 1.06 CPU-s; to 2,000 keys, 3.41.
- **Splitting the sample's figures:** the sample's split-table rows, about 3M, are ~0.9 s of either arm's encode. That leaves:
  - before: ~0.5 ms a partition file and ~0.3 ms a merged file;
  - after: ~1 ms a group file.
- **Before:** 11.5M files × ~0.8 ms ≈ 2.6 h of CPU, ~61 min of ETA.
- **After:** ~160k files × ~1 ms ≈ 3 min of CPU, ~1 min of ETA, plus ~5 min for the workers' `key_stats`.
- **So a release saves roughly 40–70 min of the pipeline's ETA.** The range is for the split between per-row and per-file costs.

This is a model, not a bench figure. The bench would show it only with a tier of many chunks
a group, or a term that scales the per-file part by a release's file counts.

## Missing

- **E2** (the split writes group files itself) would supersede (d). So would fusion A, if it makes the split write these tables' per-chunk partitions. Whichever lands second must be reconciled with the first.
- **A modelled transfer term in the bench**, which E1 needs to show on CI.
