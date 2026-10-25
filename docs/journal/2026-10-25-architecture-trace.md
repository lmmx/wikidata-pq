# 2026-10-25: Every intermediate a release writes and re-reads, and the pipeline without them

An architecture trace, not an experiment. It follows each substantial read → convert →
write → re-read from the dump to the published files. For each intermediate it asks what
information was missing earlier that forced the work to happen later. Then it draws the
pipeline as it would be designed today and ranks what is left to eliminate. The script
is in `2026-10-25-architecture-trace/`.

## Current State

- Docs only: nothing in `src/`, `reshape/` or `bench/` changed.
- **Master** is 44f4e0e (#132). The figures are bench 11.0.0's, from #132's run (the
  "after" arm): a modelled release of **9h35m**, made of the split 2h35m, the pipeline
  4h03m and finalise 2h58m.
- **In flight, and taken as done in the ideal picture:**
  - the claims built as Arrow in the split, design B (#133, #134);
  - labels, descriptions, aliases and links built in the split (fusion A);
  - in-memory keys sorted straight from their linked group files (#135);
  - bucket sorts writing straight into their part files.
- **The findings:**
  1. **Two of the four intermediates left once the work in flight lands exist only
     because of history:**
     - the per-(chunk, key) partition files, about 11.5M at a release;
     - finalise's download of the group files this same machine has just uploaded and
       verified.
  2. **The pending threshold decision is the keystone.** Every remaining layout change
     (one file per table, group files written in the split, claims bucketed at write
     time) is blocked by it. Its value is the sum of what it unblocks: about **45–70 min**
     modelled, not the 28 min credited to one file per table alone.
  3. **Hub transfers are now the largest unmodelled cost.** The bench leaves out every
     transfer. At a release, about 115–135 GB crosses the network three times: the
     build branch's upload, finalise's download and the publishing commit. As the
     pipeline's compute moves into the split, the group uploads no longer hide under
     it. That makes them the release's likely critical path.
- **Ranked eliminations** (below):
  1. the split writes each key's group file itself: 25–45 min, blocked by the decision;
  2. finalise reads the verified group files it kept: 20–45 min of wall, unblocked, not
     modelled;
  3. entities built in the split, so the chunk file and `process()` go: 15–25 min,
     after fusion A.

## Sources

- **Bench 11.0.0, #132's comment ("The ETA by stage"):**
  - **The split:** 28.4 CPU-s, at 327 ETA-s a CPU-s (by bz2 bytes, 4 cores).
  - **The pipeline:**
    - the claims stages: 12.9 CPU-s, at 353 (main) and 590 (scholarly);
    - the rest: 7.5 CPU-s, at 1,104 (entities × wall/CPU 0.30);
    - the workers' starts: 724 at a release, at 54.
  - **Finalise:**
    - the part per byte is half the 3× run less the first, at 330 (claims, 1,321×/4) and
      917 (other tables, 3,667×/4);
    - plus a fixed part per key, and the claims workers' starts.
  - **Hours of a stage** = its CPU on the sample × its rate. Where a stage mixes the
    claims and other tables, the split by table is from
    `docs/journal/2026-10-25-release-profile-bench10.md` (#131).
- **Sizes at a release** (20260928 unless said):
  - **The dump:** 103.5 GB of bz2 (156.6 GB of gz).
  - **The routed chunk files:**
    - the scholarly set: 49.5 GB, 10,853 chunks;
    - the main set: 44.7 GB, 12,182 chunks;
    - both at zstd 9. Today's zstd 3 writes about 1.1× that.
    (`2026-10-03-release-20260928-run.md`)
  - **The claims key:** 96.5 GB. The main set's 44.7 GB is measured (30 group files,
    3,769 row groups, 667 buckets, 86 part files); the scholarly set's is modelled by
    bench 10.0.0.
  - **The other tables:**
    - the main set's are at least ~15 GB (the 20260507 build);
    - the scholarly set's are not on record;
    - the bench's entity scaling (27.6 MB of the sample's group files × 3,667 ≈ 100 GB)
      overstates them about 3× (#131).
  - **So the group files:** G ≈ 115–135 GB across both sets, modelled.
  - **Partition files:** about 11–12M at a release (#131's count).
  - **Sorted buckets:** 42 GB for the main set (`2026-10-05-sort-speed.md`).
- **Code read:**
  - the split: `dump.py`, `scripts/rebuild_entities.py`;
  - the pipeline: `main.py`, `process.py`, `partitioning/`, `pool.py`, `push/`;
  - finalise: `compact.py`, `sort_by_id.py`, `claims_labels.py`;
  - #134's and #135's bodies.

## The trace: every intermediate representation

"Real" means that information the earlier stage did not have forces the intermediate.
"Historical" means nothing forces it now.

| # | intermediate | format, size at a release | written by | read by | why it exists |
|--:|---|---|---|---|---|
| 0 | the dump on disk | bz2, 103.5 GB (gz 156.6 GB) | `download-dump` | the split's decoder, following the download | **Real:** the download's restart point (resumable, md5-checked). The decoded JSON (~1 TB) is never materialised: it is held in memory, at most `MAX_PENDING_BYTES` (4 GB). |
| 1 | the routed chunk files, `chunk_N.parquet` per set | Parquet, zstd 3: `id` plus five JSON-text columns and `entity` as JSON text; ~94 GB at zstd 9, ~104 GB at zstd 3 | the split (`write_chunk` → `route_parts`), then `route` renames them to each set's numbers | `process()`: Polars reads `id`, `sitelinks` and `entity`; genson reads `labels`, `descriptions`, `aliases` and `claims` from the file itself | **Real:** the restart point between the split and the pipeline (the split's manifest). Also each set's chunk numbering and the group rule's total source bytes are known only once the split ends. **Historical:** the JSON-text shape is philippesaade/wikidata's, which `process()` was written for. The split parses every line with sonic-rs, writes it back as text, and genson lexes the text again: a serialisation boundary with nothing on the far side that needs text. Design B removes the claims column (most of the bytes), and fusion A the maps. `entity` (and the rows' ids) would remain. |
| 2 | the processed tables, `results/{table}/chunk_N.parquet` | Parquet: genson's writes for the maps and the claims, Polars' sink for links and entities. Transient: a few chunks' worth (~5 MB a chunk) at once. | `process()` | `partition_chunk`: `prepare_for_partition` + `partition_parquet` read and re-encode every table but the claims, which are hard linked (`place_unsplit`) | **Historical:** steps 2 (process) and 3 (partition) were separate. Their only use now is restart within a chunk (each table's file existing means that table is done), and a chunk is ~11 s of work. Fusion A removes it for the maps and links. |
| 3 | the partition files, `results/{table}/{key}/chunk_N.parquet`, plus an audit sidecar per (chunk, table) | Parquet from Polars' `PartitionBy` sink, one file per (chunk, key): ~11–12M files at a release, ~1 ms of sink CPU each; at most one open group's worth live (≤ 25 GB) | `partition_parquet` | `merge_group` (`pq_glue.concat_many`), then deleted | **Real:** chunks finish out of order in separate worker processes. A group closes only once a contiguous run of chunks reaches the threshold, and the threshold is projected from the ratio seen so far (`group_threshold_bytes`), so a chunk's rows must wait on disk until its group is known. **Historical:** one file per key, not one per table (#106, pending). Once the split writes every table, chunks come out in dump order, so the wait has no cause left except the group rule. |
| 4 | the group files, `staging/{table}/{key}/chunks-a-b.parquet` | Parquet, the partition files' row groups copied byte for byte; G ≈ 115–135 GB in all, ~30 groups a set | `merge_group` | `push_group` (upload), `verify_group` (size and sha256 against the Hub's), then **deleted** (`close_group`, `CLEAN_UP_LOCAL`) | **Real:** the Hub build branch is the cross-machine restart point, and a key's ~30 group files keep each repo under the Hub's file guidance. **Historical:** deleting them once verified, only for finalise on the same machine to fetch them back (row 5). |
| 5 | `compact/src/{table}` | the group files again, G ≈ 115–135 GB through the network | `compact.download` (`snapshot_download` of the build branch) | `rewrite_table` | **Real only across machines.** The `release` recipe runs both sets' pipelines and finalise on one machine, so here it re-downloads files that were verified byte for byte against the Hub's sha256 before being deleted. claims_labels already takes the local route: it is built into `compact/src` and never downloaded. |
| 6 | the compacted files, `compact/out` → `hub/{table}` | hard links (claims; after #135 every in-memory key too), or a deduplicating rewrite (claims_labels) | `rewrite_table` (`_link_run`: a link, then sha256 in the parent, one file at a time) | the sort, after `check_compacted_copy` hashes them again | **Historical:** compaction uploaded its files until 2026-10-05. Since #135 nothing is glued that the sort will read whole. What is left is a link, a hash, a manifest and claims_labels' rewrite. `check_compacted_copy` is kept by decision. |
| 7 | the claims' bucket files, `compact/sort/buckets/claims/all` | `bounds.json`; spilled fragments (lz4 Arrow IPC, the rows a bucket does not read directly: 10% on the bench's 3× run, 20–30% modelled at a release); sorted buckets in the Hub's settings (42 GB for the main set) | `bucket_key` (`_bucket_source`), `_sort_bucket` | `_read_bucket`, then the packing glue (`pack_key`; in flight to go) | **Real:** a global stable sort by string id needs every chunk, and the claims (96.5 GB) exceed memory, so the sort is external. Fragments exist because a row group spans many buckets: chunk order runs through the string-id space several times (numerically adjacent ids such as Q2 and Q3 sort far apart: Q2, Q20, Q200, …, Q3). **Historical:** the sorted buckets as files of their own, glued into part files afterwards (in flight). |
| 8 | the sorted part files, `compact/sort/out` | Parquet with the Hub's settings, ~G | `sort_in_memory` / the bucket sorts | `_check_written` (an independent decode, kept), `commit_table` (the publishing upload); they replace `hub/{table}` | **The contract.** |
| 9 | claims_labels: `refs/*.parquet` per claims file, `refs.parquet`, one group file per language | Parquet: 4.7% of the claims bytes before deduplication across files (measured below); the language files are unrecorded | `collect_refs` (from the **sorted** claims), `write_groups` (a join with both sets' labels, sorted by `ref`, `field`) | compaction's deduplicating rewrite (`_write_deduplicated_key`: Polars `unique`, a zstd write, a fingerprint check), then the sort (sort, write, check) | **Real:** the label join needs both sets' labels complete. **Historical:** the refs are re-read from every claims file in finalise (9 leaves decoded, 6.8 GiB a 529 MB file, `FINALISE_LARGE_WORKERS` at once), though the claims are built in one place where they could be taken. The build's rows are written three times (build, dedup rewrite, sort) when they are already distinct and sorted by `ref`. |

### What crosses a serialisation boundary, and how often (after the work in flight)

| | parsed or decoded | encoded |
|---|---|---|
| a claims row | the dump line (sonic-rs, once); the build-branch file (`_read_bucket`, ~1.15× a row); `_check_written` (kept) | the build-branch file (`write_table`, in the split); the published part (`_write_file_parallel`) |
| an other-table row | the dump line (once); the group files (`_read_key`); `_check_written` | the partition file (row groups copied into the group file, not re-encoded); the published part |

Both are at the floor the contract sets:
- an encode for the build branch (the restart point);
- a decode for the global sort;
- an encode for publishing;
- an independent decode to check it.

What is still above the floor is **files and copies**, not encodes:
- 11.5M partition files and the glue of them;
- G downloaded again;
- claims_labels' extra writes;
- the claims' fragments and read amplification.

## The pipeline as it would be designed today

```
 download (3 connections) ──► dump.bz2 | dump.gz on disk         [restart: resumable, md5]
        │  followed as it is written (already)
        ▼
 ■ decode: lbzip2 (1h48m, floor) | igzip (14m)          unavoidable: the input's format
        │  JSON lines, in memory only (≤ 4 GB pending)
        ▼
 ■ reshape: one sonic-rs parse a line                   unavoidable: one parse
        ├─ route: is_scholarly → main | scholarly set
        ├─ build Arrow for every table:
        │     claims (B, in flight) · labels, descriptions, aliases, links (A, in flight)
        │     · entities (E3)
        ├─ claims_labels' refs from the built claims → a small sidecar per part (E4)
        └─ ■ encode each (set, table, key) slice as its row groups   unavoidable: the build
        ▼                                                     branch needs an encoded file
 group files, appended in dump order, one writer per (set, table, key)        (E2)
   today: process() → processed files → partition_parquet → 11.5M partition files
          → merge_group's glue
        │  [restart: the split's manifest up to the last closed group]
        ▼
 ■ upload and verify each closed group → the Hub build branch     the contract
        │  (cross-machine restart point)
        │  kept locally as compaction's source (E1); today deleted, then downloaded again
        ▼
 finalise, per key, from the group files (compaction is links only: #135, E1)
   ■ the global stable sort by string id                unavoidable: needs every chunk
     in-memory keys: read the ~30 group files ─ sort ─ encode parts ─ _check_written
     claims (96.5 GB > RAM, external): plan buckets (or buckets cut at write time, E5)
        ─ per bucket: read ─ sort ─ encode straight into its part file (in flight)
        ─ _check_written
        ▼
 claims_labels: union of the refs sidecars ⋈ both sets' labels → written as its
   published parts (E6); today: refs re-read from the sorted claims, rows written
   three times
        ▼
 ■ commit the sorted parts → the Hub (publish)                    the contract
```

`■` marks what is unavoidable under the constraints:
- the decode;
- one parse;
- one Arrow build;
- one encode for the build branch;
- the build branch's upload;
- one external sort (a decode, an encode and the kept independent decode);
- the publishing upload.

Everything else in today's pipeline is a file or a copy between those steps.

**What it would come to, roughly (modelled; a reading, not a model):**

| | modelled release | basis |
|---|--:|---|
| master (bench 11.0.0) | 9h35m | #132 |
| with the work in flight | ~7h55m | B about −1h (its own model); #135 −12–20m; the packing glue −2–7m; fusion A −15–25m (`normalise_in_schema` of the maps, `atomic_sink_parquet`, `normalise_sitelinks` and the processed files' re-read, less the split's builders) |
| + E2, E3, E5 | ~6h40m–7h00m | the ranking below |
| + the gz dump | ~4h50m–5h10m | the split 2h35m → 46m (#132's gz row), chosen per release |

Not in any of these:
- E1 and E4 (20–45 min of wall, and ~7–12 min);
- the three Hub transfers of G.

## Ranked eliminations

By expected saving at a release. "Modelled" uses bench 11.0.0's rates. "Unmodelled" means
the bench does not run that part.

### E2. The split writes each key's group file itself: no partition files, no `merge_group`

- **Mechanism:**
  - Once the split builds every table (B, A, E3), its chunks come out in dump order. The
    pool's out-of-order finishes are what forced the partition files (trace row 3).
  - The split keeps one open writer per (set, table, key): about 2 × 2,700 at a
    release's key counts. The fd limit here is 20,000.
  - Each chunk's slice of a key is appended as its row groups, as the partition file's
    row groups are glued today. A group is closed when it reaches the threshold.
  - The per-(chunk, table) audit sidecars become per-(group, table) rows.
  - `merge_group` is reduced to closing files.
- **Expected saving: 25–45 min, 4–8%, modelled.**
  - **One file per table** (#106) measured partition CPU −38% and merge −17–20%: about
    28 min under bench 10.0.0, the files' share of `partition_parquet` (1h03m).
  - **E2 goes further:** no per-chunk file at all, and no glue. The rest of
    `merge_group` (18 min, mostly per-file opens and copies of ~11.5M files) goes too.
  - **The lower end allows for fusion A** having already cut the per-file cost, by
    writing partitions from Rust rather than through Polars' sink (~1 ms a file).
- **Blocked by:**
  1. **The pending threshold decision.**
     - **Why it changes the files:** the group rule must change. Today it is projected
       from total source bytes, a quantity of the chunk files, and from the finish
       order. A new rule changes group membership. Through #135's planned sizes (one
       footer per run) and `_row_group_rows`' sample (the largest group file's first row
       group), that changes published row groups and, rarely, part counts.
     - **Already true today:** grouping depends on the order chunks finish, so the
       published bytes are already a function of timing. #125's order-independent
       `_row_group_rows` belongs here.
  2. **A decision for the user beside the deferred overlap.**
     - Groups would close during the split, and their uploads would have to run during
       it.
     - Otherwise ~115–135 GB of uploads become a tail after the split, at 7 MB/s, the
       one upload rate on record (`2026-10-05-sort-speed.md`): about 5 h.
  3. **pq-glue:** an appending writer, so a group file's bytes stay the glue's (pq-glue
     #12's `concat_size` is the start of it).
- **Restart:** an open group is redone from its first chunk. The split already re-decodes
  from the dump's start on a rerun, so a restart costs at most a group's reshape and
  build (1/30 of a set).
- **Feasibility:** read only.
  - The staging layout `{table}/{key}/chunks-a-b.parquet` is the group file's and
    compaction's.
  - Key counts at the bench (564, 442, 489, 827, 1, 1) and a release's are within the fd
    limit.
- **Files:**
  - `reshape/src/` (fusion A's partition writers, made appending);
  - `src/wikidata/dump.py` (`split`, `write_chunk`);
  - `src/wikidata/push/groups.py` (the group rule);
  - `src/wikidata/push/core.py` (`merge_group`, `_glue_table`);
  - pq-glue (an appending writer);
  - `scripts/test_merge.py`, `scripts/test_split_follow.py`.
  - **Overlaps fusion A:** the same owner, after it merges.

### E1. Finalise reads the group files it verified, not the build branch

- **Mechanism:**
  - `close_group` moves each verified `staging/{table}/{key}/{group}.parquet` to
    `compact/src/{table}/{key}/` (a rename: the same layout `_local_keys` globs), rather
    than deleting it.
  - It records the size and sha256 that `verify_group` just compared with the Hub's LFS
    sha256.
  - `compact.download` is skipped for a table whose every group is kept and recorded.
    It is otherwise run as now (a restart on another machine).
  - `_link_run`'s sha256 in the parent can then take the recorded hash. That saves one
    serial pass over G; `check_compacted_copy` still hashes independently.
- **Expected saving: ~20–45 min of wall at 50–100 MB/s, 4–8%. Unmodelled.**
  - G ≈ 115–135 GB fetched by `snapshot_download`, plus its disk write.
  - About 2 h on the ~15 MB/s link the gz option assumes.
  - The serial link hash: G at ~0.66 GB/s (measured here) is ~3 min more.
- **Blocked by:**
  - **The data contract and the Hub:** nothing. The uploads, the build branch and
    `hub.py` are unchanged, and the files are the bytes the Hub has, by its own sha256.
  - **Disk, to be weighed:**
    - each set's group files stay from its pipeline to its finalise, rather than being
      fetched at finalise;
    - while one set is finalised, the other's (~60–70 GB) is held;
    - for a laptop with a few hundred GB, a setting (on by default, off where disk is
      short) is the likely form.
  - **The bench can't show it.** It leaves out every transfer. Showing it needs a
    modelled transfer term in bench/, which B's stack owns now.
- **Feasibility:** code read.
  - `close_group`'s `rmtree(staging_dir / table)` (`push/core.py`);
  - `compact_table`'s unconditional `download` (`compact.py`);
  - `verify_group`'s sha256 comparison;
  - claims_labels' build into `compact/src` (precedent).
  - **Still to check:** whether `snapshot_download` re-hashes or re-fetches local files
    that have no cache metadata. Only for the fallback; no Hub calls were made here.
- **Files:**
  - `src/wikidata/push/core.py` (`close_group`'s clean-up only);
  - `src/wikidata/compact.py` (`download` / `compact_table`'s first stage, `_link_run`'s
    hash). #135 owns `rewrite_table`'s planning, so rebase on it.
  - `src/wikidata/config.py` (the setting);
  - `scripts/test_merge.py`, `scripts/test_finalise.py`.

### E3. Entities built in the split, so the chunk file and `process()` go

- **Mechanism:**
  - After B and A, the chunk file carries `id` and `entity` (JSON text) for the
    entities table alone.
  - The split builds the entities rows (`ENTITY_SCHEMA`'s fields, which reshape already
    parses) and writes them as the other tables' are.
  - **Then nothing reads the chunk file.** The pipeline's per-chunk worker reduces to
    merging and uploading:
    - no `pl.read_parquet` of the chunk;
    - no `check_ids` scans (moved into the split, as #134 moved the claims');
    - no worker starts;
    - no `process()` / `partition_chunk` step states.
  - `scripts/rebuild_entities.py` already builds entities in one pass from the dump
    (decode, reshape, route, entity rows, partition).
- **Expected saving: 15–25 min, 3–4%, modelled.**
  - **Basis:** "the pipeline: the rest" is 7.5 CPU-s. Its timed stages come to about
    6.2 s: partition 3.4, merge 1.0, sink 0.4, sitelinks 0.4, other 0.2, the maps'
    normalise ~0.8.
  - **Unattributed:** ~1.3 s × 1,104 ≈ 24 min (the chunk read, the id checks, the
    entity decode, imports).
  - **Plus** the starts (~3 min), less the entity build added to the split (`entity` is
    small).
- **Blocked by:** nothing in the contract.
  - **Sequencing:** it follows A and B, which own `dump.py` and `process.py` now.
  - **Restart:** the split's manifest becomes the only per-chunk ledger.
- **Feasibility:** code read. `process()` reads `entity` only for
  `pl.col("entity").str.json_decode(ENTITY_SCHEMA)`, so the fields are fixed.
- **Files:**
  - `reshape/src/` (an entities builder);
  - `src/wikidata/dump.py` (no chunk file);
  - `src/wikidata/process.py`, `src/wikidata/main.py` (`process_and_partition`,
    `partition_chunk`);
  - `src/wikidata/pool.py`, `src/wikidata/state.py`.
  - **Overlaps A and B:** start after them.

### E4. claims_labels' refs taken where the claims are built

- **Mechanism:** the split's claims build (#134's `claims_build`) writes each part's
  distinct (field, ref, id) beside its claims file. `collect_refs` becomes a union of
  those sidecars, not a re-read of every sorted claims file's 9 leaves.
- **Expected saving: ~7–12 min, 1–2%, plus memory. Unmodelled:** the bench doesn't run
  claims_labels.
  - **Measured here** (`refs_cost.py`, the bench's 4-chunk claims group file, 73 MB,
    1.18M rows):
    - `file_refs` takes 1.8–2.7 CPU-s;
    - the same refs from arrays already in memory take ~0.5 CPU-s.
  - **At a release:** ×330 that is 10–15 min removed from finalise and ~3 min added to
    the split (×327).
  - **Memory:** the 6.8 GiB a 529 MB file that `file_refs` holds goes, the largest
    finalise peak outside the bucket sorts on a 16 GB laptop.
  - **The refs as Parquet:** 3.4 MB, 4.7% of the claims bytes before deduplication
    across files.
- **Blocked by:** nothing. claims_labels is a function of the set of refs. Parity on
  claims_labels shows it.
- **Files:**
  - `src/wikidata/claims_build.py` (#134's: one call after `statements`);
  - `src/wikidata/claims_labels.py` (`collect_refs`, `collect_refs_stage`);
  - `src/wikidata/process.py` (`place_built_claims` places the sidecar);
  - **after #134 merges.**

### E5. The claims bucketed at write time

- **Mechanism:**
  - The claims are written in id-range row groups (or per-bucket group files), cut at
    boundaries fixed before the split. For example, from the previous release's part
    boundaries: ids are stable across releases.
  - Each bucket sort then reads only its own row groups. `bucket_key`'s planning, the
    spilled fragments and the 1.13–1.21× read amplification go.
- **Expected saving: ~10–15 min, ~2%, modelled.**
  - `_bucket_source` is 6 min, understated: a release spills 20–30% against the 3×
    run's 10%.
  - Plus ~15% of `_read_bucket`'s 20 min, and the bucketing stage's disk.
- **Blocked by:** the pending threshold decision. The boundaries would no longer be
  `ceil(bytes / SORT_BUCKET_BYTES)` quantiles of this release's ids, so the published
  part boundaries change.
- **Files:**
  - `src/wikidata/claims_build.py`;
  - `src/wikidata/sort_by_id.py` (`bucket_key`, `_read_bucket`);
  - the group rule.

### E6. claims_labels written as its published files

- **Mechanism:** `_write_language` already writes each language's rows sorted by
  (`ref`, `field`), which is the stable sort by `ref`'s order. The rows are already
  distinct (refs unique, a label per id and language). So the build can write the sort's
  part files and checks directly, skipping compaction's deduplicating rewrite and the
  sort's re-sort.
- **Expected saving:** a few minutes. claims_labels' size isn't on record.
- **Blocked by:** the part and row-group rules must be reproduced (the same issue #135
  solved with planned sizes). The deduplicating rewrite is the only path a deduplicated
  table has today.
- **Files:** `src/wikidata/claims_labels.py`, `src/wikidata/compact.py`
  (`_write_deduplicated_key`).

### What the pending threshold decision is worth

- **One file per table:** 28 min.
- **E2 and E5:** 35–60 min.
- **Overlap:** they overlap, as E2 subsumes one file per table.
- **So the decision is worth about 45–70 min, 8–12% of a release, modelled.** It is the
  largest single lever left that isn't a decoder choice.

## Considered, and not proposed

- **Skipping the build branch's upload** (local restart only): this is the largest
  transfer that can be avoided, about 5 h at the one upload rate on record. It is kept
  by the user's steer as the cross-machine restart point.
- **A single encode of the claims** (the build branch's files sorted already): the
  global order needs every chunk, so the build branch's files can't be sorted when
  written. Two encodes and two decodes are the floor while the build branch holds
  encoded files.
- **Hashing:**
  - three sha256 passes over G: `verify_group`, `_link_run` and `check_compacted_copy`;
  - E1 removes `_link_run`'s;
  - `check_compacted_copy` is kept.
- **route-release:** since the split routes as it writes, it only renames (2 min).

## The bench, and what it can't see

- **Every transfer is left out:**
  - the build branch's upload of G (overlapped with the pipeline's compute today);
  - finalise's download of G;
  - the publishing commit.
- **Why it matters now:** the work in flight moves most of the pipeline's compute into
  the split: the claims stages (1h42m modelled) with B, and the maps' and links'
  normalise and partitioning (about 1h30m) with A. That leaves well under an hour of
  pipeline compute for the group uploads to hide under. At the one
  upload rate on record (7 MB/s, `2026-10-05-sort-speed.md`), G takes ~5 h, more than
  the pipeline's modelled 4h03m today.
- **What would settle it:** a release-shaped measure of the Hub's upload and download
  rates, or a bench term that models them. Either would decide E1's size and show
  whether uploads already bound a release's wall.

## Missing

- **The other tables' bytes for the scholarly set,** and so G. The ranges above assume
  ~20–40 GB for both sets' other tables.
- **Hub transfer rates:** only 7 MB/s for one upload in 20260928 is on record.
- **claims_labels' size,** for E6.
- **Fusion A's per-file cost in Rust,** which sets where E2 falls in its range.
