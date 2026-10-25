# 2026-10-25: claims_labels' refs from sidecars written with the claims (E4)

The architecture trace (#136, E4) found claims_labels' refs stage reading every sorted claims
file again: `file_refs` decodes 9 leaves of each, about 2 CPU-s and 6.8 GiB for a 529 MB
file, 2 files at once, about 7–12 min of a release's finalise. Yet the claims are in memory
where they are built. lmmx/wikidata-pq#139 takes the refs there.

## Design

- **Where the claims are in memory.** Since #134, a part's claims partition file is written
  in two places:
  - by the split (`claims_build.write_claims`), from the reshape's arrays;
  - for a part left as JSON (a misfit), by the pipeline (`process.normalise_claims_direct`).

  - **The JSON path** passes the statements table to `write_table`. The hook replaces that
    call with `claims_refs.write_with_refs(table, tmp, claims_file)`: the file to `tmp`,
    then its sidecar, then the caller's rename.
  - **The split, since #141,** writes a part a batch at a time through
    `claims_build.RowGroups`, never holding it whole. The hook wraps that writer:
    `out = with_refs(RowGroups(...))`. It gathers each batch's refs as it is added, and
    writes the sidecar in `close`, before the glued file is renamed into place.
  - Either way, a claims file in place always has a sidecar. A part that turns out to be
    a misfit mid-way is never closed, so it gets no sidecar from the split; the
    pipeline's JSON path writes its sidecar instead.
- **What a sidecar holds.**
  - The statements' distinct `(field, ref)`, taken by `table_refs`, which is `file_refs`'
    own code given the table (now shared by `file_refs` too), sorted.
  - The statements' row count, in its metadata.
  - Not `id`: it is a function of `(field, ref)` in `_snak_refs`, and half the bytes. It is
    put back on read.
- **Naming and place.**
  - The obvious name, the chunk's, does not work. The split numbers parts by split chunk,
    and routing renames them to set chunk numbers (`_move_routed`). A sidecar inside
    `routed/` would also stop routing's `rmdir`. dump.py was not this change's to touch.
  - So a sidecar is named by its content: `{rows}-{first id}-{last id}.parquet`. It is
    one name whichever of the two writers wrote the part, and a rewrite replaces it.
  - It is kept in the set's work directory (`claims_refs/`). The writer finds that from
    the claims file's path: the `WORK_DIR` (this set's or the other's) that holds it.
- **Completeness: row counts, not names.**
  - Each statement is in exactly one claims file, and the sorted claims are those files'
    rows.
  - `sidecar_refs` therefore takes the union only when the sidecars' row counts add up to
    the sorted claims' rows, read from their footers. A missing sidecar or a stray one
    makes the sums differ, and `collect_refs` reads the claims as before.
  - The union deduplicates as it reads, whenever the rows read since the last pass exceed
    those kept, so it holds about twice the distinct refs (20.7M at a release).

## Correctness

`scripts/test_claims_refs.py` was run on all 11 bench chunks. Every check passed:
- **Per chunk:**
  - the split's and the pipeline's claims files, and their sidecars, are the same bytes;
  - each sidecar equals master's `file_refs` (copied verbatim into the test) of its claims
    file.
- **The 11 chunks sorted by id into two files:** the union equals `collect_refs` (422,732
  refs), `refs.parquet` is byte-identical, and so are `write_groups`' 520 language files
  (204,186 rows, with the chunks' labels partitioned by language).
- **Fallback:** a missing sidecar and a stray one each fall back.

**Also run:**
- `scripts/test_split_claims.py` (r50, s30) passes, including its three
  interrupted-and-rerun cases.
- A real split, then each set's `process`, with a misfit part kept as JSON: both sets'
  sidecars cover every claims row, the JSON part's included.

## Measured locally (indicative)

The 11 chunks: 88.6 MiB of claims files, 94 MiB sorted, 1.56M statements. Each figure is
from fresh processes, three runs.

| | before | after |
|---|---|---|
| refs stage CPU (with workers) | 2.5–3.4 s | 0.37–0.40 s |
| refs stage wall | 2.4–3.6 s | 0.41–0.43 s |
| refs stage peak | an 870–920 MiB worker | no worker |
| split CPU for the sidecars | – | +1.4–1.6 s |
| sidecars on disk | – | 2.25 MiB (2.5% of the claims bytes; 4.5% on s30) |

- `file_refs` on the 94 MiB file takes 1.7–2.1 CPU-s in one process. `table_refs` takes
  0.6 CPU-s on the same table in memory, which matches the trace's 1.8–2.7 against 0.5.
- **Per chunk, the in-memory cost is higher than the trace's estimate.** The sum over
  chunks pays the per-call overhead 11 times, plus each sidecar's sort and write: 0.47
  CPU-s on c0, 0.37 on s90.
- **So the CPU moves more than it disappears:** net about −1 to −1.6 CPU-s per 94 MiB,
  roughly half. What does go entirely is the finalise worker's peak.

## On CI (bench 12.0.0, quick tier, 3 rounds, parity passed)

`fa959e4` → `2bea4ad`. The bench runs no claims_labels, so it can show only the split's
cost.

| | before | after | change |
|---|---|---|---|
| measured end to end, CPU (split + pipeline + finalise) | 2m13s ± 20.6s | 2m15s ± 19.3s | within noise |
| measured end to end, wall | 50.16s ± 7.61s | 50.20s ± 7.28s | within noise |
| CPU, split (bz2) | 34.7s ± 7.6s | 35.8s ± 6.8s | within noise |
| CPU, split from gz | 14.9s ± 2.8s | 15.6s ± 3.0s | **+0.8s (+4.9% ±3.0%)** |
| ETA, a release | 8h34m49s | 8h37m56s | within noise |
| ETA: split (bz2) | 3h08m44s | 3h15m06s | within noise |
| ETA: split from gz | 1h23m00s | 1h27m11s | **+4m11s (+4.9% ±3.0%)** |
| ETA: finalise | 2h42m19s | 2h40m17s | within noise (no claims_labels in it) |
| CPU, finalise | 1m29s ± 11.4s | 1m29s ± 10.9s | = |
| peak RSS, split (GB) | 3.545 ± 0.149 | 3.637 ± 0.107 | within noise |
| peak RSS, finalise: all processes / one worker (GB) | 4.173 / 1.813 | 4.158 / 1.817 | = |
| peak RSS, pipeline: all processes (GB) / one worker (MB) | 2.144 / 579 | 2.146 / 602 | within noise |

- The split's added CPU, about +1 s on the 4 chunks, matches the local estimate (c0, r50,
  s30 and s90: about 1 CPU-s).
- At a release, the bench's model makes that about +4–6 min of split ETA.
- The finalise side is unmodelled. Locally, the refs stage's ~2.5–3 CPU-s per 94 MiB of
  sorted claims becomes ~0.4 s.
- Scaled by a release's claims bytes (the trace's 96.5 GB; the bench models the claims key at
  90.1 GB; about 1,000× the sample's sorted file), that is:
  - the refs stage's CPU from about 45 min to about 6 min;
  - its wall (2 workers) from about 20–25 min to the union's few minutes;
  - its two 6.8 GiB workers gone.

  This is modelled from local runs, not measured.
- The net release ETA change cannot be shown by this bench. Whether it merges is the
  coordinator's call: a measured +5% on the gz split against an unmeasured finalise saving
  and a 13.6 GiB finalise peak removed.

## On CI with the refs stage measured (bench 14.1.0, quick tier, 3 rounds, parity passed)

#145 added claims_labels' refs stage to the bench, timed apart from the figures it had. It
runs after each finalise run (the sample's rows, and 3× them). Master was measured against
itself at #145's head af7fa2e: nothing beyond noise. This PR was measured at e37f517,
against master 0a574cf.

| | before | after | change |
|---|---|---|---|
| claims refs, wall | 1.01s ± 0.06s | 0.15s ± 0.01s | **−0.85s (÷6.5)** |
| claims refs, CPU | 2.2s ± 0.1s | 0.5s ± 0.0s | **−1.8s (÷4.6)** |
| claims refs, 3× the rows: wall / CPU | 2.09s / 6.0s | 0.22s / 0.6s | **÷9.4 / ÷9.7** |
| peak RSS, claims refs: all processes | 821 MB | 412 MB | **−46% ± 17%** |
| peak RSS, claims refs: one worker | 508 MB | 21 MB | **÷17.5** |
| peak RSS, claims refs 3×: all processes / one worker | 1.52 GB / 661 MB | 0.51 GB / 21 MB | **÷3.0 / ÷30.6** |
| ETA: claims refs, modelled | 8m56s ± 21s | 20s ± 4s | **−8m36s (÷27.7)** |
| CPU, split from gz | 17.1s | 18.3s | **+1.2s (+6.9% ± 3.3%)** |
| ETA: split from gz | 1h35m33s | 1h42m09s | **+6m36s** |
| CPU, split (bz2) | 37.3s | 38.5s | within noise |
| ETA: split (bz2) | 3h22m59s | 3h29m48s | within noise |
| measured end to end with the claims refs: CPU | 2m14s | 2m14s | within noise |
| measured end to end with the claims refs: wall | 51.48s | 50.42s | within noise |
| ETA, a release, with the claims refs | 8h26m40s | 8h26m18s | within noise |
| measured end to end without the refs (split + pipeline + finalise), CPU | 2m12s | 2m13s | within noise |
| CPU, finalise | 1m28s | 1m28s | = |
| peak RSS, finalise (all processes / at a release, modelled) | 3.83 / 20.21 GB | 3.83 / 20.27 GB | within noise |

- **The refs stage, as the bench measures it, goes from reading the claims to reading
  sidecars.** Its time and memory fall by large factors:
  - its modelled time at a release, from about 9 min to 20 s;
  - its worker from 0.5–0.7 GB here to none.
  - The bench's model scales by bytes; it does not model the 6.8 GiB a 529 MB file's
    worker holds at a release.
- **The whole run does not show the gain beyond noise.** The split pays for the sidecars:
  - +1.2 CPU-s, which is significant on gz (+6.9%), and +6m36s of split ETA;
  - on bz2, +1.2 CPU-s and +6m49s of split ETA, within noise.

  That takes back most of the refs stage's −8m36s. The release ETA with the refs is
  8h26m40s → 8h26m18s, −22 s, within noise.
- **What does show beyond noise is memory and finalise's refs time,** both measured: the
  refs stage's peak falls from 0.8–1.5 GB to 0.4–0.5 GB, with no worker.
- **The modelled split cost may be understated, not overstated.** The bench scales the
  split's CPU by the dump's bytes (×1,307 from the sample's 5 parts, as if a release had
  about 6,500), while a release has 23,035 parts. Part of the sidecars' cost is paid once a
  part (Polars frames, three `_snak_refs` passes, a sort, a zstd write); the rest goes with
  the statements.
- **To make the net gain show,** that per-part cost has to come down: for example a
  cheaper refs pass, entirely in Arrow, or a sidecar a group rather than a part.

## Negative findings

- **The bench cannot measure the gain.** Its finalise does not run claims_labels, so it
  shows only the split's added cost. Measuring it would need the bench (not this change's)
  to:
  1. run `collect_refs_stage` after the claims sort;
  2. link the split's `claims_refs/` into the run's work directory (`keep_parts` links
     only parts and claims files);
  3. give each cycled copy of a sidecar its own name in the full tier.

  The union ignores names, but the full tier's repeated parts have the same content name.
  Without distinct names it would fall back, correctly, to `collect_refs`.
- **The net CPU saving is about half, not the trace's ~80%.** The trace compared one file
  read against one table in memory. Per chunk, the refs' fixed costs (Polars frames,
  three `_snak_refs` passes, a sort, a zstd write) are paid once a chunk, 23,035 times a
  release, not once for each of about 330 sorted files.
- **Disk:** about 2.5% of the claims bytes is held from the split to finalise: about 2.3–2.4
  GB for a release's 90–96 GB of claims across both sets (modelled from the bench's ratio).
  The sidecars do not deduplicate across chunks.
  - A group-level merge at group close would cut it, but that is in push/, not this
    change's.
  - Dropping `id` already halved it (975 → 487 KiB on s30). zstd level 9 would save
    another 11%; not taken.
