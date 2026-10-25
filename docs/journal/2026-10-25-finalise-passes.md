# 2026-10-25: Finalise's passes over the claims, accounted at a release

This entry checks a hypothesis about finalise, the second-largest stage of a release once
fusion A (#148) lands, and re-ranks what is left to remove from it. It also does the
finalise half of the reassessment the user asked for after fusion: whether the bucketing
stage can go. The script is in `2026-10-25-finalise-passes/`.

## Current State

- Docs only: nothing in `src/`, `reshape/` or `bench/` changed.
- **Master** is f806305 (#135).
- **The figures are bench 13.0.0's,** from #135's run 40221382555. They are its "after"
  arm, which is master's code, over 6 runs (3 rounds × 2 runners).
- **The modelled release:**
  - finalise is 2h36m02s of about 8h30m;
  - with fusion A's pipeline (~1h51m), finalise is about 35% of a ~7h30m release.
- **The hypothesis:** finalise's time is mostly the claims' bucket path, and that path
  takes the claims through at least one full write and read of scratch disk between the
  group files and the published parts.
  - **The time half holds.** The claims are 88.6 min of finalise's 156.0 (57%). After
    the model's overstatement of the other tables (below), they are about 80% of a real
    release's finalise.
  - **The round-trip half holds only at a release, and for the sorted claims.**
    - **What doesn't round-trip:** since 2026-10-25 the unsorted rows go mostly
      straight from the group files into the bucket sorts. Only the spilled rows go
      through scratch: 13% on the bench's 3× run, 20–30% modelled at a release.
    - **What does:** at a release every sorted bucket is written to a `final-` scratch
      file, then glued into its part file and checked byte for byte. The bench never
      runs this, as its 3× run has one bucket a part.
  - **The disk half doesn't hold.**
    - **On CI** the bucket path is CPU-bound: the cores waiting on I/O are 0.00–0.01,
      and the disk is 3–4% busy.
    - **At a release on a laptop's NVMe** it stays CPU-bound: about 540 GB of I/O for
      the main set's claims, over about 44 min of CPU-bound work, is ~200 MB/s.
- **The judgement:** no structural saving in finalise is both in bounds and worth
  building now.
  - **About 76 min of the claims' 88.6 min is the floor the contract sets:** one decode
    for the sort, the sort, one encode for the Hub, and the kept independent decode.
  - **What remains above the floor:**
    - bucketing, 10.0 min modelled;
    - read amplification, ~2 min;
    - the glue at a release, disk only and unmodelled;
    - the other tables' per-file costs, 21 min modelled but ~1–3 min at a release.
  - Each is either blocked (by the published boundaries, or by a pq-glue writer) or
    small at a release.

## Sources

- **#135's bench comment** (bench 13.0.0, run 40221382555), with its artifacts: each run's
  `phases.jsonl` and results JSON (`gh run download 40221382555`).
- **Per stage:**
  - each stage's own CPU (`self`) on the 3× run, less the first run, is the part per
    byte the model scales;
  - a call is the claims' if it starts inside the claims' `compact_table` or sort
    `write_table`;
  - it is converted at the model's rates: the claims at 1,233.2 / 2 / 4 = 154 ETA-s a
    CPU-s, the rest at 3,666.9 / 2 / 4 = 458.
- **Agreement with the model:** the stages, the release's claims workers' starts (186 s)
  and the fixed part (27 s) come to 151.5 min of the 156.0 min finalise ETA. The 4.5 min
  left is per-byte CPU outside the timed stages.
- **The release's shape:**
  - **the main set,** from 20260928: 44.7 GB of claims in 30 group files, 3,769 row
    groups, 667 buckets, 86 part files (`2026-10-05-sort-speed.md`);
  - **both sets,** from the bench's release model: 1,171,406 rows a bucket, the claims
    90.1 GB;
  - **the spill,** modelled at 20–30% (`2026-10-25-architecture-trace.md`,
    `2026-10-25-claims-presorted.md`).
- **Code read:**
  - `sort_by_id.py`: `bucket_key`, `_bucket_source`, `_direct`, `_read_bucket`,
    `_sort_bucket`, `sort_buckets`, `_pack_file`, `KeyPlan.pack`;
  - `compact.py`: `source_group_files`, `rewrite_table`, `_link_run`;
  - `config.py`: `SORT_BUCKET_BYTES`, `COMPACT_FILE_BYTES`, `SORT_HUB_BYTES_PER_DECODED`.

## Finalise's modelled release time, by stage

`account.py` on the after arm's 6 runs. The ± is the standard deviation over the runners,
whose speeds differ by up to 1.7×. The shares are steady: the claims are 81–83% of the
part per byte in every run.

| part | stage | modelled | ± | share | what it is |
|---|---|--:|--:|--:|---|
| claims | `compact._write_file_parallel` | 35.1 min | 5.2 | 22.5% | **floor:** the part files' one encode (zstd 3, CDC, page index) |
| claims | `_check_written` | 18.5 min | 2.4 | 11.9% | **floor:** the kept independent decode |
| claims | `_read_bucket` | 15.3 min | 1.8 | 9.8% | **floor:** the decode for the sort, ×1.13–1.21 for row groups read by two buckets, plus the fragments' lz4 |
| claims | `_sorted`, `_check_sort` | 9.5 min | 1.0 | 6.1% | **floor:** the sort, and the take of every column |
| claims | `_bucket_source` | 7.2 min | 0.8 | 4.6% | bucketing: each row group's ids read and counted; row groups with spilled rows decoded whole and spilled |
| claims | `_read_streams`, `_ipc_stream` | 2.4 min | 0.3 | 1.6% | the fragments: lz4 IPC written, read back and compared, and read again by the sort |
| claims | `_boundaries`, `rewrite_table` | 0.6 min | 0.0 | 0.4% | the ids' cells counted; the link and its hash |
| **claims** | **all** | **88.6 min** | | **56.8%** | |
| rest | `_read_key` | 12.7 min | 2.3 | 8.2% | the decode, and each file opened |
| rest | `compact.rewrite_table` | 10.4 min | 2.3 | 6.7% | **per file:** each link's footer read, sha256, and the key's plan (footers again) |
| rest | `compact._write_file_parallel` | 9.1 min | 1.2 | 5.9% | the encode |
| rest | `check_compacted_copy` | 7.9 min | 1.5 | 5.1% | **per file:** each file's sha256 (kept by decision) |
| rest | `_sorted`, `_check_sort` | 8.3 min | 1.4 | 5.3% | the sort |
| rest | `_check_written` | 5.5 min | 0.8 | 3.5% | the independent decode |
| rest | `sort_by_id.write_table` | 3.0 min | 0.8 | 1.9% | **per file:** each key's plan, from its files' footers |
| rest | others | 2.5 min | | 1.6% | |
| **rest** | **all** | **59.4 min** | | **38.1%** | |
| model | the claims workers' starts | 3.1 min | 0.4 | 2.0% | 1,300 at a release |
| model | fixed (per key) | 0.4 min | 0.1 | 0.3% | |
| | per-byte CPU outside the stages | 4.5 min | | 2.9% | |
| | **finalise** | **156.0 min** | 22.6 | | |

- **By step, as asked:**
  - **compaction** (`compact.py`): 11.1 min, 7.1%. All but 0.3 min of it is the other
    tables' per-file links.
  - **the claims' bucketing:** 10.0 min, 6.4%.
  - **the claims' bucket sorts:** 78.4 min, 50.3%. The read is 15.3, the sort 9.5, the
    part write 35.1 and `_check_written` 18.5.
  - **the checks, all tables:** 34.4 min, 22.1%: `_check_written` 24.0,
    `check_compacted_copy` 7.9, `_check_sort` 2.5.
  - **claims_labels:** not run by the bench, so not modelled. It belongs to E4.
- **The claims' floor** is 35.1 + 18.5 + ~13.3 + 9.5 = **~76 min**. The ~13.3 is
  `_read_bucket`'s 15.3 less about 2 min of read amplification and fragments.
- **The claims' bucketing** (`_bucket_source`, the fragments, `_boundaries`) is
  **10.0 min**, 6.4% of finalise. With the read amplification it is ~12 min, 7.7%.
- **The rest's per-file stages** (`rewrite_table`, `check_compacted_copy`,
  `write_table`) are **21.3 min**, 13.7% of finalise. They don't scale with bytes:
  - **The evidence:** on the first run, aliases' `rewrite_table` takes 0.133 s for 489
    keys of one file each, 2.7 MB in all. That is 0.27 ms a file. Hashing 2.7 MB at the
    claims' measured 1.3 GB/s would take 2 ms.
  - **What the 3× run measures:** it adds two files to every key (2,323 keys), and the
    model scales that by entities, as 3,667 extra files a key.
  - **A release has ~30 a key** (a group each): about 160k files over both sets, ~1–3
    min at ~0.6 ms a file. **The model overstates these stages ~15–50×.**
- **The rest's byte-scaled stages** (decode, sort, encode, check: 35.6 min) are
  overstated ~3× by entity scaling (#131).
- **So a real release's finalise** is likely ~1h45m–1h55m, the claims ~80% of it.
- **Small misattribution:** `check_compacted_copy` of the claims (0.11 CPU-s) falls in
  "rest", as it runs before the claims' sort `write_table`. At the claims' rate it would
  be 0.3 min, not 0.8.

## The claims' passes, between the group files and the published parts

Each pass over a set's claims at a release, in order. The sizes are for the main set
(44.7 GB of group files, ~43 GB of part files); the scholarly set's are similar.

| # | pass | reads | writes | CPU, modelled | kept because |
|--:|---|---|---|---|---|
| 1 | `source_group_files`: sha256 of each kept group file against its verified record (only with kept files, #143) | 44.7 GB | | unmodelled | the kept files' check against the Hub's bytes |
| 2 | `_link_run`: a hard link, then the file's sha256 for compaction's manifest, serial in the parent | 44.7 GB | | 0.3 min | the manifest's hash |
| 3 | `check_compacted_copy`: sha256 against the manifest, FINALISE_WORKERS threads | 44.7 GB | | 0.8 min | decided |
| 4 | `_boundaries`: the id column, cells counted (Polars, streaming) | ids only | | 0.3 min | the boundaries are this release's row quantiles |
| 5 | `_bucket_source`: each row group's ids; a row group with spilled rows decoded whole, its spilled rows taken, lz4 IPC, read back and compared, digested | ids, + 9–13 GB | fragments ~20–30 GB | 9.6 min, with the fragments' IPC | the external sort's routing |
| 6 | `_read_bucket`: each bucket's row groups decoded directly (×1.13–1.21) and its fragments | ~39 + ~25 GB | | 15.3 min | the sort's decode |
| 7 | `_sorted`, then `_write_file_parallel` to a `final-` scratch file (a part of one bucket: the part itself) | | ~43 GB | 44.6 min | the sort and the one encode |
| 8 | `_check_written`: the scratch file decoded by pyarrow, compared with the sorted rows | ~43 GB | | 18.5 min | decided |
| 9 | `_pack_file`: `pq_glue.concat` of a part's buckets (about 8 for the main set, 10 for the scholarly set), then `_check_glued` byte for byte against them | ~43 + ~86 GB | ~43 GB | ~0 (on a thread) | a part is several buckets |
| 10 | `_entry`: the part's sha256 | ~43 GB | | | the manifest and the commit |

- **About 430 GB read and 110 GB written for the main set.** Over its ~44 min of
  modelled CPU-bound work, that averages ~200 MB/s.
- **A laptop's NVMe** (2–3 GB/s read, ~1 GB/s sustained write) is ~10% busy, so the
  path is CPU-bound.
- **Slower disks:**
  - on a SATA SSD (~500 MB/s) the disk is ~40% busy;
  - on a ~150 MB/s disk the path is disk-bound, at ~60 min of I/O against 44 of CPU.
- **The glue (pass 9) is the largest removable I/O:** 172 GB, a third of the total.
  Pass 2 is next (45 GB, with kept files).
- **Fragment sizes are an estimate.** lz4 IPC took 2.2× the Parquet bytes of a claims
  file in an older schema, measured here (57.6 MB of Parquet, 127.4 MB of IPC). Today's
  schema isn't measured.

## The candidates, evaluated

### A k-way merge over sources each already sorted: not possible, and no saving if it were

- **The sources aren't sorted by string id.** Each group file is in dump order, which is
  numeric id order with local inversions and properties among the items.
- **The nearest thing to sorted runs:** within one id length, numeric order is string
  order. So a group file is a few sorted runs a length, broken by the inversions.
- **What a merge would remove:** only `pc.sort_indices`.
- **What it would keep:**
  - the take of every column, which is most of `_sorted`'s 9.2 min;
  - the decode of each output range's rows. Their row groups are the ones bucketing
    reads now, as a short-id row group's rows fall in tens of buckets however they are
    ordered (`2026-10-25-claims-presorted.md`).
- **No pass is removed.**

### In-memory buckets within a memory budget: the spilled rows don't fit

- **The spilled rows at a release:** 20–30% of the main set's rows, 9–13 GB of Parquet,
  ~150–300 GB decoded.
- **Why they can't wait in memory:** they come from every source file, for every
  bucket, and the first bucket's sort needs the last source's rows.
- **The fragments are the bounded-memory form of that wait:** lz4 IPC, written once,
  read back once to check and once to sort.
- **A bucket reading a spread row group directly** (no spill) reads it once per bucket,
  tens of times for a short-id row group. `_direct` already makes that trade a row group
  at a time: k direct reads against 1 + 2 × the share spilled.

### Fewer passes over the claims bytes since compaction only links: one hash, ~1 min a set

- **What's left between compaction and the sort:** three sha256 passes (pass 1 only with
  kept files), two id-column reads, the spilled row groups' decode, and the sort's
  decode (passes 1–6).
- **The removable one:** pass 2's hash, when the group files were kept and pass 1 has
  just checked them against their verified sha256.
  - `_link_run` could take that sha256. The link is the same inode, and pass 3 then
    checks the local copy against the Hub's own hash.
  - That is a stronger chain than now.
  - **Saving:** one serial read of 44.7 GB a set, ~35–70 s at 0.66–1.3 GB/s.
- **The bench can't show it:** it records the download as done, and never runs the kept
  path.
- **Not worth a PR by itself:** below 0.5% of a release, and unmeasurable.

### A decode done twice of the same bytes: ~2–3 min modelled, each priced or a check

- **A row group with any spilled row** is decoded whole in bucketing, then again by each
  bucket that reads it directly. `_direct`'s cost model prices this as the 1 in 1 + 2 ×
  the share spilled. Removing it means spilling none of a shared row group, which is the
  read amplification above.
- **The id column** is read by `_boundaries` and again by `_bucket_source` for every row
  group. `_boundaries` is 0.3 min modelled. The id read in `_bucket_source` is a small
  part of its 7.2 min, most of which is the spilled row groups' decode, take and IPC.
  - **The fusion that would remove it:** record each row group's cell counts in one id
    pass, then derive the boundaries and each row group's buckets from the counts, as
    buckets are cut between cells.
  - **Its saving:** ≤ ~1 min.
  - **Overlap:** it is E5 (d)'s sidecar computed in finalise rather than in the split.
- **Each fragment is decoded twice:** the read-back check at write, then the sort. That
  is ~0.65 min each. The check stays: nothing later compares the fragment's non-id
  columns with the source's.

### Bucketing gone through a simpler design: not without changing the published files

- **What the bucket sorts need:** each bucket's rows, in source order, before its sort.
- **What would remove the bucketing stage:** rows reaching finalise already partitioned
  by bucket. But the buckets are this release's row quantiles, cut between cells
  (`_boundaries`), and their rows decide the part files (`KeyPlan.pack`) and each part's
  row groups (`row_group_rows` a bucket).
  - **Cutting upstream** by any boundaries known before the split changes the published
    files. They must not change again.
  - **E5 (d)** (`sort/cells-sidecar`, 099cf0d) keeps the boundaries. The split
    reorders each part's rows by cell and records each row group's cells. That removes
    the id reads and most of the bench's spill (92,705 → 1,638 rows on test_sort's
    sample).
  - **What E5 (d) leaves:** a release's short-id row groups span tens of buckets
    whatever their order, so their spill stays.
  - **Its value is unchanged** from its entry: ~1–1.5% of a release, on the split's
    side.
- **Bucketing's modelled cost is 10.0 min, 6.4% of finalise.**
  - At a release's 20–30% spill (against the 3× run's 13%) it is likely ~15–20 min.
  - It is the external sort's routing pass, at about one extra decode of the spilled
    rows.

## Ranked: what remains, and why none is built here

| # | what | modelled | at a release, likely | why not now |
|--:|---|--:|--:|---|
| 1 | The sorted claims' glue (pass 9): each bucket encoded to a `final-` file, then concatenated and byte-checked | 0 (the bench has one bucket a part) | ~0 wall on NVMe (a thread beside CPU-bound sorts); up to ~20 min on a ~150 MB/s disk; 172 GB of I/O for the main set | **Same bytes need an appending writer** that adds encoded row groups to an open part file, as `concat` glues them; pq-glue has none (`concat`, `concat_many`, `write_parquet` only). **The other forms change the layout:** one part a bucket (667 files of ~65 MB, rejected in `2026-10-25-release-profile-bench10.md`), or part-sized buckets (~10–14 GB of decoded rows a sort, past a 16 GB laptop). |
| 2 | Bucketing's spill (passes 4–5, the fragments) | 10.0 min | ~15–20 min | The routing an external sort needs. E5 (d) cuts the bench's spill but not a release's short-id spill (~1–1.5%), and is on the split's side. |
| 3 | The other tables' per-file stages | 21.3 min | ~1–3 min | A model artifact (above). Caching footers between `rewrite_table`'s plan, `_link_run` and `write_table`'s plan would show ~5–10 min on the bench for ~1 min at a release. |
| 4 | Read amplification in `_read_bucket` | ~2 min | ~2–3 min | Only pages wholly outside a bucket could be skipped. Page pruning through pq-glue was no cheaper (#110). |
| 5 | `_link_run`'s hash with kept files | 0 | ~1 min a set | Unmeasurable on the bench, and below 0.5%. |
| 6 | The id column's second read in bucketing | ≤ 1 min | ≤ 1 min | Below noise. |

## Missing

- **A release-scale disk run.** None has sized passes 1–10 on real hardware. The
  fragments' bytes in today's claims schema and the glue's wall time beside the sorts
  are both estimates.
- **The scholarly set's spill share.** It is modelled, not measured. Its ids are mostly
  8–9 digits, so it likely spills less than the main set.
- **A per-file term for the other tables in the bench's model.** Their stages that cost
  per file would then scale by a release's files (~keys × groups), not by entities.
  bench/ is owned elsewhere; noted here only.
