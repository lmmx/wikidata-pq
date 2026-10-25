# 2026-10-25: Each chunk's claims written sorted by id (tried, not shipped)

## Current State

- Nothing changed on master.
- lmmx/wikidata-pq#125 (not to merge) wrote each chunk's claims partition file stably
  sorted by string `id`:
  - `partitioning.claims.by_id` returns the table as it is when its ids already ascend;
    otherwise it takes its rows in stable id order;
  - `normalise_claims_direct` writes `by_id(statements(claims))`.
- `sort_by_id.py` is unchanged. This measures what finalise's existing bucket planner
  does with sorted sources.
- Parity passed. Published rows and their order are the same, but published claims files
  are **not byte-identical** (see Published files).

## The hypothesis

The hypothesis was that claims rows reach finalise in chunk order, which runs through
the id space several times, so every row group spans the whole id range. If each
chunk's file were sorted:
- each row group would cover a narrow range and land almost wholly in one bucket;
- spill and fragments would shrink, and each bucket's rows would become a few sorted
  runs that a later change could merge.

## Why it doesn't hold at a release

**A chunk is already almost in string order.** The dump is in numeric id order, and
among ids of one length that is string order. Across the bench's 11 chunk files, the
descents of `id` in string order are:

| file | ids | id lengths (Q) | descents | non-Q |
|---|---|---|---|---|
| c0 | Q31 … Q83751 | 2–6 | 9 | 0 |
| r10 | Q93331602 … | 9 | 43 | 1 |
| r50 | Q140072671 … | 10 | 6 | 1 |
| r70, s10, s70 | 9-digit | 9 | 4, 21, 5 | 0 |
| r30, r90, s30, s50, s90 | 9- or 10-digit | one length | 0 | 0 |

- **Only c0 is substantially out of order.** Its ids run through 5 lengths, and in
  string order the 2- to 5-digit ids interleave with the 6-digit ones. The other
  descents are local inversions, or a property among items. These move a few rows a
  short way, inside a row group's range.
- **What the presort changes at a release.** In the 7.4.0 model (8 shard passes, chunks
  of 85k Q numbers), these are:
  - each pass's first chunk (Q1 to about Q85k);
  - the chunks that cross Q100k, Q1M, Q10M and Q100M: about 5 a pass, 40 of some 14k
    main-set chunks;
  - local inversions and properties.

  That is about 1% of the release's rows moved any distance.

**A release's spill (modelled at 20–30%) has another cause, which a per-chunk sort
can't touch:**
- A row group of a chunk with fewer digits than the longest ids covers a string range
  that other chunks' longer ids fill densely. For example, a sorted 7-digit chunk's row
  group, `Q1234000` to `Q1264000`, takes in `Q12340000…` and `Q123400000…` too. That is
  roughly tens of the 667 buckets.
- So it is spilled whole however its rows are ordered.
- A sorted chunk-0-like chunk is no better: its 5 or so row groups each cover a fifth of
  the string range, about 130 buckets each.

**The bench overstates any gain from this change by a large factor.** On the bench, c0
is 620k of the claims key's 1,175k rows, and its unsorted ids are most of the bench's
spill. At a release, chunk 0 is one chunk in 23k.

## Measured on CI (#125, quick tier, 3 rounds, bench 8.0.0)

**The bucket path** (sample rows; 3× the rows):

| | before | after |
|---|---|---|
| buckets | 2; 4 | 2; 4 |
| rows a bucket | 587,556; 881,334 | same |
| spilled | 0%; 10% | 0%; 4% |
| rows a spill flush | none; 23,053 | 2,271; 21,945 |

**Finalise** (per byte unless marked):

| | before | after | change |
|---|---|---|---|
| `_bucket_source` | 2.0 s | 0.8 s | −58.7% ±2.8% |
| `_read_bucket` | 6.3 s | 6.6 s | within noise |
| `_sorted` | 4.4 s | 4.5 s | within noise |
| finalise, all stages | 38.6 s | 37.7 s | |
| CPU, finalise (total) | 1m27s | 1m27s | within noise |
| ETA: finalise | 5h43m07s | 5h42m44s | within noise |
| peak RSS, finalise 3× (all; one worker) | 12.31; 3.80 GB | 12.29; 3.80 GB | within noise |

**The pipeline:**

| | before | after | change |
|---|---|---|---|
| `write_table` | 4.9 s | 4.9 s | within noise |
| `normalise_claims_direct`'s own CPU (the sort) | 0.0 s | 0.4 s | +1118% ±296% |
| ETA: pipeline | 6h25m19s | 6h53m22s | +7.1% ±7.1% |
| throughput | 5,498/s | 5,144/s | −6.6% ±6.3% |
| peak RSS, one worker | 1.754 GB | 2.323 GB | **+32.5% ±3.1%** |

**Overall:**

| | before | after | change |
|---|---|---|---|
| peak RSS, all processes | 3.96 GB | 4.07 GB | within noise |
| disk, peak rise | 565 MB | 566 MB | within noise |
| ETA, a release | 14h36m04s | 14h59m56s | within noise |

- **Parity passed:**
  - `claims/all`'s group files are reordered, as intended;
  - the claims sidecars' `file_size` differs, which parity allows.
- **What the spill saving bought.** It cut `_bucket_source`'s CPU by 1.2 s per byte.
  `_read_bucket` and `_sorted` did not move, so the finalise ETA is unchanged.
- **The worker peak.** The pipeline worker's peak grew by 570 MB, on chunk c0. The
  sorted copy of the statements is taken while genson's nested claims table, which the
  unsorted statements are views of, is still alive.

## Published files

Measured locally on the quick tier's 4 chunks, through `sort_bucketed` as the bench runs
it.

**Rows are the same.**
- The ordered fingerprint of the 3× part files is the same before and after:
  (3525336, 7944385055953193485, 18329989487978413983).
- The bucket count is the same (2; 4).

**Bytes and row groups are not:**
- `sort_by_id` sets the rows a row group from the Arrow `nbytes` of the largest source
  file's first row group (`compact._row_group_rows`). That row group is c0's first. Sorted,
  it holds other entities: 158,708 → 143,503 rows a row group.
- So every claims part file has other row-group boundaries and other bytes.
- **At a release** the same happens if the largest claims group file starts with an
  unsorted chunk. Group 0 starts with chunk 0, the heaviest per entity.
- **No fix that keeps today's boundaries.** `_row_group_rows` measures the rows in that
  one row group, so any reordering of them changes it. Only an order-independent rule
  (for example a key-wide Arrow bytes per row) would remove the dependence, and it changes
  today's figure too. It belongs with the pending decision to decouple the sort's
  thresholds from encoded and in-memory sizes.

**The bucket count** is `ceil(source bytes / 64 MiB)`:
- the presorted group file encodes 0.19% larger (73,014,754 → 73,154,489 bytes);
- at 20260928's 44.7 GB (667 buckets, about 63 MB from the next threshold), about 1% of
  rows re-encoded would move the total by about 1 MB;
- so it is unlikely to change the count, but nothing rules it out.

## Finalise follow-ups considered (not made)

- **A merge of sorted runs in place of `_sorted`'s sort:** little to gain.
  - A release's chunks are already sorted runs, without this change.
  - `_sorted`'s cost is the take, not the sort: `sort_indices` is 0.1–0.2 s of a bucket.
  - A merge still has to gather the rows into their new order, which is the take.
- **Skipping the take when the sort indices are the identity:** rarely applies at a
  release, where a bucket draws on about 16 source files whose rows interleave. It is
  cheap to check.
- **A release's spill lever:** the width in string order of a short-id chunk's row
  group. It could be narrowed only by smaller row groups in the pipeline's claims files,
  which cost per-row-group overheads in the pipeline, merge and compaction. Its gain is
  unmodelled.

## Not shipped

- **No gain:** no end-to-end figure improves beyond noise.
- **A memory cost:** the pipeline worker's peak RSS rises by a third on the bench.
- **A published layout change:** the claims files' row groups and bytes change, though
  their rows don't.
- **An unrepresentative bench gain:** the one gain, `_bucket_source`'s spill, comes from
  the bench's chunk 0, and a release would not see it.
