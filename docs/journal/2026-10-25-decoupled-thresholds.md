# 2026-10-25: The sort planned from the data alone, not from encoded sizes

Authorised by the user on 2026-10-25 (lmmx/wikidata-pq#97, design B). The published
files' boundaries may change once, provided four things stay the same: the row values,
the schema, the global stable sort by string id, and every exact check.

Scripts are in `2026-10-25-decoupled-thresholds/`.

## Current State

- `sort_by_id.plan_key(table, sources) -> KeyPlan` takes every decision that shapes the
  published files. It reads the files' footers and nothing else.
- `row_group_rows(t)` sizes the row groups from the rows being written.
- Locally, on the bench's group files, the published rows, their order and the schema
  are identical to master's for all 2,324 keys. Only claims/all's row groups change.
- The same rows, re-encoded by another writer, now publish byte-identical files. On
  master they did not.

## Why

Before this change, finalise planned from encoded sizes:

| decision | taken from |
|---|---|
| in memory or through buckets | `sum(source file sizes) > SORT_IN_MEMORY_BYTES` (2 GiB) |
| bucket count | `ceil(source file sizes / SORT_BUCKET_BYTES)` (64 MiB) |
| part count, in memory | `ceil(source file sizes / COMPACT_FILE_BYTES)` |
| part count and cuts, bucketed | `_pack` over the sorted buckets' bytes (#137: over the buckets' shares of the source bytes) |
| rows a row group | `_row_group_rows`: the Arrow bytes of the largest source file's first row group |
| the larger-keys pool | source file sizes over `SORT_IN_MEMORY_BYTES // 8` |

So any upstream change that encoded the same rows differently could change what was
published:
- another writer (#106, one partition file per table);
- another claims writer (#134, the split's own);
- glued or linked group files (#135);
- reordered rows (#125: `_row_group_rows`' sample row group held other entities).

That blocked those changes, and stage 3 (claims bucketed as the split writes them).

## The rules

Each rule is a function of the key's rows and schema alone, and so:
- order-independent: every figure is a sum over the rows;
- encoding-independent: no figure depends on encoded or uncompressed bytes;
- independent of how the rows are split into files and row groups.

**A key's estimated decoded bytes, from its footers (`footer_size`).** Each leaf column
chunk's `num_values` is multiplied by a nominal width for its type, and the products
summed over all row groups. The widths are 1, 4, 8 or 12 bytes for fixed-width types, a
fixed length's own length, and `SORT_STRING_BYTES` (16) for a string. `num_values` counts
the column's definition and repetition levels: one per value, null or empty list. It
depends only on the data and the schema.

On 200,000 claims rows, every writer gave the same counts while their uncompressed sizes
were 2.7 times apart:

| writer | slots | uncompressed |
|---|--:|--:|
| pyarrow, dictionaries | 22,897,531 | 29.7 MB |
| pyarrow, plain | 22,897,531 | 79.9 MB |
| Polars | 22,897,531 | 64.5 MB |
| pq-glue (arrow-rs) | 22,897,531 | 29.5 MB |

Labels and entities behaved the same way. So the footers' uncompressed sizes, which the
brief allowed, are encoding-dependent and are not used.

**Exact decoded bytes, where the rows are decoded anyway (`decoded_bytes`).** This is the
size of the values in Arrow's standard layout:
- a fixed-width value counts its width, and a boolean one byte;
- a string or binary value counts its bytes plus a 4-byte offset (8 for the large types);
- a list counts its offset plus its values;
- a struct counts its fields;
- validity bitmaps are not counted.

It doesn't change when the rows are reordered, rechunked, or read by pq-glue rather
than pyarrow. Each reader allocates validity bitmaps differently: on entities `nbytes`
was 3,489,346 with pyarrow and 3,476,887 with pq-glue, while `decoded_bytes` was
3,472,734 for both. It costs 0.14 s on the sample's whole claims key (1.26 GB decoded).

**The plan:**

| decision | rule | why |
|---|---|---|
| in memory | estimated decoded ≤ `SORT_IN_MEMORY_BYTES` = 8 GiB | it is a memory decision, and decoded bytes are what memory holds |
| the larger-keys pool (execution only) | over 3/16 of it | the bench's pool keeps about its size: 78 keys against 81 |
| buckets | `ceil(estimated decoded / SORT_BUCKET_BYTES)`, 1.75 GiB | as many claims buckets as 64 MiB of source gave: 2 and 4 on the bench |
| parts | `ceil(estimated decoded × SORT_HUB_BYTES_PER_DECODED[table] / COMPACT_FILE_BYTES)`, at most one per row | about 500 MiB on the Hub, from a measured factor per table |
| bucketed part cuts | `KeyPlan.pack(buckets["rows"])`: consecutive buckets cut nearest even rows, at most one part per bucket | known before any bucket is sorted, as #137 needs |
| in-memory part cuts | equal rows (`part_rows`) | as before |
| rows a row group | `row_group_rows(t)`: `COMPACT_ROW_GROUP_BYTES` of the rows being written, by `decoded_bytes`; the whole key in memory, each bucket for itself | the target is decoded bytes, and exact figures are available at write time |
| bucket boundaries | `_boundaries`: equal rows as before, but cut only between cells, an id's first `SORT_CELL_CHARS` (7) characters (`cell`) | a fixed partition of the ids, known before any is written, so stage 3's split can record its rows by cell and give the same boundaries without the ids being read: the boundaries change once, here, not again then |

What stays as it was:
- `_boundaries`' rule is order-independent as it was: equal rows, from the rows counted
  by cell (before, by id), each cut after the first cell bringing the count to a
  multiple of rows / buckets.
- Compaction's `COMPACT_FILE_BYTES` runs are left on file sizes. They no longer shape
  anything published: the plan is the same from glued or linked files, since it sums the
  same row groups' counts. The runs now set only the bucketing jobs.

**The calibration (`SORT_HUB_BYTES_PER_DECODED`).** Each factor is Hub bytes per
estimated decoded byte. They were measured on the bench sample's group files (chunks c0,
r50, s30 and s90 of release 20261012), as the sort wrote each table:

| table | Hub bytes | estimated decoded | factor |
|---|--:|--:|--:|
| labels | 9.28 MB | 29.7 MB | 0.31 |
| descriptions | 8.47 MB | 64.8 MB | 0.13 |
| aliases | 2.71 MB | 8.4 MB | 0.32 |
| links | 6.46 MB | 27.0 MB | 0.24 |
| entities | 0.58 MB | 3.5 MB | 0.17 |
| claims, main set (c0, r50: 665,335 rows) | 47.5 MB | 992 MB | 0.048 |
| claims, scholarly set (s30, s90: 509,777 rows) | 26.9 MB | 986 MB | 0.027 |
| claims_labels (not in the sample) | | | 0.31, taken as labels' |

The claims factor is chosen by set (`SCHOLAR`) because the two sets compress so
differently. The sample's single figure, 0.038, would make the main set's part files 1.25
times the target and the scholarly set's 0.71 times.

**Sensitivity.** A part count is proportional to its factor: 10% more Hub bytes per
decoded byte means 10% more part files. Only keys with more than one part are affected.
At a release those are the claims of each set, each set's entities, and labels/en. The
main set's claims come to 88 parts, against 86 today, and to 79–97 if the factor is out
by ±10%.

## Restart safety

- `PLAN = "decoded-1"` is recorded in each bucketed key's `bounds.json` and
  `buckets.json`, and in each `files.jsonl` part entry.
- A key bucketed, or a part listed, under another rule is made again. Before this change
  those records carried no rule.
- `test_sort.py` covers buckets planned by another rule.
- In-memory keys are recorded per key in the manifest, as before.

## Checks

None was removed or loosened: `_check_sort`, `_check_written`, `_check_glued`,
`_check_sorted`, `_check_whole`, the bucket checks and `check_compacted_copy` are all
unchanged.

One check was added: an in-memory key read must have exactly the rows its plan counted
from its footers.

## Content identity (measured locally)

`fin_driver.py` runs finalise's local stages on the bench quick tier's group files,
exactly as `bench/pipeline.finalise_at_scale` does: the first run, then three times the
rows. `layout.py` takes each published key's ordered fingerprint (row count and two sums
of position-seeded row hashes, as parity), its schema hash, and each file's bytes, sha256
and rows per row group. Master was cec1474 (#137).

| | first run | 3× |
|---|---|---|
| keys | 2,324 | 2,324 |
| ordered fingerprint differs | 0 | 0 |
| schema differs | 0 | 0 |
| keys byte-identical to master | 2,323 | 2,323 |
| files | 2,324 → 2,324 | 2,327 → 2,327 |
| claims/all part files | 1 → 1 | 4 → 4 |
| claims/all row groups | 8 → 10 | 24 → 30 |
| claims/all bytes | 75.61 → 75.67 MB (+0.08%) | 112.38 → 112.67 MB (+0.27%) |

Rows per row group:
- **Before:** 158,708, from c0's first row group (846 B a row), the same for every bucket.
- **After, first run:** 125,469 and 124,300 in the two buckets.
- **After, 3×:** 134,751, 117,361, 103,879 and 154,699. Each bucket's own, as its rows
  vary from 865 B a row (the main set's) to 1,350 B (the scholarly set's).

**Independence of upstream encoding.** Every group file was rewritten by pyarrow, with
dictionaries, snappy and 50,000-row row groups: the same rows, 63% more bytes.

| | master | this branch |
|---|---|---|
| published files the same as from the original group files | 2,323 of 2,324 keys: claims/all changed | all 2,324 keys, byte for byte, both runs |
| what changed | claims' row groups 158,708 → 153,216 rows; at 3×, buckets and part files 4 → 6 (the re-encoded group files are 63% larger) | nothing |

`scripts/test_sort.py` and `scripts/test_finalise.py` pass. `test_sort.py` also checks
these:
- the plan is the same for its sample re-encoded, reordered and glued;
- `decoded_bytes` is the same reordered, rechunked and read by pq-glue;
- the part files are the plan's packs of even rows;
- each bucket's row groups are sized from its own rows.

**Buckets cut between cells.** With `_boundaries` cutting only between cells:
- on the bench's first run the boundaries are the same as cutting between ids (587,556
  rows a bucket on average, 587,707 the largest);
- at 3×, the largest of the 4 buckets is 882,834 rows against a mean of 881,334
  (+0.17%, against +0.03% cut between ids);
- at a release (modelled), a 9-digit cell is roughly 5–20k claims rows against a
  bucket's ~1.2–1.5M, so a cut moves by at most about 1% of a bucket.

Content identity holds with cells: against master cec1474, all 2,324 keys have the same
ordered fingerprint and schema in both runs. Only claims/all changes: its row groups
8 → 10 and 24 → 30, and its bytes +0.08% and +0.27%. The re-encoded group files still
publish byte-identical files for every key.

**#140's group files, end to end.** `pipe_driver.py` runs the bench's 4 chunks through
`main.process_and_partition`, one group closed by `merge_group`, and finalise as the
bench runs it. The split's reshape is stubbed, so the claims are decoded from the chunks'
JSON. It ran this branch's code before #140 (on cec1474) and after it (79438b1):
- 2,322 of the 2,324 group files differ in bytes; only claims/all and entities/all
  don't;
- all 2,324 published keys are byte-identical, in both runs.

Master before and after #140 was byte-stable on these chunks too, because no key there
crosses a threshold. The re-encoding test above is the general case.

## The one-time boundary change at a release (modelled)

This model uses release 20260928's compacted claims:
- the main set: 44.7 GB in 30 files, 667 buckets, 86 part files;
- the scholarly set: 49.5 GB.

It assumes the sample's estimated decoded bytes per group-file byte (21.35 for the main
set's claims, 37.54 for the scholarly set's). For the language tables it takes the
20260507 build's rows per key (`docs/dataset_cards_metadata.json`), at the sample's
estimated bytes per row. These are model figures, not measured ones.

| table | before | after |
|---|---|---|
| claims, main set: part files | 86 (about 540 MB each) | 88 (about 520 MB each) |
| claims, main set: buckets | 667 | 508 (rows a bucket +31%) |
| claims, scholarly set: part files | 95 | 96 (about 530 MB each) |
| claims, scholarly set: buckets | 738 | 989 (rows a bucket −25%) |
| claims: rows a row group | one figure for the key, from its largest file's first row group | each bucket's own: about 155k (main set) and 100k (scholarly set), the count about the same |
| entities (each set): part files | 3 | 3 |
| labels/en: part files | 2 | 2 |
| every other key | 1 file | 1 file |
| other keys: rows a row group | the largest group file's first row group | the whole key's mean; most keys are under one row group either way |
| sorted through buckets | claims | claims; each set's entities are in memory (7.3 and 4.5 GiB estimated) |

**Bucket memory evens out between the sets.** Today the scholarly set's buckets are the
heaviest: 64 MiB of source is about 1.3M rows at about 1,350 decoded bytes each, 1.76 GB.
The main set's are 0.83 GB. With buckets by estimated decoded bytes, both are about
1.1–1.3 GB, so the largest bucket at a release gets smaller.

`_bucket_memory` now estimates from the plan's bytes per row, recalibrated to 1.6 per
byte. For the bench's 3× bucket it gives the same 2.69 GB as before.

## Bench

- Parity already compares the published files logically: each key's files read in name
  order (an ordered row-hash fingerprint), plus the schema's hash. It needs no change.
- The bench's scaling of `sort_by_id.SORT_IN_MEMORY_BYTES` (by the sample's share of
  entities) and of `COMPACT_FILE_BYTES` (the 3× claims target) applies unchanged.
- The bench's shape:
  - claims keep 2 and 4 buckets, and 1 and 4 part files;
  - on the bench, entities/all (one row per entity: 3.45 MB estimated) now goes through
    one bucket;
  - labels/en and labels/nl are now sorted in memory. Their strings are heavier in Arrow
    than in Parquet, but they are small by decoded bytes.
  - the larger-keys pool has 78 keys, against 81.
- Locally, under shared load, finalise's first run took 50.6 s and 115.4 CPU-s on master,
  against 57.2 s and 116.9 CPU-s on this branch. The same branch run ranged from 54 s to
  122 s under other agents' load. CI decides.

**CI** (bench 13.0.0, quick tier, 3 rounds, 79438b1 → 7171dc5): parity passed.

| | before | after | change |
|---|--:|--:|---|
| measured end to end: CPU (split + pipeline + finalise) | 1m59s | 1m52s | −5.1% ±2.7% |
| measured end to end: wall | 43.74 s | 41.08 s | −5.9% ±4.0% |
| finalise: sort | 30.82 s | 27.87 s | −9.5% ±3.5% |
| finalise, 3× the rows | 41.42 s | 38.45 s | −7.2% ±1.5% |
| CPU, finalise | 1m20s | 1m13s | −9.1% ±3.6% |
| peak RSS, finalise, all processes | 4.195 GB | 3.832 GB | −8.6% ±3.5% |
| peak RSS, finalise 3×, all processes | 9.201 GB | 8.541 GB | −7.2% ±1.1% |
| peak RSS at a release, modelled: all processes | 20.61 GB | 20.20 GB | −2.0% ±0.7% |
| ETA, a release | 7h16m41s | 7h20m02s | within noise |
| ETA: finalise | 2h11m51s | 2h08m33s | within noise |
| disk, peak rise | 608 MB | 608 MB | = |
| `_read_key` (per byte) | 1.3 s | 0.7 s | −42.7% ±4.9%: no sample row group read |
| `compact._write_file_parallel` (per byte) | 12.0 s | 12.4 s | +2.7% ±2.2%: smaller claims row groups |

Most of finalise's gain is the bench's shape, not a release's:
- labels/en and labels/nl leave the bucket path. They were bucketed only because the
  bench scales its threshold by entities. In their place, entities/all takes one
  bucket.
- The larger-keys pools change: descriptions 40 → 62, labels 20 → 7, links 15 → 7,
  aliases 5 → 2.

At a release the in-memory and bucketed keys are as before (the claims alone through
buckets), so this is an enabler, neutral there. The `_read_key` saving holds at a
release, but it is small.

## Next

- **Stage 2 is dropped.** #106's one partition file per table per chunk is superseded by
  #140 (partitioned once per group, at its close) and fusion A (the split writing these
  tables). Under this change both publish the same files whatever their group files'
  encoding.
- **Stage 3 (E5): claims bucketed as the split writes them.** The split's claims writer
  reorders each part's rows stably by cell, and records each row group's rows by cell
  beside the file (`{stem}.cells.json`). The bucketing then reads no id to plan: the
  boundaries come from the summed cells, the same as `_boundaries`' by construction. A
  row group whose buckets all read it directly is not read at all. Prototyped on the
  stacked branch sort/cells-sidecar: on test_sort's sample, the published files are
  byte-identical, and rows spilled fall 92,705 → 1,638 (row groups spilling 54 → 12 of
  176).
- **Possible refinement:** factors by set for the language tables. The scholarly set's
  labels are longer titles, but its keys are single files.
