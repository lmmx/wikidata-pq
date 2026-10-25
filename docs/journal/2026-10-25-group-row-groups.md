# 2026-10-25: The group files' row groups: #140's +3.2%, and a key's row groups merged at the close

This entry follows up two results from master's CI ledger:
- #140 (79438b1) showed measured CPU +3.2% ±0.3;
- #148 (0a574cf) showed disk_peak +1.9% and finalise_cpu +1.8% ±2.1.

The hypothesis was that the group files' row-group layout costs finalise and disk. The
entry also models what #148's layout, one row group per key per chunk, costs at a release.

## Current State

- **#140's +3.2% wasn't the layout.** It was 3 more worker processes in finalise, from a
  sort threshold on encoded bytes, plus noise in the split, whose code #140 didn't touch.
  #147 removed the threshold, and its ledger run took the workers back out (below).
- **#148's layout costs nothing on the bench, but a lot at a release.**
  - **On the bench:** a group has 4-5 chunks, so a key has at most 5 row groups.
  - **At a release:** a group has about 400 chunks. That gives about 11.4M row groups
    over the four tables' group files, with a median of 2-35 rows each.
  - **The close's copy is Python:** it reads and writes each chunk's footer as thrift,
    at ~360 µs of CPU a row group. That's about 68 CPU-min at a release, where #140's
    sink was modelled at ~15.
  - **The footers take a quarter of the bytes:** about 245 bytes a row group, which is
    ~2.8 GB of the four tables' group files at a release.
  - **Finalise pays per row group too:** each one is a read of its own.
- **The change (this PR):** the close merges a key's consecutive row groups into one, of
  up to 122,880 rows (`ARROW_ROW_GROUP_ROWS`, the split writer's own cap). It does this
  at the column-chunk level (`rowgroups.merge_row_groups`).
  - Pages are copied as bytes: nothing is decoded or encoded.
  - Each column's data pages are laid end to end under the first source's dictionary
    page. Only the key column has a dictionary, its one value, and every source's must
    be the same bytes. That is checked.
  - Values and sizes are summed; min and max are bounded.
  - The footer is read by a one-pass reader of just the split writer's fields
    (`ChunkFooter`). Any other field raises `Unsupported`, and the close then copies the
    row groups one by one, as before (`copy_row_groups`, kept).
- **Measured locally** (one main-set group of 400 chunks: the bench's five routed chunks,
  cycled):
  - close CPU: 122.2 s → 23.9 s (÷5.1);
  - group files: 339.3 MB → 255.9 MB (−24.6%);
  - row groups: 340,480 → 1,336.
- **The published files are byte-identical:** 2,326 of 2,326 keys, sorted from two
  groups' files made both ways. The group files read the same in pyarrow, pq_glue and
  Polars, with the same Arrow schema and the same `footer_size`, so the same sort plan.
- **Tests pass:**
  - `test_fused_close`, extended (below);
  - `test_group_restart` at all 14 kill points;
  - `test_merge`, `test_sidecar` and `test_partitioning`.

## #140's +3.2%: what it was

The ledger run 40218127750 compared 79438b1 with 2954123 over 3 rounds × 2 runners. Means
of the 6 runs, CPU-s:

| | before | after | change |
|---|--:|--:|--:|
| measured CPU (split + pipeline + finalise) | 120.23 | 124.12 | +3.89 (+3.2%) |
| split (`split_cpu_total`) | 31.37 | 33.17 | +1.80 (+5.5%) |
| pipeline (`cpu_total`) | 8.35 | 7.90 | −0.45 (−5.4%) |
| finalise (`finalise_cpu`) | 80.52 | 83.05 | +2.53 (+3.2%) |
| finalise: traced functions' own CPU (phases) | 22.95 | 22.67 | −0.28 ±0.40 |
| finalise: `_read_key` | 1.59 | 1.24 | −0.35 (−22%) |
| finalise: processes | 99 | 102-103 | +3 |

- **The split, +1.80 CPU-s:** #140 changed no split code (`git diff --stat 2954123
  79438b1`: nothing under `dump.py`, `reshape/` or `claims_build.py`). Its per-run
  changes ran from −0.4% to +18.8%, with an interval of ±10%. It was noise.
- **Finalise, +2.53 CPU-s, was 3 more worker processes.**
  - The traced functions fell by 0.28 s. The group files' one row group a key made the
    reads cheaper (`_read_key` −22%).
  - What rose was CPU outside the traced functions: 99 → 102 processes in every run.
  - The logs show why. The "larger keys" pool (a key over `SORT_IN_MEMORY_BYTES // 8` of
    its files' encoded bytes, a worker each) went from 41 descriptions keys to 45, and
    from 16 links keys to 15.
  - #140's sink re-encoded the group files, so their sizes changed, and 4 keys crossed
    the threshold. Each worker costs about 0.66 CPU-s for the interpreter and the
    imports, so ~2.0 s in all.
- **Why ±0.3:** the bench pairs each round's forward and reverse runs.
  - The 3 rounds' means of measured CPU came out 3.15%, 3.25% and 3.35%, three nearly
    equal numbers from three samples.
  - Its components' round means were 5.45%, 9.75% and 1.4% for the split, and 3.1%,
    1.45% and 5.05% for finalise.
  - The tight interval was chance, not a measurement of a +3% effect.
- **#147 confirms it** (ledger run 40221195177: 5df0c40, "planned from the data alone,
  not from encoded sizes").
  - Finalise's processes went from 102 to 92-94.
  - finalise_cpu fell 8.9% and measured CPU 4.8%.
  - The workers #140's bytes had added left with the threshold.
  - #147's release ETA of +1.6% ±1.4 comes from its split, +5.2% ±6.1, which #147 also
    didn't touch.
- **#148** (ledger run 40223102466):
  - finalise_cpu +1.8% ±2.1, with the traced finalise functions +0.5% ±2.7: n.s.;
  - processes 92-94 in both arms;
  - disk_peak +10 MB (530 → 540).
- **The disk peak isn't the layout.** The quick tier's group files are 26.55 MB copied,
  26.27 MB merged, and 27.15 MB if Polars encodes the same rows (as #140's sink did).
  #148's group files are smaller than #140's, so its +10 MB comes from elsewhere, not
  from the group files.

**A pattern worth the bench owner's look:** the split's CPU rose in all three of these
ledger runs (+5.5%, +5.2% and +2.6%), and none of the three changed the split. In each
run it rose more in the forward runs (before run first) than in the reverse.

## The release, modelled from real counts

**Row groups.** A key's group file has a row group per chunk in which the key has a row.
The counts come from the bench's chunks, built as the split builds them (one file per
table, a row group per key). Mean keys a chunk:

| table | main (r10-r90) | scholarly (s10-s90) | row groups at a release | rows a row group: median, p90 |
|---|--:|--:|--:|--:|
| labels | 296 | 26 | 3.90M | 7, 71 |
| descriptions | 233 | 66 | 3.55M | 35, 279 |
| aliases | 247 | 4 | 3.05M | 3, 20 |
| links | 75 | 2 | 0.94M | 2, 21 |
| all four | | | **11.44M** | |

- The release has 12,182 main and 10,853 scholarly chunks.
- The 11.44M agrees with the ~11.5M partition files that earlier releases wrote, one a
  (chunk, key).
- **A main-set group** is ~400 chunks (30 groups a set). The sample's group of 400 chunks
  (the five routed chunks cycled) gives each key file this many row groups:
  - labels and descriptions: a median of 400, one a chunk;
  - aliases and links: a median of 160.

**Footer and file bytes:**
- **Per row group:**
  - the footer entry is 200-280 bytes;
  - each column's tiny zstd page and its header add more;
  - the key column's dictionary page is repeated in every row group.
- **The footers' share of the group files' bytes** in the 400-chunk group: labels 17%,
  descriptions 17%, aliases 41%, links 36%.
- **Merged**, the group's four tables come to 339.3 MB → 255.9 MB, 245 bytes less a row
  group.
  - **At a release that is about 2.8 GB** of the four tables' group files, ~19 GB
    copied. It is uploaded to the build branch, hashed by verify_group, downloaded by
    finalise unless kept, and hashed again by `check_compacted_copy`.
  - Re-encoding the rows would save more, since it also merges the pages. A
    pyarrow-written group of five distinct chunks saves 280-350 bytes a row group. But
    re-encoding costs a decode and an encode of every row (below).

**The close's CPU,** for each of the four tables, on that 400-chunk group, with the real
`_copy_built` (8 threads):

| | labels | descriptions | aliases | links | all | per row group |
|---|--:|--:|--:|--:|--:|--:|
| copied a row group at a time (master) | 41.4 s | 34.0 s | 35.9 s | 10.8 s | 122.2 s | 359 µs |
| merged (this PR) | 8.2 s | 6.3 s | 7.1 s | 2.3 s | 23.9 s | 70 µs |

- **Where the copy's time went** (cProfile, labels):
  - the footer's thrift read into nested lists (`Footer`): 33 s;
  - writing it back (`_put`): 15 s;
  - moving each row group's offsets (`_moved`): 5 s;
  - copy_file_range: little of it.
- **At a release:** 11.44M row groups × 359 µs ≈ **68 CPU-min** of the close, against
  ≈ **13** merged. The close runs beside the chunk workers on the same cores.
- For comparison, #140's sink was modelled at ~15 CPU-min: 0.27 µs a row and ~1 ms a
  file. #148's copy is cheaper on the bench only because its groups have 4-5 chunks.

**Finalise's per-row-group costs,** on the same group's files, CPU-s over each table's key
files:

| | row groups | `footer_size` | `read_metadata` | sha256 | `_read_key`, pyarrow | `_read_key`, pq_glue |
|---|--:|--:|--:|--:|--:|--:|
| copied | 340,480 | 1.25 | 0.61 | 0.33 | 32.7 | 45.8 |
| merged | 1,336 | 0.06 | 0.05 | 0.25 | 22.1 | 3.4 |

- **What each column is:**
  - `footer_size`: the sort's plan, run at compaction's planning and at the sort;
  - `read_metadata`: compaction's link;
  - sha256: the link's entry and `check_compacted_copy`;
  - `_read_key`: the sort's read, by pyarrow on one thread for a key under 16 MB, by
    pq_glue on every core above.
- **pq_glue (arrow-rs) pays per row group:** ~125 µs.
- **pyarrow pays per page as well:** a merged column chunk still has a page per chunk.
  So the pyarrow path keeps two thirds of its time.
- **At a release:** 8-25 CPU-min of finalise, by the mix of the two readers.

## The change

- **`partitioning/rowgroups.py`:**
  - `Unsupported`.
  - `ChunkFooter`, with its `RowGroup` and `Column`:
    - one pass over the footer;
    - integers read inline;
    - the path in schema and the other top-level fields kept as their encoded bytes;
    - every invariant the merge relies on checked: a column chunk's `file_offset` is its
      first page, column chunks are contiguous, and a row group's offset and sizes are
      its columns'.
  - `merge_row_groups(dst, parts, max_rows)`:
    - each part's row groups are read with one pread (the caller may give an open fd);
    - runs of consecutive row groups up to `max_rows` rows are merged;
    - each merged column is the first source's dictionary page plus every source's data
      pages;
    - its metadata: summed values and sizes, the union of encodings, summed null
      counts, min of mins and max of maxes (unsigned bytes, a string column's order),
      and exactness kept;
    - a dictionary page that differs, or another type, path or codec, raises ValueError.
  - `copy_row_groups` and `Footer` are unchanged.
- **`push/core.py`, `_copy_built`:**
  - reads the built files' footers by `ChunkFooter`, and on `Unsupported` falls back to
    `Footer` and `copy_row_groups` for the table;
  - opens each built file once for all its keys (within a quarter of the open-file
    limit);
  - merges each key's file at `GROUP_ROW_GROUP_ROWS` (= `ARROW_ROW_GROUP_ROWS`, 122,880);
  - copies a key whose row groups don't merge (ValueError) a row group at a time instead
    (`_write_key`), each file's tree footer read once.
- **Unchanged:** the rows, their order, the Arrow schema, the sidecars and their sums, the
  checks against the sidecars, and the marker and restart logic.
- **The sort's plan can't move.** `footer_size` sums each column chunk's `num_values`,
  which merging sums.

## Correctness

- **`scripts/test_fused_close.py`** (the bench's c0, r50, s30, s90 as one group), added
  per table:
  - every key merged reads as its row groups copied one by one, in pyarrow
    (`ParquetFile.read` on one thread, and `read_table`), pq_glue and Polars, with the
    same Arrow schema with its metadata, and the same `footer_size`. This holds at the
    cap, at 1,000 rows (runs cut) and at 1 (each row group alone, the built ones
    themselves);
  - the merged row groups are within the cap, or a built one, and as few as it allows;
  - two keys' row groups (other dictionary pages) raise rather than merge;
  - a footer the merge doesn't read is copied a row group at a time by the close, with
    the same rows, and so is a key whose merge raises;
  - the existing checks pass: the same keys, rows and order as the processed group, the
    mixed group, and an interrupted close's rerun byte for byte.
- **The published files,** the claim that carries the contract:
  - two groups (c0, r50, s30, s90 and r10, r30, s10, s50) were made both ways;
  - each key was sorted by `sort_by_id.sort_in_memory` from its two group files;
  - all 567 labels, 443 descriptions, 489 aliases and 827 links keys gave the same part
    files, name, rows and sha256 (`2026-10-25-group-row-groups/published.py`);
  - this covers both of the sort's readers, since some keys are over 16 MB.
- **`scripts/test_group_restart.py`:** all 14 kill points pass. Its synthetic chunks are
  processed, not built, so it covers the close's sink path, not this one. The built
  path's interruption is `test_fused_close`'s.
- **`test_merge`, `test_sidecar` and `test_partitioning`:** pass.

## What the bench can see

- **The quick tier's group has 4-5 chunks**, so a key's row groups go 3,431 → 2,322.
  - Its close gains a little: the four tables' copy, ~0.72 → ~0.54 CPU-s locally.
  - Its group files shrink by 0.28 MB.
- **The full tier** (groups of 22) would show more: up to 22 row groups a key file become
  one.
- **Most of the gain is the release's,** in proportion to chunks a group: the close
  (~−55 CPU-min), the group files (~−2.8 GB) and finalise's reads (−8 to −25 CPU-min).
  The bench models none of these, as it scales the close by entities.

## Rejected

- **Re-encoding each key's rows at the close** (pq_glue or pyarrow reads, pyarrow writes):
  - on the quick tier it cost 1.8-5× the copy (0.46-1.52 CPU-s a table against 0.26-0.37);
  - at a release it would cost a decode and encode of 2.46 billion rows;
  - pyarrow's `ParquetWriter.write_batch` doesn't merge batches into one row group, so it
    would also need the group's rows held per key.
- **The brief's byte or row threshold** (copy row groups of ≥ ~1 MB or 64k rows, re-encode
  the rest):
  - at a release no key has 64k rows in a chunk of ~6,000 entities;
  - so nearly everything would be re-encoded, and the copy kept only for the bench.
- **A pq-glue function for row-group ranges:**
  - it would make the copy native, but keep the 11.4M row groups and their bytes;
  - not needed, as the merge is in Python and already 5× cheaper;
  - no pq-glue release asked for.

## Missing

- **pyarrow's reads of merged column chunks** still pay per page (a page a chunk). Reading
  every key with pq_glue (`sort_by_id._read_key`'s 16 MB threshold) would take the
  pyarrow path's remaining 22 s to about 3 on the 400-chunk group. That is the sort's
  code, not changed here.
- **The disk peak's +10 MB under #148** isn't explained here; the group files aren't it.
- **A bench term for chunks a group,** or the full tier on CI, would show the close's
  per-row-group cost, which the quick tier can't.
