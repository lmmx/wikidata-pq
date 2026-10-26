# 2026-10-26: Smaller group files without more CPU: encodings, metadata, page size (negative)

At a release about 124 GB of group files go up to the build branch, and finalise
downloads them again. #191 (`2026-10-26-xet-dedup.md`) measured that they dedup about 0%
across releases. #187 (`2026-10-26-upload-throughput.md`) found the uplink is most likely
about 7 MB/s. At that rate the release is network-bound, so every group-file byte is on
the critical path.

**The hypothesis:** the group files are written for speed and read once, by finalise.
So they could be made markedly smaller, and at 7 MB/s that would outweigh the CPU.

**What was already settled:** #174 (`2026-10-26-link-bytes.md`) swept zstd levels for
these files and rejected them:
- the claims are three quarters of the bytes and barely compress further (−3.5% at level
  9, for 7× the compress CPU);
- the extra CPU lands in the split, which is the critical path from 15 MB/s up.

That sweep is not repeated here. This entry tests what #174 did not:
- the other encodings, and a dictionary on or off, per column;
- the statistics and page indexes;
- page size;
- columns finalise could recompute.

All of it is on top of #184 (`statement_id` in DELTA_BYTE_ARRAY, merged as 8f7720b).

The scripts and their outputs are in `2026-10-26-group-file-bytes/`.

## Current State

- **Negative. Nothing in `src/`, `reshape/` or `bench/` changed.** No change open to the
  group files' writers saves enough bytes, without costing CPU, to show beyond the
  bench's noise.
- **Per-column encodings:** −1.7% of the claims' group bytes, and less on read.
  - The best encoding per column: `id` in DELTA_LENGTH_BYTE_ARRAY, `datavalue.numeric-id`
    in BYTE_STREAM_SPLIT, and so on (below).
  - Write CPU is the same.
  - Read CPU is +7% by pq-glue (the sort's reader) and +18% by pyarrow. Finalise's claims
    sort is on the path at 7 MB/s, so this cost lands there too.
  - At a release that is about 1.6 GB less: about −4 min at 7 MB/s, and ~0 at 60 MB/s.
    The bench's interval on the 7 MB/s modelled wall is about ±1.2% (±8 min, #184). So
    it would not show, and the read CPU takes back part of it. The no-regret subset
    (only the columns whose read CPU doesn't rise) is about −1%.
- **Statistics and page indexes are under 1% of a release-shaped claims group file:**
  - the footer is 0.66%, the page index 0.28%;
  - the min/max values are about a quarter of each row group's ~15 KB of footer.

  Dropping everything but `id`'s chunk statistics (`file_stats` and the sort read those)
  saves at most ~0.45%. pq-glue's writer has no option for it, so it would need a pq-glue
  change and release first. Not worth it at ~1 min at 7 MB/s.
- **Page size: nothing.** With 1, 4 and 16 MiB pages, the claims come within 0.01% of
  each other (pyarrow; neither pq-glue's writer nor genson's exposes a page size).
- **Row-group size: nothing open.**
  - At a release, a chunk's claims fit one row group, and the group merge copies them
    as encoded (`pq_glue.concat`). So a larger row group would only save footer entries,
    which are within the 0.66% above.
  - Re-encoding the claims in the merge would cost CPU on the path, as #174 found for
    levels.
- **The bench already sees group-file bytes.** No harness change was needed:
  - the 7/30/60/120 MB/s modelled walls take the overlap stage's uploaded files' real
    sizes (`overlap._uploaded`) and finalise's downloads of the build branch
    (`finalise_link_bytes["group"]`);
  - both are scaled by the bench data's chunk files, which no group-file writer
    changes.
- **Two leads, each outside the group files' writers:**
  - **Columns finalise could recompute (−6.3% of the claims' column bytes, ~−15 min at
    7 MB/s before its CPU).** Every `numeric-id` and `entity-type` in the claims (the
    main snak's, the qualifiers' and the references') is exactly a function of its `id`
    on the sample (2.4M values, no exception, `derive.py`, `derive_refs.py`). `numeric-id`
    is 5.0% of the claims' bytes and `entity-type` 1.3%. Leaving them out of the group
    files means finalise must rebuild them before its sort, bit for bit:
    - that is a change to the published path's reads (`compact`, `sort_by_id`) and to
      the release schema's file checks;
    - it adds CPU to finalise's claims sort, which is on the path at 7 MB/s.

    Not attempted here.
  - **The scholarly descriptions with a dictionary (−20%, and less write CPU).** The
    fused writer (`reshape/src/partition.rs`, owned by #185) writes the split tables'
    `value` in PLAIN pages. In a large key's row groups, pq-glue's adaptive dictionary
    makes the scholarly set's descriptions 20% smaller, at 55% of the write CPU
    (`fused2.py`). That set's descriptions are 9.2 GB of group bytes at a release, so
    ~1.8 GB, ~−5 min at 7 MB/s. The other tables gain 0–3%.
    - But `_copy_built` merges a key's runs from every chunk under one dictionary page
      (`rowgroups.merge_row_groups`). A dictionary per chunk would push those keys onto
      its row-group-at-a-time fallback, with a footer entry per chunk: the quarter of the
      group bytes that merge removed (`2026-10-25-group-row-groups.md`).
    - So it needs a dictionary shared across a group's chunks, or a merge that re-maps
      the indices. That work belongs with the fused writer's owner.

## The sample

- **The data:** the quick tier's chunk files (c0, r50, s30, s90; bench data e8c97e1).
- **The split:** run by `bench/pipeline.split_stage` (`split.py`), on master with #184.
  Its claims partition files (5 files, 1,175,112 rows, 71.18 MB) are what
  `merge_group` glues into the claims group files.
- **The rewrites:** each file was rewritten as the split writes it,
  `pq_glue.write_parquet`:
  - with genson's settings, it writes byte-identical files to polars-genson's
    `write_parquet` (`write_table`), so the two are interchangeable here;
  - row groups of 122,880 rows, zstd 1, only `column_encoding` varied.
- **The figures:**
  - **CPU:** the process's own, the best of 3 to 5 rounds.
  - **Bytes:** exact.
  - **Values:** every rewrite reads back equal by pyarrow and Polars.
- **The machine:** the shared 4-core machine, idle otherwise. Indicative only.

## Encodings, per column (`encsweep.py`, `percol.py`, `chosen.py`)

**The sweep (`encsweep.py`):** every leaf in each family at once, then the best per leaf:
- byte arrays in DELTA_BYTE_ARRAY, DELTA_LENGTH_BYTE_ARRAY or PLAIN (no dictionary);
- integers in DELTA_BINARY_PACKED, BYTE_STREAM_SPLIT or PLAIN;
- doubles in BYTE_STREAM_SPLIT, ALP or PLAIN.

ALP is left out: it is not in the Parquet specification pyarrow and Polars read.

| family, every leaf at once | bytes | vs master | write CPU-s |
|---|---|---|---|
| master (#184: `statement_id` DBA, the rest adaptive dictionary) | 71,180,623 | — | 5.12 |
| byte arrays DBA, ints DELTA_BINARY_PACKED, doubles BSS | 79,688,388 | +12.0% | 5.31 |
| byte arrays DLBA, ints BSS, doubles ALP | 76,650,038 | +7.7% | 4.77 |
| PLAIN, no dictionary | 76,037,769 | +6.8% | 4.86 |
| **the best per leaf** | **69,921,103** | **−1.77%** | |

The dictionary (adaptive) is right for most leaves. Where it isn't, each column's change
alone, with its read CPU against master's:

| column | encoding | bytes | vs the file | read CPU-s, pq-glue | read CPU-s, pyarrow |
|---|---|---|---|---|---|
| `datavalue.numeric-id` | BYTE_STREAM_SPLIT | −314,399 | −0.44% | +0.04 | +0.45 |
| `datavalue.id` | DELTA_BYTE_ARRAY | −292,497 | −0.41% | +0.17 | +0.55 |
| `datavalue.datavalue__string` | DELTA_LENGTH_BYTE_ARRAY | −261,844 | −0.37% | +0.03 | +0.13 |
| `references.list.element.hash` | DELTA_LENGTH_BYTE_ARRAY | −133,676 | −0.19% | +0.14 | +0.15 |
| `id` | DELTA_LENGTH_BYTE_ARRAY | −126,770 | −0.18% | +0.06 | +0.23 |
| `id` | PLAIN | −119,714 | −0.17% | −0.07 | +0.03 |
| `datavalue.text` | DELTA_LENGTH_BYTE_ARRAY | −35,466 | −0.05% | −0.08 | +0.19 |

- **Master's read:** 2.18–2.21 CPU-s by pq-glue, 2.69–2.89 by pyarrow, across its two
  runs. So the pyarrow column is within about ±0.2 of noise.
- **`id`'s dictionary is worse than none:** its distinct ids make a dictionary page per
  row group about as large as the column. Without one, zstd takes each run of a repeated
  id almost for free.
- **All of them at once (`chosen.py`; the table's ten columns, the qualifiers' two
  included):**

| | bytes | main set | scholarly set | write CPU-s | read CPU-s: pq-glue / pyarrow / Polars |
|---|---|---|---|---|---|
| master | 71,180,623 | 45,740,575 | 25,440,048 | 4.94 | 2.11 / 2.89 / 5.75 |
| chosen | 69,952,076 | 44,990,114 | 24,961,962 | 5.03 | 2.24 / 3.40 / 6.54 |
| change | **−1.73%** | −1.64% | −1.88% | +2% | **+7% / +18% / +14%** |

- **Not consistent across the sets:**
  - `references.list.element.hash` in DLBA: −13% in the scholarly set, +1% in the main;
  - the qualifiers' two columns, and `datavalue.amount`: −7 to −8% in the main set, +6
    to +9% in the scholarly.

  A release's chunks are smaller than c0 (a row group each, ~45–70k rows), so the
  per-column picks would need checking at that shape too.
- **At a release:**
  - the claims' group files are ~90 GB (92.6 GB before #184's −2.7%), so −1.73% is
    ~1.56 GB: −3.7 min uploading at 7 MB/s and −21 s downloading at 75 MB/s;
  - at 60 MB/s the build branch's uploads are not the path, so ~0;
  - against that, finalise's claims reads take +7% (pq-glue) to +18% (pyarrow) of
    their CPU, on finalise's path at 7 MB/s.

## Statistics and page indexes (`meta.py`, `anatomy.py`)

- **The file:** a release-shaped claims group file, 40 copies of the sample's
  release-sized chunk (`main/claims/chunk_1`: 45,546 rows, one row group, 2.32 MB),
  glued by `pq_glue.concat_many` as `merge_group` glues them.
- **What it holds:**
  - 92.21 MB in all, 40 row groups;
  - column chunks 91.34 MB;
  - the footer 610 KB (0.66%), 14.9 KB a row group, plus the file's own 7.9 KB of
    Arrow schema;
  - the page index 259 KB (0.28%).
- **The statistics:** min/max values are at most ~4 KB of each row group's 14.9 KB, even
  counted in both their legacy and current fields. The rest is the 93 column chunks'
  own metadata (paths, offsets, sizes, encodings), which every reader needs.
- **What finalise reads:** `id`'s chunk statistics (`partitioning.core.file_stats`, the
  sort's plans).
- **So the ceiling** for dropping the other statistics and the page index is ~0.45% of
  the claims' group bytes, ~0.4 GB at a release, ~1 min at 7 MB/s.

## Page size (`pagesize.py`)

pyarrow, zstd 1, the same encodings, data and dictionary page limits of 1, 4 and 16 MiB:
- 74,389,656, 74,381,732 and 74,381,732 bytes;
- −0.01% from 1 MiB, the default of both arrow-rs and pyarrow.

So larger pages give zstd nothing more to work with on the claims.

## Recomputable columns (`derive.py`, `derive_refs.py`)

Checked on every entity value in the sample's claims: 545,688 main snaks, 100,327
qualifier values and 777,676 reference values.
- **`numeric-id`** is `int(id[1:])` where `id` is `[QPLM]<digits>`, and null otherwise.
  The forms, senses and entity schemas have none.
- **`entity-type`** is `id`'s prefix:
  - `Q` item, `P` property, `L` lexeme;
  - `L…-F…` form, `L…-S…` sense;
  - `E` entity-schema.
- No value differs. The bytes are from `chosen_l1.json`: `numeric-id` 3.57 MB (5.04%),
  `entity-type` 0.91 MB (1.29%).

## The split tables' encodings (`fused2.py`)

- **The rows:** each set's rows of a table, all chunks', ordered by key (stable),
  written as a large key's group file holds them (row groups of 122,880 rows, zstd 3).
- **Compared:** PLAIN (the fused writer's encoding for every column but the key)
  against pq-glue's adaptive dictionary.

| table, set | rows | PLAIN bytes | dictionary | write CPU-s | read CPU-s (pyarrow) |
|---|---|---|---|---|---|
| descriptions, scholarly | 978,044 | 2,352,576 | **−20.0%** | 0.24 → 0.13 | 0.10 → 0.08 |
| descriptions, main | 371,582 | 3,420,160 | −1.1% | 0.13 → 0.13 | 0.05 → 0.07 |
| labels, main | 577,282 | 4,928,391 | −2.0% | 0.19 → 0.18 | 0.06 → 0.06 |
| links, main | 421,933 | 4,037,391 | −2.7% | 0.16 → 0.12 | 0.06 → 0.06 |
| labels, scholarly; aliases | | | 0 to +1.8% | | |

- **The sample's keys are far smaller than a release's,** so its built files' row
  groups (a key's run per chunk) are not the shape a release's group files have. This
  is the large-key shape only.
- **The change is the fused writer's** (`reshape/`), not open here. It also has to keep
  `_copy_built`'s one dictionary page per merged row group: see Current State.

## Missing

- **Per-column encodings at a release's chunk shape** (one row group of ~45–70k rows a
  chunk), and each set's own best map. Only worth it with the recomputable columns or
  the split tables' dictionary, together enough to show at 7 MB/s.
- **Finalise's CPU to rebuild `numeric-id` and `entity-type`** for ~1.8 billion claims
  rows, and whether the release schema's file checks can take a group file without
  them.
- **A dictionary shared across a group's chunks** for the fused writer's `value`
  columns, or a row-group merge that re-maps dictionary indices.
