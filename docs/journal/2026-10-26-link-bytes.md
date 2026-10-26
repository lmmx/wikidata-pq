# 2026-10-26: Fewer bytes on the link: zstd levels and encodings (negative)

A full release is network-bound (`2026-10-25-release-critical-path.md`, #167; bench
17.2.0, #168). The hypothesis: when the link is the bottleneck, CPU spent shrinking the
bytes on it is nearly free. So a higher zstd level, or better encodings, for the files
on the link's critical path would cut the release wall by tens of minutes at 7–30 MB/s,
without changing any published value. Those files are:
- the build branch's group files, uploaded by the runs and downloaded again by finalise;
- the published tables finalise writes and commits.

The scripts and their outputs are in `2026-10-26-link-bytes/`.

## Current State

- **Negative. Nothing in `src/`, `reshape/` or `bench/` changed.**
- **No zstd level for either class of file meets the bar:** ≳15 min saved at 7–30
  MB/s, with no meaningful loss at 120 MB/s.
  - **The claims are 73–75% of the bytes on the link, and they barely compress further.**
    From master's level to 9 they shrink 3.5% (group files) and 4.6% (published files).
    Even level 19 saves only 12%, at 50–80 times the compress CPU.
    - 36% of the published claims' bytes are `statement_id`, a UUID per row. The
      references' and qualifiers' hashes (SHA-1 hex) add another 10%.
    - The claims' extra CPU lands on the critical path at every rate. At 7 MB/s, each
      finalise run's claims sort comes before its claims upload starts. At ≥15 MB/s the
      split, where the group files are written, is the path.
  - **The other tables do compress further** (−9 to −25% at level 9: descriptions most,
    then entities, labels, links and aliases), but they are only ~26% of the bytes.
    - **Published, at 9:** −10 min at 7 MB/s, since their compute hides under the
      claims' upload (#170, L1a). But +4 min at 60 MB/s and +8 min at 120.
    - **Group files, at 9:** −10 min at 7 MB/s, but +9.5 min at every rate from 15 MB/s
      up. That extra CPU lands in the split, which is the path once the build branch's
      uploads keep up.
  - **The best combination at 7 MB/s:** the claims at 1, the rest at 9, both classes.
    It saves 23.5 min at 7 MB/s, but costs +2 min at 15 MB/s, +5 at 30, +11 at 60 and
    +14 at 120.
- **The one no-regret change is small:** the published claims at zstd **1** instead of
  3. On the claims, level 1 is both smaller (−0.9%) and cheaper (−39% of compress CPU)
  than level 3, as the claims' group files at 1 already are. Modelled, it saves 2–4 min
  at every rate. #86 (rejected) measured every table at 1 as no gain beyond the bench's
  noise. On this model, claims-only beats #86's every-table form by ~1–2 min below 60
  MB/s, and is about level with it above. Not opened: it is below what the bench resolves, and it costs a release's Xet
  deduplication against the one before (below).
- **The one lead is an encoding, not a level:** `statement_id` as `DELTA_BYTE_ARRAY`.
  Every statement id starts with its row's entity id, so consecutive rows share the
  prefix.
  - Measured with pyarrow: the column goes 27.2 → 23.9 MB in the published claims (−12%,
    −4.3% of the file) and 25.4 → 23.6 MB in the group file (−7%, −2.5%), at no extra
    CPU (less: no length prefixes to compress).
  - **Modelled:** −15.5 min at 7 MB/s, −5 min at 15, −2.8 at 30, ~0 at 120. That passes
    the bar at 7 MB/s only.
  - It needs a per-column encoding option in pq-glue's writer (the published files) and
    in polars-genson's (the claims' group files). Neither has one today. So it is a
    change to those two repositories first, and it is left as a lead (Missing).
- **The bench would see a level change as it is.** Bench 17.2.0's release wall takes:
  - its bytes from the files the commit writes (the overlap stage's uploads and
    `finalise_link_bytes` from the stand-in Hub's branches);
  - its CPU from the measured ETAs.

  So no harness change is needed for this class of change. The bench has no 15 MB/s
  row.

## Inventory: the bytes on the link at a release

Release bytes are bench 17.2.0's model on master 718e028 (run 40259387706): the sample's
files scaled by each set's chunk files' bytes, as `finalise_link` and
`upload_eta_modelled` scale them. Sample bytes are this run's quick tier, kept with
`KEEP=1 bench/pipeline.py` (data e8c97e1, files c0 r50 s30 s90).

| file | direction | at a release | compression at master | where it is set |
|---|---|---|---|---|
| the dump's bz2 | in, from Wikimedia | 103.5 GB | bz2 (Wikimedia's) | not ours; off the path (the split is longer) |
| chunk files | local only | — | zstd 3 | `dump.ZSTD_LEVEL` |
| **group files: claims** | **up** (build branch), **and down** in finalise | 92.6 GB (main 40.7, scholarly 51.9) | **zstd 1**, arrow-rs | polars-genson's `write_parquet` (no level parameter), from `partitioning.core.write_table` in the split's claims build; pages copied by `merge_group`'s glue |
| group files: labels, descriptions, aliases, links | up, and down | 10.1, 13.2, 2.3, 5.4 GB | zstd 3, PLAIN pages | the fused writer, `reshape/src/partition.rs` `ZSTD_LEVEL`; pages copied by `_copy_built`. Unfused, Polars' sink at its default 3 (`push/core._sink_group`, L5's file) |
| group files: entities | up, and down | 0.8 GB | zstd 3 | Polars' sink (`entities_build`) |
| group files, all | up 124.3 GB; down 124.3 GB at 75 MB/s (28 min), unless `WIKIDATA_KEEP_GROUP_FILES` | | | |
| **published: claims** | **up** | 94.2 GB (main 41.4, scholarly 52.8) | **zstd 3**, CDC, page index | `compact._write_file_parallel` (pq-glue, parallel) |
| published: labels, descriptions, aliases, links, entities, claims_labels | up | 10.1, 11.4, 2.4, 5.7, 0.8, 4.4 GB | zstd 3, CDC, page index | the same (`sort_by_id.write_parts`, `sort_bucket`; claims_labels through `write_parts`) |
| published, all | up 128.9 GB | | | |
| cards, metadata | up | KB | — | — |

- **The claims are 74.5% of the group bytes and 73% of the published.**
- **On the sample,** the group files are 100.1 MB and the published files 106.4 MB.
- **None of the levels is in `push/core.py`** but the unfused sink's (Polars' default).
  The group files' bytes are the split's own pages, copied.

## Measured: size against CPU, on the sample

`pages.py` reads every page of every file, decompresses its body, and compresses it again
at each level, one page a call, as the writer that wrote it compresses that page.
- **Fixed:** encoding, dictionaries and page boundaries, content-defined chunking's
  included. So only the compression changes, and the size is exact: at the writer's own
  level it reproduces each file's bytes to the byte.
- **The CPU is the compression's alone,** on one thread, the median of 3 rounds. That is
  the part of a write that a level changes.
- **The check:** `writer.py` rewrote the published files whole with pq-glue's writer at
  each level. Its whole-write CPU moved by the same amount: the claims at 3 → 9, +3.3
  CPU-s against `pages.py`'s +3.4.
- **The machine:** the shared 4-core machine, no other job running. Indicative only.

**Group files** (master: claims at 1, the rest at 3):

| level | claims MB | vs master | compress CPU-s | decompress CPU-s | the rest MB | vs master | compress CPU-s | decompress CPU-s |
|---|---|---|---|---|---|---|---|---|
| 1 | 73.11 | — | 0.68 | 0.23 | 27.71 | +2.7% | 0.62 | 0.39 |
| 3 | 73.76 | +0.9% | 1.10 | 0.24 | 26.98 | — | 0.71 | 0.35 |
| 6 | 71.65 | −2.0% | 2.99 | 0.23 | 24.62 | −8.8% | 1.57 | 0.35 |
| 9 | 70.54 | −3.5% | 4.69 | 0.25 | 23.11 | −14.4% | 2.60 | 0.34 |
| 12 | 70.11 | −4.1% | 10.72 | 0.24 | 22.75 | −15.7% | 6.42 | 0.34 |
| 15 | 68.78 | −5.9% | 20.05 | 0.26 | 21.37 | −20.8% | 11.48 | 0.34 |
| 19 | 64.64 | −11.6% | 57.37 | 0.25 | 20.66 | −23.4% | 36.96 | 0.36 |

**Published files** (master: every table at 3):

| level | claims MB | vs master | compress CPU-s | decompress CPU-s | the rest MB | vs master | compress CPU-s | decompress CPU-s |
|---|---|---|---|---|---|---|---|---|
| 1 | 74.98 | −0.9% | 0.66 | 0.24 | 31.65 | +3.1% | 0.64 | 0.40 |
| 3 | 75.67 | — | 1.08 | 0.25 | 30.68 | — | 0.78 | 0.43 |
| 6 | 73.30 | −3.1% | 2.86 | 0.26 | 28.80 | −6.2% | 1.60 | 0.40 |
| 9 | 72.21 | −4.6% | 4.51 | 0.24 | 27.29 | −11.1% | 2.48 | 0.42 |
| 12 | 71.80 | −5.1% | 10.44 | 0.23 | 26.94 | −12.2% | 4.13 | 0.41 |
| 15 | 70.43 | −6.9% | 17.15 | 0.26 | 25.49 | −16.9% | 7.77 | 0.41 |
| 19 | 66.69 | −11.9% | 52.21 | 0.25 | 24.69 | −19.5% | 19.62 | 0.44 |

Every level is in `group_levels.jsonl` and `published_levels.jsonl` (1 to 9, 12, 15, 19).

- **By table at 9, published:**
  - descriptions −19.3%, entities −20.0%, labels −9.0%, links −7.4%, claims_labels −6.0%,
    aliases −5.6%, claims −4.6%;
  - the group files are close: descriptions −24.9%, labels −9.7%, links −8.4%.
- **Reading costs the same at every level:**
  - zstd's decompression doesn't depend on the level: the claims' pages decompress in
    0.23–0.29 CPU-s at every level;
  - pyarrow's and Polars' whole reads of the published files were flat across levels
    too, within the machine's noise (`published_writer_levels.jsonl`).

  So neither finalise (it reads the group files) nor the dataset's readers gain or lose
  read CPU from a level.
- **The claims at 1 against 3:** on the claims, zstd 1 is smaller than 3 and 4, and 5
  is barely below 1. `statement_id` is most of that: at 1 it is 25.4 MB, at 3 27.2 MB.
- **Pages at a release are larger** than the sample's (a key's file there has
  hundreds of times the rows). `bysize.py` groups the pages by size:
  - **the claims and labels gain the same at every page size:** the claims 0.95–0.97×
    from 3 to 9, labels 0.90–0.92×;
  - **descriptions gain more on large pages:** 0.73× on pages of ≥128 KiB, against 0.81×
    over all.

  The sensitivity rows below take that, and more.

### The encodings, on `statement_id` (`enc.py`, pyarrow, 1,175,112 rows)

| encoding | published, zstd 3 | zstd 9 | group (zstd 1) |
|---|---|---|---|
| PLAIN (today: its values all distinct, so no dictionary) | 27.22 MB | 26.15 MB | 25.41 MB |
| DELTA_LENGTH_BYTE_ARRAY | 26.72 MB | 25.93 MB | 24.72 MB |
| **DELTA_BYTE_ARRAY** | **23.94 MB** | 23.93 MB | **23.57 MB** |

- Every row's statement id begins with its entity id, so consecutive rows share the
  prefix.
- The write's CPU at zstd 3 was 0.42 CPU-s against PLAIN's 0.47, and the read's 0.30
  against 0.47.
- What is left is about 20 bytes a row, against the UUID's ~16 bytes of entropy.
  Packing the UUID as binary would change the published values, so it is out.

## Modelled: the release's wall at 7, 15, 30, 60 and 120 MB/s

`model.py` runs bench 17.2.0's own model (`overlap.release_wall`, the
`release_wall_link_{r}_modelled` rows) on master's six bench jobs (`master_runs.json`:
718e028, run 40259387706, the "after" arm), with 15 MB/s added.
- **Its inputs are perturbed by the measures above:**
  - each set's and table's group and published bytes by the table's size factor;
  - the build branch's uploads by the group bytes;
  - the extra compress CPU per sample byte, times the release's bytes, over the tier's
    4 workers, as the bench's ETAs take CPU. The group files' goes on `split_eta` (they
    are written in the split), the published files' on `finalise_eta` (the claims' share
    or the rest's).
- **Master's walls:** 12h17m, 8h36m, 7h23m, 6h48m, 6h42m (the six jobs' mean).
- Minutes against master; − is faster. **Every figure here is modelled.**

| case | G, GB | P, GB | split ETA, min | finalise ETA, min | 7 MB/s | 15 MB/s | 30 MB/s | 60 MB/s | 120 MB/s |
|---|---|---|---|---|---|---|---|---|---|
| master | 124.3 | 128.9 | | | 12h17m | 8h36m | 7h23m | 6h48m | 6h42m |
| published: every table at 1 (#86) | 124.3 | 129.1 | | −2.8 | −1.8 | −2.0 | −2.1 | −2.3 | −2.8 |
| **published: claims at 1, the rest at 3** | 124.3 | 128.1 | | −2.2 | **−4.2** | **−3.1** | **−2.6** | **−2.4** | **−2.2** |
| published: every table at 6 | 124.3 | 123.8 | | +13.1 | −3.1 | +3.5 | +6.4 | +10.4 | +13.0 |
| published: every table at 9 | 124.3 | 120.5 | | +25.8 | −2.2 | +8.5 | +13.1 | +21.8 | +25.7 |
| published: claims at 9, the rest at 3 | 124.3 | 124.6 | | +17.8 | +7.5 | +13.0 | +15.4 | +16.9 | +17.8 |
| published: claims at 3, the rest at 9 | 124.3 | 124.8 | | +8.0 | −9.7 | −4.5 | −2.3 | +4.3 | +7.9 |
| published: claims at 1, the rest at 6 | 124.3 | 125.8 | | +1.7 | −9.5 | −5.6 | −3.9 | −0.7 | +1.6 |
| published: claims at 1, the rest at 9 | 124.3 | 124.0 | | +5.9 | −13.9 | −7.7 | −4.9 | +2.1 | +5.7 |
| published: claims at 1, the rest at 12 | 124.3 | 123.6 | | +13.6 | −14.8 | −8.1 | −5.1 | +9.1 | +13.4 |
| published: claims at 3, the rest at 19 | 124.3 | 121.8 | | +91.3 | −17.0 | +2.6 | +54.1 | +86.1 | +90.7 |
| group: claims at 3, the rest at 3 | 125.1 | 128.9 | +2.2 | | +2.2 | +2.4 | +2.4 | +2.4 | +2.4 |
| group: claims at 6, the rest at 3 | 122.5 | 128.9 | +12.2 | | −3.3 | +11.8 | +11.8 | +11.8 | +11.8 |
| group: claims at 9, the rest at 3 | 121.1 | 128.9 | +21.2 | | −4.7 | +20.4 | +20.4 | +20.4 | +20.4 |
| group: claims at 1, the rest at 6 | 121.3 | 128.9 | +4.4 | | −6.6 | +4.4 | +4.4 | +4.4 | +3.8 |
| group: claims at 1, the rest at 9 | 119.3 | 128.9 | +9.6 | | −9.6 | +9.5 | +9.6 | +9.6 | +8.5 |
| both: claims at 1, the rest at 6 | 121.3 | 125.8 | +4.4 | +1.7 | −16.1 | −1.2 | +0.5 | +3.7 | +5.4 |
| both: claims at 1, the rest at 9 | 119.3 | 124.0 | +9.6 | +5.9 | −23.5 | +1.9 | +4.7 | +11.4 | +14.2 |
| **statement_id DELTA_BYTE_ARRAY, both** | 122.0 | 124.9 | 0 | 0 | **−15.5** | −5.0 | −2.8 | −1.3 | −0.5 |
| statement_id DELTA_BYTE_ARRAY, published only | 124.3 | 124.9 | 0 | 0 | −9.5 | −4.4 | −2.2 | −0.8 | 0.0 |
| *sensitivity:* both, claims at 1, the rest at 9 | 117.0 | 121.1 | +9.6 | +5.9 | −34.9 | −1.4 | +3.0 | +10.7 | +13.6 |
| *sensitivity:* published, claims at 1, the rest at 9 | 124.3 | 121.1 | | +5.9 | −20.8 | −10.9 | −6.5 | +1.9 | +5.6 |

The full set, levels 12–19 included, is `python model.py`'s output.

- **The sensitivity rows** take the other tables to gain more at a release than on the
  sample, at the same CPU:
  - descriptions 0.70× at 9 (its large pages' 0.73×, rounded down) and 0.80× at 6;
  - labels, links and aliases 10% smaller again than measured.

  Even then, no case reaches −15 min at 30 MB/s, and every case that gains at 7 MB/s
  loses at 60–120 MB/s.
- **Why the published claims' extra CPU costs at 7 MB/s:**
  - with L1a, each finalise run sorts its claims, then uploads them while the later
    tables are written;
  - the claims' sort is ~86% of finalise's compute and runs on all 4 cores;
  - so every CPU-second added to it delays the claims' upload, which is the run's path.
- **Why the group files' extra CPU costs from 15 MB/s:**
  - they are written in the split, which runs on ~3.6 cores;
  - at 7 MB/s the build branch's uploads end ~1h after the compute, and absorb it;
  - from 15 MB/s the uploads keep up with the compute, so the split is the path.
- **The best level for each class:**

  | class | 7 MB/s | 15–30 MB/s | 60–120 MB/s |
  |---|---|---|---|
  | group files, claims | 1 (master) | 1 | 1 |
  | group files, the rest | 9 (−10 min) | 3 (master) | 3 |
  | published, claims | 1 | 1 | 1 |
  | published, the rest | 9–12 (−10 to −11 min) | 6 (−2 to −6 min) | 3 |

  Only "published claims at 1" is the same at every rate.

### What the model assumes

- **One link each way, shared.** The build branch's uploads at the rate, finalise's
  downloads at 75 MB/s, as 17.2.0 does.
- **CPU at the sample's rate per byte,** times the release's bytes, over 4 cores.
  - The split's extra goes on its ETA, which `_phase` spreads as it does the split's
    own.
  - Finalise's goes on its claims share or the rest's. A real release's arrow-rs
    writers compress in their encoding threads, so the extra is as parallel as the
    write.
- **Each table's size factor is the sample's, for both sets.**
  - The sample's single group mixes the main and scholarly sets.
  - At a release, the scholarly set's descriptions (9.2 of the 13.2 GB of group
    descriptions) are its repetitive "scholarly article" descriptions. They could gain
    more than the sample's mix. The sensitivity rows cover a larger gain.
- **The same levels are used in every release.** The level's first release changes
  every compressed page. Xet's content-defined deduplication against the previous
  release then finds fewer shared chunks: a cost no model here counts, and one that
  #86 named too. Later releases at the same level deduplicate as now.

## The trade-off on an ordinary laptop

- **Memory and disk:** a level is CPU only. It doesn't change memory, and its files are
  smaller on disk too.
- **On a laptop's slow uplink** (7–15 MB/s), one option saves 10–21 min, modelled: the
  non-claims published tables at 9, the claims at 1. It adds +5.9 min to finalise's ETA
  (about 24 CPU-min on 4 cores).
- **On a fast link** the same option costs 2–6 min.

So the only way to have the slow-link gain without the fast-link loss is a per-release
setting, chosen by the user's link. That was not implemented: a few minutes at most,
against a configuration surface and a deduplication cost.

## Parity

- **Nothing published changed:** no code changed.
- **A level or encoding change would change the files' bytes, not their values.** The
  bench's parity compares values (`bench/parity.py`), and `_check_written` checks the
  decoded rows. The published contract (schema, content, order) has no statement on
  compressed bytes or sha256 stability across releases. Each run's sha256s are only
  checked against that run's own upload (DESIGN.md).

## Missing

- **`statement_id` as `DELTA_BYTE_ARRAY`.** −15.5 min at 7 MB/s, ~0 at 120, modelled.
  It needs:
  - a per-column encoding option in pq-glue's `write_parquet` (arrow-rs
    `WriterProperties::set_column_encoding`, the dictionary off for that column), used
    by `compact._write_file_parallel`;
  - the same in polars-genson's `write_parquet`, or the claims build moved to pq-glue's
    writer with a level and that option;
  - a check that the Hub's dataset viewer and the usual readers read `DELTA_BYTE_ARRAY`.
    pyarrow, Polars, DuckDB and arrow-rs do; not checked here.

  It passes the bar at 7 MB/s only, so it is worth doing only if the Hub's upload rate
  turns out to be near 7 MB/s.
- **The Hub's real upload rate,** which ranks everything here (#167).
- **Xet's deduplication between releases.** If it cuts a second release's uploads
  severalfold, every saving above shrinks with it, and a level change costs one
  release's deduplication.
