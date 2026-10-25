# 2026-10-25: The term and sitelink tables built and partitioned in the split (fusion A, stage 1)

Continues docs/journal/2026-10-25-partition-costs.md, 2026-10-25-one-file-per-table.md and
2026-10-25-claims-arrow-study.md (design B, merged in #134). Scripts and results are in
`2026-10-25-fused-terms-links/`.

## Current State

- **Stage 1, not wired:** dump.py, process.py, main.py and claims_build.py are unchanged.
  Nothing calls the new code yet, so the bench should show no change.
- **New code:**
  - `reshape/src/terms.rs`: labels, descriptions and aliases built from a part's reshaped
    JSON cells, as genson's normalisation and the explode build them.
  - `reshape/src/links.rs`: the same for sitelinks.
  - `reshape/src/partition.rs`: a Parquet writer for these tables' partition files.
  - `reshape/src/fuse.rs`: the entry point, `wikidata_reshape.fused_partitions`.
  - `src/wikidata/partitioning/fused.py`: `write_terms` (the split's step, in
    `claims_build.write_claims`' shape) and `place_built_terms` (the pipeline's step).
  - `scripts/test_fused_terms.py`: the identity test.
- `reshape/src/lib.rs` gains two lines: the module, and its registration.
- `reshape` gains one dependency, `zstd-safe` (libzstd 1.5.7, as Polars links).
- **Proven identical:** on all 11 bench chunks, whole and as routed parts, the fused
  partitions have the pipeline's rows, files, schemas, row groups, id and key statistics
  and sidecars. Only the encoded bytes (and so `file_size`) differ.
- **Measured locally:** the four tables' path costs about a quarter of what it does now
  (see Measured), most of it moved into the split.
- **Modelled:** about −1h20m of a release's ETA.

## What it replaces

For labels, descriptions, aliases and links, after the split has written their JSON into
the part file:

1. the part file's zstd of those four JSON columns;
2. the pipeline's read of them: genson reads the three term columns from the file itself,
   and process() reads `sitelinks`;
3. genson's normalisation: `normalise_map_direct` ×3 and `normalise_sitelinks`, with their
   processed files and sinks;
4. the explode, `prepare_for_partition`;
5. Polars' sink, `partition_parquet`, a file per key, and `write_frame_sidecar`.

**In their place:**
- the split: one call on the part's columns, already in memory (`write_terms`). It builds
  the four tables' rows and writes their partition files;
- the pipeline: renames the files into place and writes the sidecar
  (`place_built_terms`).

## Design

**Input: the split's reshaped cells.** The builders read the cells `entity_rows` writes
(orjson's compact JSON), not the dump line.
- These cells are what every part holds, including the lines the compiled reshape leaves
  to Python's `entity_row`. So there is one path for all lines.
- They are exactly the bench chunks' columns, so identity is tested against the pipeline
  on the same input.
- Reading them again costs little: the build is 0.16–0.19 CPU-s of c0's 1.5 s, about
  12% of the call (callgrind, below). Feeding the builders from lib.rs's own scan of
  `term()` would save part of that, at the cost of editing the claims agent's scan.

**The builders** (`terms.rs`, `links.rs`):
- A row is (entity, key, value), and for links also the badges. Rows are in the entities'
  order and each map's order, as genson's `kv` encoding and the explode give them.
- Keys are interned (FxHash), then grouped by a stable counting sort into the keys' byte
  order. That is the order of the sidecar's sorted paths.
- **Semantics reproduced:**
  - an empty map is null, so the row is dropped and gives no rows;
  - the links' key is the record's own `site`, and the map's key is dropped;
  - badges that are `[]` or null are null;
  - strings are fully JSON-unescaped (surrogate pairs included).

**Misfits:** a cell outside what the build reproduces exactly. Nothing is written for the
part, which keeps its JSON for the pipeline's path, where genson's schema check decides
it as now. The cases:
- a value that is not a string;
- a null;
- an empty alias list (genson makes it a null value, which the explode keeps as a row);
- a link field other than site, title and badges;
- a link without a string site or title;
- a badge that is not a string;
- a key that is empty or not a plain path component;
- anything but an object.

**The writer** (`partition.rs`):
- **Schema:** the sink's Parquet schema exactly: optional BYTE_ARRAY UTF8/STRING leaves,
  and the badges as a LIST group with its `list`/`element` children.
- **Arrow schema:** the sink's own `ARROW:schema` footer value, embedded verbatim. It is one
  constant for labels, descriptions and aliases, and one for links, in every file of every
  chunk. So Polars, pyarrow and pq_glue decode these files to the sink's Arrow types:
  `large_string`, and `large_list<large_string>`.
- **Encoding:**
  - data pages v1;
  - PLAIN values;
  - levels as RLE runs;
  - the key column as a one-entry dictionary;
  - null count, min and max on every column chunk, with `column_orders`;
  - zstd at Polars' default level, 3.
- **One zstd context for every page of every file,** and each file built in memory and
  written with one call.
- **Row groups:** at most 122,880 rows, as `ARROW_ROW_GROUP_ROWS`. See "Byte identity"
  for how the sink's differ above about 100,000 rows.
- **Layouts:** a file per key (the baseline, `partition_parquet`'s), or `one_file=True`:
  one file per table, a row group (or more) per key. With one file, the call returns each
  key's first row group, row-group count and bytes. Those bytes add up to the file's size,
  as #106's sidecar needed.

**The Python side** (`partitioning/fused.py`, in design B's shape):
- `write_terms(ids, texts, rows, routed, file) -> why | None`. `ids` and `texts` are the
  chunk's columns, as large strings. `rows` is the part's int64 positions in them, as
  `route_parts` gives `write_claims`, so nothing is copied per part.
  - Each table goes to `routed/{table}/chunk_N/{key}.parquet`.
  - Its `index.json` is written last (tmp, then rename). The index holds the keys' rows,
    bytes and id bounds, and its presence marks the table complete.
- `place_built_terms(built, name, output_dir, audit_dir, *, in_place=False)`.
  - It renames each key's file to `results/{table}/{key}/chunk_M.parquet`, where
    `partition_parquet` writes it.
  - Then it writes the sidecar as `write_frame_sidecar` writes it.
  - Then it removes the built directory.
  - It resumes: a rename already made is skipped.
- With `in_place=True`, nothing is renamed, and the sidecar names the files where the split
  wrote them. `merge_group` reads partition files only through the sidecars' paths, and
  nothing else in the pipeline globs the key directories.

## Identity (proven, `scripts/test_fused_terms.py`)

**What each run checks,** for each of the four tables, on each chunk:
- the same partition files, by relative path (so the same keys and names);
- each file's rows equal to the pipeline's exploded rows of its key, in order, with the
  same Polars dtypes;
- each file read the same as the sink's by Polars, pyarrow and `pq_glue.read_parquet`:
  values and Arrow schema;
- the same Parquet schema, the same row groups (count and rows each), and per row group
  the same `id` and key statistics: min, max and null count, which the sort's planning
  reads;
- each fused file glued (`pq_glue.concat`, as `merge_group` glues) after the sink's file
  of its key, reading back as both. So a group may mix fused parts with parts that took
  the JSON path;
- the sidecar equal to `write_frame_sidecar`'s in schema, rows and order, except
  `file_size`, which equals each fused file's own size;
- the in-place placement: the sidecar names the built files, and each reads as the sink's;
- the one-file layout: each key's row groups read back as its key file.

**The inputs:**
- the 11 bench chunks (c0, r10–r90, s10–s90), each whole, and as a routed part (every
  third row, given as positions in the chunk's columns, against the pipeline run on a
  chunk file of just those rows);
- edge cases:
  - escapes (`"`, `\`, `/`, tab, newline, `\u0001`, `\u001f`, an astral emoji);
  - empty strings and empty maps;
  - an entity with no terms;
  - badges `[]`, null and several;
  - fields out of order;
  - a map key that differs from its link's site;
  - multi-byte keys;
  - `be` / `be-tarask` and `zh` / `zh-hans` / `zh_min_nan` (keys that sort differently as
    paths than as strings);
  - a key of 99,000 rows;
- nine misfits, each writing nothing.

**Result:** all 841 checks pass. The rows and files compared over the 11 chunks:

| table | rows | partition files | sink bytes | fused bytes | fused / sink |
|---|--:|--:|--:|--:|--:|
| labels | 708,524 | 2,177 | 13,337,558 | 12,231,881 | 0.917 |
| descriptions | 1,863,727 | 1,934 | 12,419,315 | 12,444,314 | 1.002 |
| aliases | 185,421 | 1,743 | 4,530,516 | 3,512,734 | 0.775 |
| links | 424,962 | 1,212 | 7,367,021 | 6,548,945 | 0.889 |
| all | 3,182,634 | 7,066 | 37,654,410 | 34,737,874 | 0.923 |

(Per chunk: `sizes.jsonl`.)

## Byte identity with Polars' sink: not attempted, and why

- **Polars' encoder makes choices this writer doesn't copy:**
  - which columns get a dictionary: its files dictionary-encode `language`, and some
    files' `id`;
  - its page boundaries;
  - its statistics fields;
  - its `created_by` (`Polars (python) version 2.0.0 (build …)`). Writing that string
    would claim to be Polars, which readers use for version-specific workarounds.
- **Its row groups come from its streaming engine, not a fixed size.**
  - A key of up to about 100,000 rows in a chunk is one row group, as here.
  - Above that the sink splits by its morsels: 140,000 aliases rows came out as
    105,000 + 35,000, and 200,000 labels rows as 2 × 100,000. Those boundaries depend
    on the engine.
  - At a release's chunk size (10,000 entities), a labels, descriptions or links key
    holds at most 10,000 rows a chunk. An aliases key would pass 100,000 only with an
    average of 10 aliases an entity in one language. On the bench, c0's `en` has 13,529.
  - Above 100,000 rows the fused file keeps the same rows, order and schema, in row groups
    of 122,880. The test's 210,000-row case checks exactly that.
- **What depends on the bytes downstream:** `file_size`. It feeds the group threshold
  (`chunk_partition_bytes`) and, through the group files' sizes, compaction's runs and the
  sort's thresholds (#106's journal lists them).
  - The fused files are 7.7% smaller overall, so at a release a key near one of those
    thresholds could take another branch. That would give other part-file boundaries for
    the same rows in the same order.
  - The user has since authorised decoupling those thresholds from encoded sizes
    (sort/decoupled-thresholds), so this is in hand there.
  - Every reader decodes the files identically: values, schema and statistics. The glue
    copies their row groups as they are, and the sort re-encodes every key from decoded
    rows. So once the thresholds no longer read encoded sizes, nothing published depends on
    these bytes.
  - That end-to-end claim (the published part files unchanged) is for CI to show at
    stage 2. It has not been run end to end here.

## Measured (local, indicative: the shared 4-core VM)

**CPU of the four tables' path** (user + sys, every thread; `cpu.py`, the median of 5 runs
after a warm-up, in one process; `cpu.jsonl`). CPU-s:

| chunk | pipeline: normalise + read | pipeline: `partition_parquet` | pipeline total | split: `write_terms` | split: part file's zstd saved | split, net | pipeline: place (renamed) | pipeline: place (in place) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| c0 | 0.93 | 4.39 | 5.32 | 1.53 | −0.34 | +1.19 | 0.10 | 0.04 |
| r50 | 0.05 | 1.45 | 1.50 | 0.30 | −0.02 | +0.29 | 0.04 | 0.02 |
| s30 | 0.16 | 0.40 | 0.57 | 0.21 | −0.06 | +0.16 | 0.02 | 0.01 |
| s90 | 0.27 | 0.54 | 0.81 | 0.29 | −0.07 | +0.21 | 0.02 | 0.01 |
| the four | 1.41 | 6.78 | 8.19 | 2.34 | −0.49 | +1.85 | 0.18 | 0.09 |

How the columns are made:
- **The part file's zstd saved:** the part written as design B writes it (no claims
  column), less the part with the four JSON columns gone too (`id`, `entity`).
- **`write_terms`** includes the columns' export to Arrow large strings, out of Polars'
  string views.
- **The renames** are timed with the key directories already made, as they are after a
  release's first chunk.

**Before → after:**
- the pipeline: 8.19 → 0.18 CPU-s (−98%);
- the split: +1.85;
- the path as a whole: 8.19 → 2.03 CPU-s (−75%).

**Where `write_terms`' time goes:**
- **sys time, file creation:** on this VM's disk, creating a file costs about 0.38 ms of
  sys time. On c0, building and writing 2,327 files took 0.72 s user and 0.89 s sys.
- **The same on tmpfs:** 0.60 s user, 0.03 s sys.
- **The one-file layout on disk:** 0.70 s user, 0.02 s sys.
- So the per-file cost left is the file system's, which Polars' sink pays too. Only fewer
  files remove it: the one-file layout. Fresh key directories (mkdir) and renames cost
  about as much again, which is why the timed renames have the directories made first.

**callgrind** (instruction counts, stable on this VM; the call on c0, one file per table,
on tmpfs; before the switch to FxHash), 4.50G instructions:

| | Ir | share |
|---|--:|--:|
| zstd (`ZSTD_compressCCtx`, inclusive) | 2.21G | 49% |
| cell parsing (`Cell::string`) | 0.54G | 12% |
| key interning (SipHash, since replaced by FxHash) | 0.35G | 8% |
| PLAIN encoding and min/max (`Writer::strings`, memcmp) | 0.51G | 11% |

- zstd's share is real compression (`doubleFast` blocks, sequence encoding), not context
  set-up: the context is created once and the page memsets are small.
- Every row group costs a few pages' compression. That is why the user time is the same
  in either layout.

**Peak RSS added to a process holding a part's columns** (`rss.py`, a fresh process each;
`rss.jsonl`):

| chunk | entities | the four columns' cells | peak added |
|---|--:|--:|--:|
| c0 | 10,000 | 58.8 MB | 122 MB |
| r50 | 7,000 | 3.0 MB | 14 MB |
| s30 | 7,301 | 17.9 MB | 45 MB |
| s90 | 8,919 | 23.8 MB | 59 MB |

Where the 2.1–2.5× the cells' bytes goes:
- about 1× is the decoded rows;
- about 1× is the export copy out of Polars' string views. Stage 2 can drop it by passing
  the reshape's own Arrow large strings, as `entity_rows` hands them over, rather than the
  frame's columns;
- the rest is a file's buffer and its row groups' slices.

**What is held:** one part's rows at once, nothing larger. A part is at most
`CHUNK_ENTITIES` (10,000) entities. Each table is built, written and dropped in turn.

## Modelled: the release (not measured)

**Rates:** bench 11.0.0's ETA by stage, as the brief gives them: the split at 327 ETA-s a
CPU-s of the sample, and the pipeline's rest at 1,142. (#132's comment has 1,104 for the
rest: 3% less saving.)

**Assumption: CI's CPU, not this VM's.** The local figures above are about twice CI's for
this file-heavy work. The four tables' `partition_parquet` takes 6.78 CPU-s here, against
3.5 s for every table's on CI's quick tier (#132, "after"). So the model starts from CI's
measured figures and scales the fused path by the local ratios:

| | CI-anchored CPU-s on the sample | how |
|---|--:|---|
| `partition_parquet`, now | 3.5 | #132, all tables: entities' one file a part and the empty claims_labels taken in, a slight overstatement |
| `normalise_sitelinks` and the links' processed sink and read, now | 0.5 | #132's `normalise_sitelinks` 0.4, and about 0.1 of `atomic_sink_parquet` |
| genson's term normalisation, now | 0.7 | local 0.86 × 0.8 (CI's `normalise_sitelinks` over local's) |
| **the pipeline, now** | **4.7** | |
| the pipeline, fused: placement (renamed) | 0.09 | local 0.18 × 0.52 (CI's partition over local's) |
| the split, fused: `write_terms` | +1.21 | local 2.34 × 0.52 |
| the split, fused: the part file's zstd saved | −0.39 | local 0.49 × 0.8 |

**The release:**
- the pipeline: −(4.7 − 0.09) × 1,142 ≈ **−1h28m**;
- the split: +0.82 × 327 ≈ **+4.5m**;
- **net: about −1h23m** of #132's 9h35m (−14%).

This matches the release profile's modelled costs of these stages: `partition_parquet`
about 1h05m, genson's normalisation about 15m, and the links' sink and read about 6m.

**Taking the local CPU as it is:** −2h32m + 10m ≈ −2h22m. That is an upper bound: it
applies CI's rates to the VM's slower CPU-s.

**Wall time:**
- From bz2, the split is CPU-bound (lbzip2), so the split's +4.5m is wall time.
- From gz, the split waits on the download, and the added work is nearly free.
- The pipeline's −1h28m is wall time at its usual busy factor.

**With the one-file layout** (stage 2, once its sidecar format is settled), the split's
file creations drop from about 2,300 a chunk to 4. On this VM that is about 0.9 s of c0's
1.5 s. On CI the share is unknown until measured.

## Stage 2: what wiring needs (for the agent the coordinator assigns)

**The split** (`dump.route_parts`' per-part loop, beside `write_claims`, and before the
part file is written):

```python
texts = {t: fused.large_strings(frame.get_column(c)) for t, c in fused.FUSED.items()}  # once a chunk
...
why_terms = fused.write_terms(frame.get_column("id"), texts, rows, set_dir / ROUTED, entry["file"])
if why_terms is None:
    part = part.drop(*fused.FUSED.values())   # the part file without the four JSON columns
else:
    out[name]["terms_json"] = why_terms       # the misfit: the part keeps them, as now
```

- `rows` is the same int64 positions array `write_claims` takes.
- Better for memory and CPU: build `texts` from the reshape's own large-string buffers
  (`entity_rows`' output, with Python's fallback cells scattered in) before `pl.from_arrow`
  turns them into string views.
- The fused build could go under `claims_build.BUDGET`. It holds about 2–2.5× the part's
  cells, against the claims' 3×.

**The misfit signal:** `write_terms` returns None once written, else why
(`"aliases: row 3: an empty list of aliases"`).
- The decision is per part, all four tables or none. A per-table fallback is open: pass a
  subset of `texts`.
- `route_chunk` (old, unrouted chunk files) passes no texts, so its parts keep their JSON.

**Restart safety:**
- `write_terms` first removes what an earlier attempt left of the part.
- Each table's `index.json` is written after all its files (tmp, then rename).
- The part file comes after, then the manifest line (already the order in `split()`).
- `place_built_terms` resumes: renames already made are skipped, and the sidecar comes
  after all of a table's files. The index is removed last.

**`_move_routed`:** for each table in FUSED, move `routed/{table}/chunk_N/` to
`{set_dir}/{table}/chunk_M/` (one rename a table a part), skip it if already moved, then
remove `routed/{table}`.

**process.py:** for a part without the four columns (or one whose entry has no
`terms_json`), skip `normalise_map_direct` ×3 and `normalise_sitelinks`, and their
processed files and `check_ids`. The build cannot lose an id: each entity's cells give
its rows or the part is a misfit.

**main.partition_chunk:**
- For these four tables of a built part, call
  `fused.place_built_terms(data_dir, filename, output_dir, AUDIT_DIR)` in place of
  `prepare_for_partition` + `partition_parquet`.
- `validate_chunk_outputs` expects a processed file per table: it could take the built
  directory's `index.json` instead.
- With `in_place=True`, the emptied `{set_dir}/{table}/chunk_M/` directories need removing
  after the merge, as `_remove_empty_dirs` does for the results' key dirs.

**The bench:**
- The split's parity compares the chunks' JSON columns. Built parts lack four of them, so
  it must compare the decoded tables or the partition files, as #133 did for the claims.
- The pipeline parity compares sidecars without `path` and `file_size` (bench 8.0.0), so
  it holds.

**The one-file layout:** `fused_partitions(..., one_file=True)` already writes it and
returns each key's row groups and bytes. `write_terms` and `place_built_terms` would take
it once sort/decoupled-thresholds settles the sidecar's form.

## Negative findings and what is left

- **No byte identity** with Polars' sink, as above.
- **Rust doesn't remove the per-file cost** on this VM's disk. File creation's sys time
  (about 0.38 ms a file) is the file system's, and it stays as long as there is a file
  per key. The writer's own per-file cost is small: a key's file is a few pages'
  compression. Only the one-file layout removes the file system's share.
- **zstd is half the writer's CPU,** and it is real compression of small pages.
  - Untested: zstd level 1 (the sink at level 1 was 5% cheaper and 3.6% larger,
    partition-costs journal).
  - Untested: compressing tiny pages at all (the codec is per column chunk, so a page can't
    go uncompressed on its own).
- **Feeding the builders from lib.rs's scan, not the cells,** would save at most the
  cells' parse (about 12% of the call). Not done, since that scan belongs to the claims
  agent's file.

## Scripts and results

In `2026-10-25-fused-terms-links/`:
- `cpu.py` (results in `cpu.jsonl`): CPU of the pipeline's path against the fused path;
- `rss.py` (`rss.jsonl`): the peak RSS added;
- `sizes.py` (`sizes.jsonl`): rows, files and bytes by table.
