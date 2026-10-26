# 2026-10-26: `statement_id` as DELTA_BYTE_ARRAY in the claims' group and published files

This follows the lead in `2026-10-26-link-bytes.md` (#174). A full release is network-bound, and
`statement_id` is a third of the claims' bytes on the link. pyarrow writing it as
`DELTA_BYTE_ARRAY` cut the column 12% (published) and 7% (group) with no extra CPU, which
modelled as −15.5 min at 7 MB/s and ~0 at 120 MB/s. #174 couldn't ship it because neither of
the writers that write the claims could set a column's encoding.

**This changes the physical Parquet bytes of the claims files, so they are not
byte-identical to before. Values, schema and row order are unchanged.**

## Current State

- **The code:**
  - pq-glue's writer (lmmx/pq-glue#13) and polars-genson's (lmmx/polars-genson#322) take a per-column encoding.
  - wikidata-pq writes the claims' `statement_id` in DELTA_BYTE_ARRAY, in the group files and the published files.
  - The pull request takes both writers from their pull requests' CI wheels, until they are released.
- **Measured on the bench sample:**
  - group claims files −2.7%;
  - published claims files −4.2% (the column −7.7% and −11.4%);
  - the same values, schema and row groups.
- **Not used:** the reference and qualifier hashes, and `id`, grow in DELTA_BYTE_ARRAY, so they keep their encodings.

## Where the encodings are chosen

| file | writer | where its encodings were chosen |
|---|---|---|
| group files: claims | `partitioning.core.write_table` → `polars_genson.write_parquet` → `genson_core::parquet::write_arrow_batches` → pq-glue's `write::write_batches` (arrow-rs, columns in parallel, zstd 1) | pq-glue `write::properties`: arrow-rs defaults (PLAIN, `WriterVersion::PARQUET_1_0`), dictionary per leaf unless its sampled values are >¾ distinct (`unrepeated_leaves`). `statement_id` was PLAIN, no dictionary. The split's `claims_build.RowGroups` writes each row group with `write_table` and glues them with `pq_glue.concat`, as does `release_schema` for its files. |
| published: claims (and every table) | `sort_by_id.write_parts` / `_sort_bucket` → `compact._write_file_parallel` → `pq_glue.write_parquet(engine="parallel")` (arrow-rs, row groups glued with CDC, zstd 3) | the same `write::properties` |
| published, the bucket sort's part files | `_pack_file` → `pq_glue.concat` of the sorted buckets | copied verbatim: each bucket's encodings carry through |
| published, deduplicated keys (`compact` with overlapping group files) | `compact._write_file` → `pq.ParquetWriter` (pyarrow) | pyarrow's defaults: a dictionary for every leaf, falling back to PLAIN |

## The change

- **pq-glue (lmmx/pq-glue#13):**
  - `WriteOptions::column_encodings`, which is `column_encoding=` in Python, works in every engine. Each named leaf column is written in its encoding with no dictionary.
  - `write::leaf_encodings` / `check_encoding` check the option before anything is written. A column the schema lacks, an unknown encoding, a dictionary encoding, or one the column's type does not take raises `ValueError`.
  - `concat` copies column chunks as they are, so the encoding carries through the glue, and files in different encodings glue together.
- **polars-genson (lmmx/polars-genson#322):**
  - `write_parquet(..., column_encoding=)` passes the option through. genson-core adds `column_encodings` and `write_arrow_batches_encoded`.
  - genson-core takes pq-glue from #13's git rev until pq-glue's release.
- **wikidata-pq:**
  - `partitioning.core.COLUMN_ENCODINGS = {"statement_id": "DELTA_BYTE_ARRAY"}`.
  - `column_encoding(schema)` gives the entries whose leaf the table has, so only claims tables are affected.
  - It is passed by `write_table` (group files), `_write_file_parallel` (published and the sort's scratch files) and `_write_file` (pyarrow: `use_dictionary` is every other leaf, as pyarrow takes an encoding only for a column with no dictionary).

## Measured: bytes, on the bench sample

- **The sample:** the quick tier's claims (c0 r50 s30 s90: 1,175,112 rows) as the bench's split writes them.
  - `split_sample.py` runs `bench/pipeline.split_stage` with `keep`, on master's code and master's genson 1.5.0. Its parity passed.
  - The 5 group files total 73.16 MB (#174's 73.11).
- **The writers:** `encmeasure.py` rewrites them as the writers do.
  - Group files are written as `write_table` writes them (pq-glue `write_columns`, zstd 1, rows of 122,880). The baseline gives 73.14 MB, within 0.03% of the split's files; the split glues row groups written apart.
  - Published files are each set's claims sorted by id (stable), as `_write_file_parallel` writes them (zstd 3, CDC, `row_group_rows`).
- **Checks:** every variant's values read back equal to the input's. Data: `encodings.jsonl`.

| | master | `statement_id` DELTA_BYTE_ARRAY | change |
|---|---|---|---|
| group: `statement_id` | 25.40 MB | 23.44 MB | −7.7% |
| group: claims files | 73.14 MB | 71.18 MB | **−2.7%** |
| published: `statement_id` | 27.14 MB | 24.03 MB | −11.4% |
| published: claims files | 74.40 MB | 71.29 MB | **−4.2%** |

- **arrow-rs gives what pyarrow gave.** #174 measured 25.41 → 23.57 (group) and 27.22 → 23.94 (published) with pyarrow.
- **Other columns, measured and not taken.** Each is relative to `statement_id` alone, in the published files:

  | column in DELTA_BYTE_ARRAY | change |
  |---|---|
  | `references.list.element.hash` | +7.7% (+0.30 MB) |
  | `qualifiers.list.element.value.list.element.hash` | +10.7% |
  | `id` | +10% (its dictionary and RLE do better) |
  | `statement_id` as DELTA_LENGTH_BYTE_ARRAY instead | only −1.8% on the column |

  The hashes are SHA-1 hex digests with no prefix shared with the value before, and many repeat, which the dictionary already gets. The group files show the same.
- **Pages must not be tiny.** In pq-glue's tests, CDC chunks of 256–1024 B made `statement_id` *larger* in DELTA_BYTE_ARRAY (539 kB against 528 kB): each page carries the delta blocks' headers. At the default chunk sizes (256 KiB–1 MiB), which the pipeline uses, it is smaller.

## CPU

- `encmeasure.py`'s figures are order-biased, because master always ran first, and the machine is shared. As measured there:
  - group write: 5.42 → 4.94 CPU-s;
  - published write: 7.09 → 7.31 CPU-s, within noise;
  - pyarrow whole read: 5.96 → 4.38 (group) and 4.95 → 4.16 (published).
- #174 measured with pyarrow 0.42 against 0.47 CPU-s to write the column, and 0.30 against 0.47 to read it.
- **`cpu_ab.py` interleaves the arms** (A B / B A, 5 rounds, the CI wheels; `cpu_ab.jsonl`; the machine at a load of 3–5). Medians, in CPU-s:

  | | master | DELTA_BYTE_ARRAY |
  |---|---|---|
  | group write | 5.45 | 5.22 |
  | published write | 7.26 | 7.33 |
  | `statement_id` read alone, group | 0.275 | 0.312 |
  | `statement_id` read alone, published | 0.344 | 0.403 |

  - The writes show no change beyond noise.
  - Decoding the column costs about +0.05 CPU-s per 1.2M rows. #174's pyarrow pages read faster; pq-glue's arrow-rs pages don't.
  - Whole-file reads swung ±1–3 CPU-s either way under the load, so they can't resolve it.

## On CI: the bench (#184, run 40376886224, head 5b7b8c2)

- **The stack:** the after arm took both pull requests' wheels.
- **Parity passed.** Only the sidecars' `claims.file_size` differs, which `MAY_DIFFER` allows.
- **Bytes:**
  - the groups' uploads on the sample: 101.0 → 99.0 MB;
  - finalise's published files on the sample: 105.6 → 102.5 MB;
  - at a release, the build branch 124.3 → 121.5 GB and the published files 128.9 → 124.6 GB.
- **Round 3r was a runner slowdown.** Its after arm's bz2 decode alone, untouched by the change, took 29.9 CPU-s against 15.8.
- **The other five rounds agree.**

| after − before, by round (1f 1r 2f 2r 3f 3r) | | median of 6 |
|---|---|---|
| measured CPU (s) | −0.2 −1.4 −1.8 −1.3 −7.2 +57.5 | −1.4 |
| release wall at 7 MB/s (min) | −10.7 −18.2 −15.7 −23.8 −16.2 +92.0 | **−15.9** |
| release wall at 30 MB/s (min) | −1.1 −6.6 −5.4 −27.6 −11.3 +157.8 | −6.0 |
| release wall at 120 MB/s (min) | +2.8 −4.6 −3.9 −25.5 −7.5 +165.6 | −4.2 |

- **At 7 MB/s:** #174 modelled −15.5 min. The bench's summary reads "no change beyond noise" because of 3r.
- **CPU:** no regression.

## Parity, and what compares bytes

- **Values, schema, row order and row groups are unchanged.** Every writer was checked (`smoke.py`: `write_table`, `_write_file`, `_write_file_parallel`, scratch included): the rows read back are equal, and the Arrow schema with its metadata is equal. pq-glue's and genson's tests check statistics and key-value metadata too.
- **The bench's parity (`bench/parity.py`) does not see encodings.** It compares row hashes, and the SHA-256 of `pq.read_schema(f).serialize()`, which is the Arrow schema and its key-value metadata. The sidecars' `file_size` is already in `MAY_DIFFER`.
- **`split_manifests_sha256` does cover encodings, indirectly.** It hashes the split's manifests, whose entries hold each part's `claims_bytes` (the claims file's size).
  - It is compared only within an arm: the gz split's against the bz2 split's (`gz_split_parity_ok`). Both run the same code, so the encoding changes neither side's claim. It is never compared across arms.
  - Across arms it differs, as the byte figures do. That is intended.
- **The exact checks are unaffected:**
  - `_check_written` and `check_compacted_copy` decode rows.
  - `_check_glued` compares a glued file with its parts, which are written the same way.
  - `check_same` (pq-glue's glue) compares the Parquet schema, the column orders and the key-value metadata, which encodings don't touch.
- **Restart safety:** a key whose files were written before the change, then glued with new ones, mixes PLAIN and DELTA_BYTE_ARRAY chunks. That is valid Parquet, and pq-glue's tests glue exactly that.
- **Readers:** pyarrow, Polars and arrow-rs read it (tested). DELTA_BYTE_ARRAY is in the Parquet format since 2.0 and DuckDB reads it. The Hub's dataset viewer (DuckDB and pyarrow) is not checked here; nothing here touches the Hub.

## Missing

### Before release: the stack, measured on CI

The pull request takes pq-glue and genson from the wheels their pull requests' CI built (`bench/wheels`, `[tool.uv.sources]`), so the bench runs the whole stack before anything is released.

Once both are released (pq-glue 0.3.0, then polars-genson 1.6.0 on it):
- `bench/wheels` and the path sources go;
- `pyproject.toml` takes `pq-glue>=0.3.0` and `polars-genson>=1.6.0`, and `uv lock` runs again.

Done: both are released (pq-glue 0.3.0 from lmmx/pq-glue#13; polars-genson 1.6.0 from lmmx/polars-genson#322, on pq-glue 0.3.0 from crates.io, the same source as the rev the wheels were built from). `bench/wheels` and the path sources are gone, and the lock takes both from PyPI. With the released wheels, the four writers (`write_table`, `_write_file`, `_write_file_parallel`, and its scratch path) write `statement_id` as DELTA_BYTE_ARRAY with no dictionary, and each file reads back equal to its table, schema and metadata included. The bench's rounds after this are on the released stack.
