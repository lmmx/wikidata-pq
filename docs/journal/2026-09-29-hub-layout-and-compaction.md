# 2026-09-29: Hub layout, download cost, compaction

Facts gathered on 2026-09-29 after the run finished (`[run] All chunks complete.`, 31 groups in `state/groups.jsonl`, the last `chunks-7263-7448`). Hub figures come from `HfApi.list_repo_tree(recursive=True)` and Parquet footers read through `HfFileSystem`, both read-only; audit figures come from `audit/{table}/chunk_*.parquet`.

## Current State

### Layout on the Hub

- `push_group` uploads one file per partition key per group, at `{key}/{group.name}.parquet`, with `upload_folder(allow_patterns=f"*/{group.name}.parquet")` (src/wikidata/push/core.py:124-128) — a key present in all 31 groups has 31 files.
- The group size targets `GROUP_TARGET_COUNT = 30` groups to keep each repo under the Hub's 100k file guidance (src/wikidata/config.py:76-78), and does not take per-key file size into account.
- Files and bytes per repo on the Hub:

| Repo | Keys | Files | GB | Mean file | Largest key |
|---|---|---|---|---|---|
| wikidata-labels | 621 | 17,075 | 7.54 | 440 KB | en 692 MB |
| wikidata-descriptions | 594 | 13,912 | 6.84 | 490 KB | en 461 MB |
| wikidata-aliases | 587 | 15,674 | 1.66 | 106 KB | en 174 MB |
| wikidata-links | 955 | 25,057 | 1.45 | 58 KB | enwiki 146 MB |
| wikidata-claims | 1 | 31 | 18.24 | 590 MB | all 18,236 MB |
| wikidata-claims_labels | 618 | 18,955 | 17.8 | 940 KB | en 738 MB |

- The keys of every repo on the Hub equal the keys in the audit sidecars for the same table.
- Keys with under 1 MB of partition files in total, from the audit sidecars: labels 136, descriptions 244, aliases 142, links 499 of 955, claims_labels 48.
- `hf download permutans/wikidata-links --repo-type dataset` on the user's machine fetched 20,106 of 25,059 files in 27:48 at 13.77 files/s and about 50 kB/s — the download rate follows the file count, not the 1.45 GB.
- 16 links keys contain `_` (`zh_min_nanwiki`, `be_x_oldwiki`, `roa_rupwiktionary`, ...); every other key of every table matches `[a-z0-9]+(-[a-z0-9]+)*`.
- `mul` is a key of labels, aliases and claims_labels, and is absent from descriptions — Wikidata's `mul` code applies to labels and aliases only (https://www.wikidata.org/wiki/Help:Default_values_for_labels_and_aliases).

### Parquet settings of the uploaded files

- `merge_group` writes each staged file with `lf.sink_parquet(tmp)` and Polars defaults (src/wikidata/push/core.py:72).
- The footers of `chunks-0000-0088.parquet` in links/enwiki, labels/en, claims_labels/en and claims/all report `created_by` Polars, ZSTD, column statistics and an offset (page) index.
- Row groups in those four files: links/enwiki 7 groups of about 83k rows and at most 3.3 MB uncompressed, labels/en 10 of about 88k rows and 3.4 MB, claims_labels/en 18 of 122,880 rows and 4.4 MB, claims/all 182 of 122,880 rows and 21.6 MB — Hugging Face's Dataset Viewer documentation gives 100-300 MB uncompressed per row group and about 500 MB per file.
- pyarrow 25.0.1 is locked in uv.lock; `use_content_defined_chunking` in `pyarrow.parquet` needs pyarrow 21 or later.

### Row counts

- `merge_group` checks the merged row count equals the audit sidecar sum for labels, descriptions, aliases, links and claims (src/wikidata/push/core.py:75) — for those five tables the audit sums equal the row counts on the Hub.
- Audit row sums: labels 642,354,157, descriptions 1,550,918,676, aliases 171,954,210, links 98,773,816, claims 774,255,243.
- `merge_group` deduplicates claims_labels within each group (`DEDUPLICATE`, `lf.unique(maintain_order=True)`, src/wikidata/push/core.py:29, 69) and checks only `0 < n <= expected` (src/wikidata/push/core.py:75) — the merged claims_labels count is not recorded anywhere, and the audit sum of 6,160,704,670 counts rows before deduplication.
- claims_labels/cy has 38,831,447 rows in the audit sidecars and 8,243,053 rows in its 31 files on the Hub (footer `num_rows`).
- Reading the footers of the 31 claims_labels/cy files with 16 threads took 2.6 s.

### Dataset cards

- `_ensure_dataset_card` uploads `docs/dataset_cards/{table}.md` as README.md only when the repo has no README.md (src/wikidata/push/core.py:96-110) — edits to the templates after a repo's first push do not reach the Hub.
- Each card declares one config, `default`, with `data_files: "*/*.parquet"` (claims: `"all/*.parquet"`), and no `dataset_info` — the Hub shows no subsets and no per-key sizes.
- README.md describes a claims table with one row per claim per language and `language=en/` directory partitioning; neither matches the claims and claims_labels tables on the Hub.

### Hub upload behaviour (huggingface_hub 1.32.0, the uv.lock version)

- `upload_folder` with `hf_xet` installed commits through `pipelined_upload`, which splits a large folder into several commits with a `(part N)` message suffix (huggingface_hub/hf_api.py, `upload_folder` docstring and body).
- `upload_folder(delete_patterns=...)` puts every delete operation into the first of those commits (huggingface_hub/_upload_pipeline.py:584-586, "Deletions and `parent_commit` ride the first commit") — files added in later commits are absent from the repo between the first commit and theirs.
- `create_commit` applies its add and delete operations in one commit.

### Compaction (src/wikidata/compact.py)

- `finalise-wikidata` (`wikidata.main:finalise`, pyproject.toml, `just finalise`) runs `compact_table` on each table after checking every chunk is complete and no group is unfinished, and `run()` calls `finalise` after `[run] All chunks complete.` (src/wikidata/main.py).
- `compact_table` records the stages downloaded, written, committed, verified and done per table in `state/compact.jsonl` and resumes after the last recorded stage.
- `download` fetches a table's `*/chunks-*.parquet` files with `snapshot_download` into `compact/src/{table}`.
- `write_key` splits a key's group files into runs of consecutive group files of at most `COMPACT_FILE_BYTES` (500 MiB) and names each output file by the chunk range of its run (`chunks-{first}-{last}.parquet`) — a group file larger than 500 MiB forms a run of its own.
- `_write_file` writes with pyarrow `ParquetWriter`, ZSTD level 3, page index and `use_content_defined_chunking`, in row groups of `_row_group_rows` rows — `_row_group_rows` derives rows from `COMPACT_ROW_GROUP_BYTES` (128 MiB) and the Arrow in-memory bytes per row of the first row group of the key's largest group file.
- `write_key` deduplicates claims_labels across all of a key's group files, keeping each row's first occurrence in chunk order, and sizes runs by the rows each group file keeps.
- `_compare` checks output files have the group files' schema, then reads both back with Polars and compares `_fingerprint`s: `_check_file` compares each output file against its run of group files as soon as the file is written, and `_check_deduplicated` compares all of a claims_labels key's output files against the group files' distinct rows in first-occurrence order.
- `write_key` appends each output file that passes `_check_file` to `compact/out/{table}/files.jsonl` (key, name, group files, rows, bytes, sha256), and on a restart reuses a listed file without rewriting it when its group files, size and sha256 match — `rewrite_table` appends a key to `manifest.jsonl` once all its files are written and skips keys already there, so a restart inside a key redoes only its unfinished file, and claims_labels keys resume per key.
- A synthetic 4-file claims key with a simulated crash while writing its third file resumed by rewriting only the third and fourth files, left the first two unchanged (same mtime), and rewrote a listed file whose bytes were altered after its check.
- `_fingerprint` computes, with Polars' streaming engine, the row count and two sums of `pl.struct(pl.all()).hash(seed)` over rows carrying a row index, for seeds 0 and 1 — a changed value, a null replaced by a value, a dropped row and a row reordering each changed the fingerprint of a synthetic links key and of a claims-shaped frame.
- `commit_table` sends each key's additions and deletions in the same `create_commit`, batching keys up to 50 additions or 2,000 operations per commit, skips a key whose Hub files match its output files by name, size and sha256, and refuses a key whose Hub files include a name that is neither a group file nor an output file.
- `verify_table` checks the Hub keys equal the manifest keys and every key's Hub files match its output files; `write_metadata` then writes files, bytes and rows per key to `docs/dataset_cards_metadata.json`.
- `finalise` compacts the tables one at a time in `Table` order (labels, descriptions, aliases, links, claims, claims_labels), each through all its stages before the next (src/wikidata/main.py, src/wikidata/config.py).
- `download` runs `snapshot_download` with `max_workers=COMPACT_DOWNLOAD_WORKERS` (32); `compact_table` builds its `HfApi()` from the environment's Hugging Face token, as `close_group` does.
- `_write_file` writes to `{name}.tmp` and renames it to `{name}` once closed, and starts a new row group when the pending batches reach `_row_group_rows` rows — a row group can exceed `_row_group_rows` by up to one batch.
- `write_key` skips a claims_labels run whose group files' rows all occur in earlier group files of the key, so no empty file is written and the key's chunk ranges can have gaps.
- claims_labels output has each (field, ref, language, label) row once per key, where the group files have it once per group.
- `commit_table` refuses the whole table when a key on the Hub is absent from the manifest, names each commit `Compact {n} keys ({first key} to {last key})`, and adds all of a key's output files in its commit — a file already on the Hub with the same content is uploaded again by name and deduplicated by the Hub's storage.
- `verify_table` compares Hub files to the local output files by size and sha256 (LFS) or git blob sha1, and does not read their content.
- `write_metadata` writes `{table: {key: {"files", "bytes", "rows"}}}` to `docs/dataset_cards_metadata.json`, keys sorted by name and tables in `Table` order, replacing only the table compacted.
- At the done stage with `CLEAN_UP_LOCAL`, `compact_table` deletes `compact/src/{table}` and each key directory under `compact/out/{table}`, and keeps `manifest.jsonl` and `files.jsonl` in `compact/out/{table}`.
- `write_key` and `_check_file` need about one output row group per Polars thread for claims (see the local tests below); `_deduplicated` holds a claims_labels key's distinct rows in memory.
- pyproject.toml lists `pyarrow>=25` and the `finalise-wikidata` script.
- `_source_batches` reads group files a row group at a time without casting — `RecordBatch.cast` to the same schema on a claims batch from `iter_batches` raised `Struct child array #16 has length smaller than expected for struct array (65536 < 85150)`.
- The cast failure reproduces on pyarrow 25.0.1, the latest release on PyPI on 2026-09-29: casting a struct with a `null`-typed field (claims `datavalue.altitude`) to its own type, inside a list with fewer rows than structs or sliced, returns an array whose null-typed child has the input's length instead of the struct's — `Array.cast`, `ChunkedArray.cast` and `pyarrow.compute.cast` return it without raising, and `RecordBatch.cast` raises because it validates its result (docs/pyarrow_bug_report/README.md, repro.py, matrix.py, repro_parquet.py).

### Local rewrite test (download and write stages only, nothing committed)

- links cywiki, enwikiquote and zh_min_nanwiki: 31 group files each became one `chunks-0000-7448.parquet`, with rows equal to the group files' rows in order (`DataFrame.equals`); cywiki 5.3 MB became 5.5 MB, zh_min_nanwiki 6.0 MB became 6.2 MB.
- claims_labels/cy: 31 group files with 8,243,053 rows became one file with 1,485,803 rows, equal to the group files' distinct rows in first-occurrence order; 100.0 MB became 19.7 MB.
- claims `all/chunks-7263-7448.parquet` (276 MB, 13,827,474 rows) with row groups sized from uncompressed Parquet bytes per row: 15 row groups of about 950k rows and 45-79 MB of uncompressed Parquet each, and `_fingerprint` of that file took 3.44 GB peak RSS on one Polars thread — claims rows take several times more bytes in memory than in uncompressed Parquet.
- claims `all/chunks-7263-7448.parquet` with row groups sized from Arrow bytes per row (`_row_group_rows`): rewritten in 38 s at 0.94 GB peak RSS to 278 MB, in 45 row groups of about 295k rows and 9-30 MB of uncompressed Parquet each.
- `_fingerprint` of that rewritten claims file equals `_fingerprint` of its group file; computing both took 70 s at 1.55 GB peak RSS with `POLARS_MAX_THREADS=1` and 25 s at 3.65 GB with 4 threads.
- `_fingerprint` peak RSS grows with the Polars thread count, not the file size: hashing the claims `datavalue` column peaked at 0.25, 0.65 and 1.90 GB on 1, 4 and 16 threads for the 276 MB file, and at 0.21, 0.51 and 1.81 GB for half of it.

## Missing

- A record of the claims_labels row count per key on the Hub, before compaction.
- A lock against two `finalise` runs at once — two runs would both work on the first table not done.
- Recovery of `state/compact.jsonl`: without it, `compact_table` downloads a partly compacted repo again, and keys already compacted are rewritten and then skipped at commit by their sha256.
- A compaction path for groups pushed after a table is compacted — `push_group` would add `{key}/{group.name}.parquet` files beside the compacted ones, and `compact_table` skips a table whose last recorded stage is done.
- Per-key subsets (configs), sizes and row counts in the dataset cards.
- A way to push edited dataset card templates to repos that already have a README.md.
- Language fallback (MediaWiki fallback chains, `mul`) in the dataset cards.

## Divergence

- DESIGN.md "4. Push (grouped)" describes the Hub layout as one file per language per group and has no compaction step (DESIGN.md:92-126).
- DESIGN.md gives `GROUP_TARGET_COUNT` a default of 100 (DESIGN.md:104), and src/wikidata/config.py sets 30.
- docs/dataset_cards/claims_labels.md says a name appears once in each file whose source chunks use it, "so take the unique rows when you read more than one file" (lines 47-48) — compaction writes each row once per language.
- docs/dataset_cards/*.md give file paths as `{language}/chunks-NNNN-NNNN.parquet`, "one file per group of source chunks" — compaction writes one file per run of groups of about 500 MiB, most keys one file.
