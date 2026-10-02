# 2026-10-01: Building the datasets from an official Wikidata JSON dump (releases)

## Current State

### The two sources

- The six datasets on `main` (labels, descriptions, aliases, links, claims, claims_labels) are built from `philippesaade/wikidata`, whose card describes it as the Wikidata dump of 2026-05-07 with scholarly articles filtered out: 73,769,737 of 120,182,414 entities, in 7,449 Parquet files of 10,000 rows with columns `id, labels, descriptions, aliases, sitelinks, claims` holding JSON strings (chunk_0.parquet: 10,000 rows, 625 row groups).
- dumps.wikimedia.org/wikidatawiki/entities/ publishes the full JSON dump weekly in a directory named by its date (a release): 20260928 holds `wikidata-20260928-all.json.bz2` (103,272,886,026 bytes), `wikidata-20260928-all.json.gz` (156,315,459,742 bytes) and md5/sha1 sums; the directories between (20260930, 20260923) hold lexemes or RDF only.
- The official dump holds one entity per line between `[` and `]`, each line but the last ending in `,`; an entity has `type, id, labels, descriptions, aliases, sitelinks, claims, ns, title, pageid, lastrevid, modified` (Q31 in 20260928).
- The official dump stores labels and descriptions as `{lang: {language, value}}`, aliases as `{lang: [{language, value}]}`, sitelinks as `{site: {site, title, badges}}`; an official snak holds `snaktype, property, hash, datavalue: {value, type}, datatype` (main snaks without `hash`), items as `{entity-type, numeric-id, id}`; a reference holds `hash, snaks, snaks-order`; a statement also `type, id, qualifiers-order`.
- philippesaade stores `{lang: value}`, `{lang: [value]}` and `{site: {site, title}}`; its snaks hold `property, datavalue, datatype, property-labels`, its `datavalue` the value itself with `labels` beside an item's `id` and `unit-labels` beside a unit; its references are lists of `{P: [snak]}` and its statements `{mainsnak, rank, qualifiers, references}`; it has no `snaktype`, hashes, statement ids or types, orders, badges, `entity-type`, `numeric-id`, datavalue types or entity-level fields.
- The decisions for the official dumps: all entities, scholarly articles included, in the datasets and in the SAE; every field of the dump kept; on the Hub, `main` holds the latest release and every release is also a git tag (the 2026-05-07 data tagged `20260507` before replacement); the datasets first, then the SAE from them as run v2; the source file is `all.json.bz2`, decompressed with `lbzip2`.

### Release settings (src/wikidata/config.py)

- `WIKIDATA_RELEASE` sets `RELEASE`; with it set, the working directories (`state, data, results, audit, quarantine, staging, compact, hub`, and `dump`) are under `releases/{release}/`, the card figures and rendered cards under `docs/releases/{release}/`, the card templates are `docs/dataset_cards/dump/`, `Table` gains `ENTITIES` ("entities", partitioned unsplit), and `HUB_REVISION` is `build-{release}`; unset, every path, the table list and the Hub revision (`None`, main) are as before, and every module imports in both modes.
- `releases/` is in .gitignore.

### Dump to chunks (src/wikidata/dump.py)

- `latest-dump` prints the newest dated directory holding `wikidata-{date}-all.json.bz2`.
- dumps.wikimedia.org answers 403 to urllib's default User-Agent and 200 to a descriptive one; dump.py sends `wikidata-pq/0.1 (https://github.com/lmmx/wikidata-pq)` on every request, with which the listing (`latest-dump`: 20260928), the md5sums, a HEAD of the bz2 (103,272,886,026 bytes) and a range request (206) succeed.
- `download-dump` downloads `wikidata-{release}-all.json.bz2` to `releases/{release}/dump/` with HTTP range resumption from a `.part` file and a tqdm bar, from dumps.wikimedia.org or `WIKIDATA_DUMPS_URL`, and checks it against `wikidata-{release}-md5sums.txt`; 20260928 downloaded on the host and passed its md5, leaving 185 GB free on its disk.
- The six tables on `main` (from philippesaade, 73.8M entities) come to 35.5 GB on the Hub: claims 17.7, labels 6.5, descriptions 5.5, claims_labels 3.2, aliases 1.4, links 1.3.
- `split-dump` feeds the bz2 to `lbzip2 -dc` (its offset driving a tqdm bar), cuts the lines into batches of 10,000 entities, reshapes them in a process pool (cores minus two) with orjson, at most 4 GiB of decompressed lines in flight, and writes `releases/{release}/data/chunk_{N}.parquet` (zstd level 9, atomic rename) with one manifest line per chunk (`chunk, file, rows, bytes, first, last, mismatched_terms, fields`), then `split.done`; a rerun skips the chunks in the manifest.
- The reshaping keeps every field: labels and descriptions become `{lang: value}` and aliases `{lang: [value]}`, counting in `mismatched_terms` any entry whose `language` differs from its key or that has other fields; a snak's `datavalue` becomes the value with its type in `datavalue_type`; the entity's own fields go as JSON into an `entity` column; everything else (snaktype, hashes, statement ids and types, orders, reference hashes and snaks-order, badges, entity-type, numeric-id) passes through as the dump has it, with Wikibase's `[]` for an empty map read as `{}`.
- `split_manifest` halts when a chunk lists an entity or sitelink field outside `type, ns, title, pageid, lastrevid, modified` and `site, title, badges`, the fields the pipeline reads.
- A test dump of 2,500 entities (the first of 20260928, bzip2 behind an `lbzip2` shim) split into 3 chunks of 1,000 with 2 workers and a manifest; with a chunk dropped from the manifest, a rerun rewrote that chunk only. The first 5,604 entities gave `mismatched_terms` 0 and the fields above.
- With every field kept, 5,604 entities came to 32.9 MB of Parquet at zstd level 3 (17.1 MB without the fields the old source lacked); 3,000 entities came to 20.3 MB at level 3, 18.3 MB at level 9, 17.8 MB at 15 and 16.8 MB at 19 (28 s), against 18.6 MB as bz2; at level 9 the chunks of 20260928 come to about the bz2's 103 GB.

### Processing a release (process.py, partitioning/, main.py, initial.py)

- `setup_state` lists a release's chunks from the split manifest; the pull step checks each local chunk against the manifest's size (`dump.check_chunk`), prefetch is off, and group sizing reads the manifest's bytes.
- `process` reads a release's chunks from its data directory and, for a release, writes an entities table from the `entity` column (`ENTITY_SCHEMA`: type, ns, title, pageid as Int64, lastrevid as Int64, modified as string); the claims schema for a release has snaks of `snaktype, property, hash, datavalue, datavalue_type, datatype` with `entity-type` and `numeric-id` in the datavalue, statements with `type, id, qualifiers-order`, and references as `{hash, snaks, snaks-order}`; sitelinks decode with `badges`; an empty label lookup gives an empty claims_labels table.
- `claims_base` renames a statement's `id` and `type` to `statement_id` and `statement_type` before unnesting it beside the entity's `id`; a release's claims rows have `id, snaktype, property, hash, datavalue, datavalue_type, datatype, statement_type, statement_id, rank, references, qualifiers, qualifiers-order`.
- On the first 5,604 entities of 20260928 the claims schema inferred by polars-genson differs from the stored one only by fields absent from the sample; on the first 300, process wrote all seven tables, and the partition step gave 52,187 claims rows (52,145 `value`, 11 `somevalue`, 31 `novalue`; `statement_id` never null; main-snak `hash` always null; `numeric-id` on 18,267), links with `badges`, and entities with all six fields.
- The processed claims of the first 300 entities of 20260928 (4.9 MB, zstd, before partitioning) by column: statement ids 25.0%; snak and reference hashes 12.1% (qualifier snaks 7.6, references 3.9, reference snaks 0.6, main snaks 0.1); `numeric-id` 3.6% and `entity-type` 1.5%; `datavalue_type` 2.3%; `snaks-order` and `qualifiers-order` 2.2%; snaktype 1.1%; statement `type` 0.1%.
- A release of 601 entities in chunks of 200 went through split, `setup_state`, `check_chunk`, `process_and_partition` (chunks 0 and 1 to PARTITION) and `merge_group` into staging for all seven tables, claims_labels merging with no files.
- Processing the first 2,000 entities of 20260928 (the largest items: countries, cities) in one chunk ran out of the 4 GB container's memory when flattening claims for a test; the pipeline flattens per chunk in a subprocess on the host.

### Hub branch and promotion (src/wikidata/hub.py)

- Every call to the table repos in push/core.py, compact.py, sort_by_id.py, cards.py and main.py (`download-wikidata`) passes `revision=HUB_REVISION`; `push_group` creates the branch with `ensure_build_branch`, which on creation deletes every file but README.md and .gitattributes from it in one commit and leaves an existing branch as it is.
- `promote-release` (with `WIKIDATA_PREVIOUS_RELEASE`) tags each repo's `main` with the previous release where untagged, copies the branch's files to `main` with `CommitOperationCopy` (server-side for LFS files) and deletes `main`'s other data files, in commits of 1,000 operations, tags `main` with the release and deletes the branch; a repo already tagged with the release is skipped. Neither has been run against the Hub.

### claims_labels for a release (src/wikidata/claims_labels.py)

- `finalise` for a release compacts and sorts every table but claims_labels (creating each local copy directory for the sort to fill), then `build_claims_labels` collects the distinct (field, ref, id) of every snak's property (`property-labels`), item or property value (`labels`) and `http://www.wikidata.org/entity/` unit (`unit-labels`, id without the prefix) from the local claims, joins them per language with the local labels, writes each language's rows sorted by ref as `{lang}/chunks-0000-{last chunk}.parquet`, uploads them to the branch, and then compacts and sorts claims_labels; its stages (refs, written, uploaded) are in `state/claims_labels_build.jsonl`.
- On the 400 entities partitioned in the test release, the refs came to 10,111 labels, 2,456 property-labels and 39 unit-labels, and 430 languages were written (`en`: 81 labels rows and 1 unit-labels row, `kilogram`).

### Cards, recipes and the SAE

- docs/dataset_cards/dump/ holds the release templates: no `source_datasets`, a Releases section with `{{release}}` (the dump's URL, `main` as the latest, tags, the `20260507` tag for the philippesaade build), seven tables, the claims schema with every new column, `snaktype` in place of the "does not tell apart" note, deleted-property snaks kept with a null datatype, links with `badges`, claims_labels built from the release's labels, and an entities card; every table's card renders with empty metadata in release mode, and the six legacy cards render as before.
- The Justfile has `latest-dump`, `download-dump`, `split-dump`, `run-release`, `finalise-release` and `promote-release` recipes taking the release; README.md has a Releases section with the commands.
- sae/release.sh, sourced by sae/id_sets.sh, train.sh, export.sh, neighbours.sh and publish.sh, sets a release's local copy (`releases/{release}/hub`) and identifier-set folder (`sae/output/releases/{release}/`) from `RELEASE=` or the run's recorded `sae/output/$RUN/release` (written by train.sh), halting on a mismatch; unset, `hub/` and `sae/output/` as before.
- space/index.html shows each run's Wikidata date in the run note, from the run's `release` in runs.json, or 2026-05-07 without one.

## Missing

- A download, split and run of 20260928 on the host (`lbzip2` installed there), and the run's timings, disk use and memory per chunk.
- `promote-release` and the build branch exercised against the Hub repos.
- The card figures (card_stats) for the entities table and the release's cards rendered from a release's metadata.
- Run v2 of the SAE on 20260928's identifier sets, and its entry (with `release`) in sae/runs.json.
- Lexemes (a separate dump) are not read.
