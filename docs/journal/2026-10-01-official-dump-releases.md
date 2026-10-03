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
- In the same 300 entities' 52,187 statements, 20,318 references have 6,700 distinct hashes; written alone with zstd, the `references` column is 1.22 MB, the distinct references 0.60 MB and the statements' lists of reference hashes 0.19 MB.
- A release of 601 entities in chunks of 200 went through split, `setup_state`, `check_chunk`, `process_and_partition` (chunks 0 and 1 to PARTITION) and `merge_group` into staging for all seven tables, claims_labels merging with no files.
- Processing the first 2,000 entities of 20260928 (the largest items: countries, cities) in one chunk ran out of the 4 GB container's memory when flattening claims for a test; the pipeline flattens per chunk in a subprocess on the host.

### What philippesaade's copy left out (scripts/p31_survey.py)

- philippesaade's card says it filtered out "scholarly articles" and gives no rule.
- 20260928 split into 12,182 chunks, 121,815,642 entities; the bz2 was deleted after the split.
- The anti-join of 20260928's ids against those of the claims on `main`, by polars on the host, gave 48,185,485 ids.
- `scripts/p31_survey.py` reads `id, claims` from a release's chunks, takes each entity's P31 values from its main snaks by regex on the claims JSON (value snaks only, any rank), and marks the ids in the anti-join as missing; ids above the largest id in both (Q139677818) are new. It labels the classes it prints in English from the Wikidata API (`wbgetentities`, with dump.py's User-Agent), cached beside its output. On 20260928 with 8 workers it took 2 min 59 s.
- Of 120,070,317 old entities, 46,440,160 are missing (the card: 120,182,414 − 73,769,737 = 46,412,677); all 1,745,325 new ones are missing.
- 38 classes with at least 100 old entities have ≥ 99% of them missing; one or more of them is a P31 of 45,665,518 of the 46,440,160 old missing entities:

| Class | Label | Old entities | Missing |
|---|---|--:|--:|
| Q13442814 | scholarly article | 45,431,973 | 45,431,670 |
| Q871232 | editorial | 513,082 | 512,756 |
| Q2782326 | case report | 187,464 | 187,464 |
| Q815382 | meta-analysis | 109,685 | 109,685 |
| Q187685 | doctoral thesis | 104,180 | 104,177 |
| Q1348305 | erratum | 93,382 | 93,382 |
| Q1907875 | master's thesis | 47,231 | 47,225 |
| Q18918145 | academic journal article | 39,615 | 39,604 |
| Q45182324 | retracted paper | 23,941 | 23,940 |
| Q23927052 | conference paper | 19,956 | 19,955 |
| Q1402850 | field study report | 8,872 | 8,872 |
| Q15781350 | final project report | 5,340 | 5,336 |
| Q1266946 | thesis | 5,231 | 5,229 |
| Q7316896 | retraction notice | 3,947 | 3,947 |
| Q580922 | preprint | 2,902 | 2,886 |
| Q5246046 | academic publishing | 2,234 | 2,214 |
| Q69488 | MDMA | 1,866 | 1,866 |
| Q30749496 | diploma thesis | 1,535 | 1,535 |
| Q111475835 | bachelor's with honors thesis | 1,330 | 1,330 |
| Q56478376 | expression of concern | 768 | 764 |
| Q798134 | bachelor's thesis | 538 | 538 |
| Q114613919 | scientific note | 472 | 472 |
| Q92998777 | opinion paper | 471 | 471 |
| Q58901591 | comparative study | 471 | 470 |
| Q10885494 | academic conference paper | 466 | 466 |
| Q132115645 | standard analytical method | 413 | 413 |
| Q59387148 | research report | 373 | 373 |
| Q1385450 | dissertation | 294 | 294 |
| Q130709863 | book or chapter | 278 | 278 |
| Q54670950 | conference poster | 228 | 227 |
| Q51282918 | Doctor of Philosophy thesis | 159 | 159 |
| Q51282711 | Doctor of Clinical Psychology thesis | 150 | 150 |
| Q15706459 | research article | 146 | 145 |
| Q111475860 | postgraduate diploma thesis | 124 | 124 |
| Q51283092 | Master of Arts thesis | 121 | 121 |
| Q58900768 | consensus statement | 115 | 115 |
| Q58897583 | comment | 112 | 112 |
| Q58898636 | evaluation study | 107 | 107 |

- Classes partly missing (old entities, share missing): scientific publication (Q591041) 13,209, 93.4%; geological map (Q193842) 7,351, 95.3%; scholarly conference abstract (Q58632367) 7,111, 89.0%; technical report (Q3099732) 5,712, 72.6%; book review (Q637866) 26,504, 43.6%; newsletter (Q264238) 1,276, 48.4%; catalog (Q2352616) 2,810, 34.8%; presentation (Q604733) 1,206, 32.0%; patent (Q253623) 1,442, 28.4%; blog post (Q17928402) 3,619, 28.3%; scholarly chapter (Q21481766) 11,041, 23.3%; review (Q265158) 2,533, 20.5%; edition of a translation (Q21112633) 3,025, 18.3%; Wikimedia module (Q15184295) 64,567, 7.0%; written work (Q47461344) 176,126, 6.7%; publication (Q732577) 33,888, 5.3%.
- Classes with < 1% missing include human (Q5) 2,371 of 13,494,455, Wikimedia category (Q4167836) 1,136 of 5,759,183, Wikimedia template (Q11266439) 1,458 of 827,542 and version, edition or translation (Q3331189) 1,165 of 821,995.
- Of the 774,642 old missing entities without any of the 38 classes, 714,789 have no P31 value snak; the most common P31 of the rest are scientific publication (12,234), scholarly conference abstract (6,290), Wikimedia module (4,549), technical report (3,996) and human (2,358).

### The scholarly set (src/wikidata/scholarly.py, dump.py `route`)

- The decision: a release is published as two sets of tables, its scholarly works as `permutans/wikidata-scholar-{tbl}` (new repos) and everything else as `permutans/wikidata-{tbl}` (the existing repos, republished).
- `SCHOLARLY_CLASSES` holds 43 classes: the 37 scholarly classes among the 38 at ≥ 99% missing above, and scientific publication, geological map, scholarly conference abstract, technical report, book review and scholarly chapter. An entity is scholarly if any of its P31 value snaks is one of them.
- The P31 regex anchored on a fixed key order (`snaktype, property, datatype, datavalue`) matched the dump's main snaks but none fetched from the Wikidata API, whose main snaks also have `hash`; `P31_SNAK` allows any keys without a brace between `"mainsnak":{` and `datavalue`.
- `WIKIDATA_SCHOLAR=1` (with `WIKIDATA_RELEASE`) selects the scholarly set: working directories under `releases/{release}-scholar/`, card figures and rendered cards under `docs/releases/{release}-scholar/`, repos `wikidata-scholar-{tbl}`, the same Hub branch name; download, split and route refuse it.
- `route-release` (dump.py `route`, 8 workers) reads each split chunk, writes its scholarly and other rows to `routed/chunk_{N}.parquet` in the two sets' data directories, logs both parts in `route.jsonl`, then deletes the chunk. Once all are logged, each set's manifest numbers its non-empty parts from 0 in split order (with `split_chunk`), as the pipeline's groups are ranges of consecutive chunks; the split's manifest is kept as `manifest.split.jsonl`, the parts are moved to their numbers, and both sets get `split.done` and `route.done`. `split_manifest` halts without `route.done`.
- A test dump of 16 entities fetched from the Wikidata API (Q42, two DNA-structure papers, "Attention Is All You Need", Q5, Nature, ..., P31, P1433), split into 8 chunks of 2: routed to 12 entities in 7 chunks and 4 scholarly in 3; a split chunk of only scholarly works left no main chunk and the later main chunks were renumbered, one without any left no scholarly chunk. With a worker failing on the sixth chunk, and with the move stopped after three parts, a rerun gave the same chunks.

### polars-genson on the routed sets (0.9.3)

- Processing the 4 scholarly entities of the routed test release (3 chunks) halted in chunk 0's links: genson's forced-map values of `sitelinks` came back as objects (`badges` as null), and `json_decode` of them as strings failed.
- With `force_field_types={"sitelinks": "map"}` and no decode schema, a map's values come back as JSON strings when one value's `badges` is `[]` and another's `["Q…"]` (in one row or in two), and as objects when every `badges` is empty, every one non-empty (any lengths), or absent in some values. An empty array's schema has no `items`, and `unify_array_schemas` (genson-core/src/schema/map_inference/unification.rs) returns None on an array schema without `items`, so `forced_map_value_schema` falls back to string.
- Under the claims options (`map_threshold: 0`, `unify_maps`, `force_scalar_promotion` with `datavalue`), qualifier snaks were inferred as maps of strings in chunks 0 and 2 of the scholarly test set (`qualifiers` values `List(List(Struct{key, value: String}))`) and as records in chunk 1. Chunk 0's 8 qualifier snaks are all string-valued (`snaktype: value`, `datatype: string`). Reproduced with one statement: string-valued qualifier snaks only give a map of strings; with an item-valued qualifier snak in another row, a record; a philippesaade-style snak (with a `property-labels` object), a record. In `rewrite_objects` (map_inference.rs) the homogeneity check sees the snak's six fields all as strings (`datavalue` before promotion) and rewrites it as a map; the `force_scalar_promotion` guard applies only when recursing into `datavalue` itself.
- The first 300 entities of 20260928 (processed earlier) have item-valued qualifier snaks and mixed badges, which gave records and JSON strings, the shapes process.py expects.
- A workaround decoding release sitelinks with orjson instead of genson was written and reverted (not committed); the fixes go in polars-genson, on the branch `map-inference-release-dumps`.
- An empty array's schema there is `{"type": "array", "items": {}}` (no `items` where an array was seen only empty under a record that lacks it elsewhere). polars-genson ba8037f (branch `map-inference-release-dumps`): `unify_array_schemas` skips missing or empty `items`, returns the one remaining items schema as it is (unifying it alone made it nullable), and an array with none as `{"type": "array"}`; `rewrite_objects` keeps an object holding a `force_scalar_promotion` field as a record (recursing into its fields), unless the object is itself such a field. New tests: forced_map.rs (empty with non-empty `badges` in one row and in two: values a record with `badges` an array of strings; only empty `badges`, missing in one value: nullable array), promoted_field_record.rs (string-only qualifier snaks: a record with `datavalue__string`; with an item-valued one in another row: a record with both). The genson-core and genson-cli test suites pass (with `avro`), with no snapshot changes.
- The fix merged as polars-genson #213 (0fcf6a2), released as polars-genson 0.9.4 (tag `py-0.9.4`) and genson-core 0.9.3 (with genson-cli 0.9.3, which needs prune's `normalise_values_pruned` from #212: crates.io had genson-core 0.9.2 from before prune, under the same number as the local one). pyproject.toml requires polars-genson>=0.9.4.
- Publishing the crates: `ship-rust`'s dry run failed compiling the packaged genson-cli 0.9.3 (`unresolved imports genson_core::normalise::normalise_values_pruned, prune_schema`). The last Rust bump (0945ec5, 27 Sep, before prune #212) had set genson-core 0.9.2 and genson-cli 0.9.3; crates.io had genson-core 0.9.2 (pre-prune) and genson-cli up to 0.9.2, so the dry run verified genson-cli against the published pre-prune genson-core. `release-plz update` bumped genson-core 0.9.2 → 0.9.3 (API compatible; genson-cli left, already differing from the registry), committed as 59e9cd5. `publish-rust --dry-run` then failed on `genson-core = "^0.9.3"` not on crates.io (a dry run skips the upload genson-cli depends on); `publish-rust` published genson-core 0.9.3 and genson-cli 0.9.3; polars-jsonschema-bridge 0.9.0 was already published.
- The Python wheel did not build in the 4 GB container (`maturin build --release`, cargo exit 101 with no compiler error).
- process.py's `normalise_sitelinks` decodes the map values as `SITELINK_SCHEMA` records (in place of strings then `json_decode`), as the fixed genson unifies them; untested against the fixed wheel.

- `just release 20260928 20260507` halted at once in `split_manifest`: `{'entity': ['datatype'], 'sitelink': []}`. Properties have an entity-level `datatype` (P31, P1433: `wikibase-item`); the 5,604 entities checked before were items. `ENTITY_SCHEMA` and `EXPECTED_FIELDS` gained `datatype` (after `type`; null for items), and both entities cards a row for it; on the routed test release the manifest check passes and P31, P1433 decode with `wikibase-item`, Q42 with null.

- With polars-genson 0.9.4, `just release` processed the scholarly set's chunk 0 (`Processing 1 strings` in the claims profile) and its partitions joined the open group; chunk 1 halted in `normalise_map_direct` for aliases: `Failed to create Parquet writer: Arrow: Parquet does not support writing empty structs`. Reproduced (0.9.3) with two rows whose `aliases` is `{}`; with one non-empty row the output is a map. genson's schema for an object empty in every row is `{"type": "object"}` (a record without fields) with or without `map_threshold: 0`, and with `force_field_types` map `additionalProperties: {"type": "string"}`.
- polars-genson 861b4ab (branch `map-inference-empty-objects`): with `map_threshold: 0` (not at a root with `no_root_map`), an object (or nullable object) without properties or `additionalProperties` becomes a map with `{"type": "null"}` values, and a forced map without properties gets null values in place of the string fallback. New tests (empty_objects.rs, `avro` and `parquet`): threshold 0 and forced give null values, the default threshold leaves `{"type": "object"}`, and the typed output of two `{}` rows (kv encoding) writes to Parquet and reads back 2 rows. The genson-core and genson-cli suites pass; the two `claims_c0_p24_ddminv2` snapshots (forced `labels` map, empty in every row there) changed from string to null values, accepted with `cargo insta accept`.
- process.py: `is_acceptable_diff` accepts `type_changes` and `values_changed` where the inferred type is `Null` or a record with a subset of the stored one's fields (a record with an extra field, or `String` in place of `List(String)`, still halts); `normalise_map_direct` casts its output to the expected schema. Polars casts `List(Struct{key, value: Null})` to the stored map type both eagerly and in the claims' `scan_parquet` with a target schema (a Null-valued `qualifiers` map in a struct read as the stored snak list). pyproject.toml requires polars-genson>=0.9.5.

- polars-genson PR for 861b4ab: a PR body written to the polars-genson root as PR_BODY.md (untracked, for copying). The user released it and updated wikidata's uv.lock (left uncommitted in the checkout).

### The run of 20260928 (host)

- `route-release 20260928` ran on the host before `just release` (its counts were not recorded here).
- `just release 20260928 20260507`, first attempt: halted in `split_manifest` on the entity field `datatype` (above). Second attempt: the scholarly set's chunk 0 processed and partitioned; chunk 1 halted on the empty aliases struct (above). Third attempt (polars-genson with 861b4ab): chunk 1 (`aliases` inferred `List(Struct({'key': String, 'value': Null}))`) and chunks 2 to 4 processed, partitioned and their sources deleted, at about 4.5 s each; each held one entity (`Processing 1 strings` in the claims profile). The open group stood at 5 chunks, 0.00 GB, against a threshold of 25.00 GB falling to 22.48 GB. Left running overnight on 2026-10-03.
- Fourth attempt, overnight: scholarly chunks 0 to 56 processed (a few entities each; the open group 57 chunks, 0.01 GB); chunk 57 halted in `normalise_sitelinks`: `error deserializing value "String(...specieswiki...)" as struct`. Chunk 57 has 49 entities, 2 with sitelinks (each one `specieswiki` link, `badges` empty), 47 with `{}`.
- With the installed polars-genson 0.9.5 (its abi3 extension loaded from the host venv into the scratchpad's Python 3.13), `infer_json_schema` on chunk 57's sitelinks (forced map, `wrap_root`, avro) gives `values: string`; its 2 non-empty rows alone give a record; 0.9.5 gives records for the #213 and #214 cases. With 1, 2, 3 or 5 `{}` rows before the 2 values: a record; with 10, 20 or 47: string. genson-cli and genson-core from source give a record for the chunk as one NDJSON string (sequential).
- From `PARALLEL_THRESHOLD` (10) strings, genson-core builds each string's schema on its own and runs `apply_force_field_types` on it before merging; that pass set every forced map's `additionalProperties` to `{"type": "string"}`, discarding the value schema, so every forced map came out `map<string>` from 10 rows (a 12-row map of integers too) and with its values' type below. The philippesaade build's 10,000-row chunks therefore had string sitelink values, which process.py decoded with `json_decode`. With the pre-merge pass keeping the value schema, merging per-string maps kept the first `additionalProperties` seen: genson-rs's object strategy treated it as an extra keyword (first wins), so `badges: []` and `badges: ["Q…"]` from two strings merged to `items: {}`.
- polars-genson 8d4f669 (branch `forced-map-parallel-values`): the pre-merge pass gives a forced map with keys its `forced_map_value_schema` and leaves one without keys as it is; the object strategy merges an `additionalProperties` schema as a node (in `add_schema`, `add_schemas`, `add_schemas_par`). Four parallel-path tests (12+ rows: records, empty rows with empty and non-empty `badges`, all empty giving null, integers) failed with `string` before and pass; the genson-core and genson-cli suites pass with no snapshot changes. A PR body is in the polars-genson root as PR_BODY.md.
- The first split chunks are the lowest ids (old, well-known items), with few scholarly works each, so the scholarly set's first chunks are small; its later chunks hold up to 10,000.
- scripts/release_eta.py (daedfc0) gives a release's progress across both sets from each set's manifest (bytes per chunk) and claims audit files (one per partitioned chunk, by mtime): chunks and GB done per set, the rate in GB of source per hour since the in-progress set's first chunk, and the hours left for both sets' processing. On the routed test release with one audit file it printed both sets' counts and a rate.

### Finalising two sets (main.py, claims_labels.py, hub.py, cards)

- `process-wikidata` for a release no longer runs `finalise` at its end; the `release` recipe runs `run-release` scholar then main, then `finalise-release` main, scholar, main, then `promote-release` main and scholar.
- `finalise` for a release compacts and sorts claims first, then collects claims_labels' refs from its local copy (stage `refs`, `collect_refs_stage`) and deletes that copy (`CLEAN_UP_LOCAL`), then the other tables one at a time, compacted then sorted. claims_labels joins the refs with the labels of both sets' local copies (`labels_dirs`: `releases/{release}/hub/labels` and `releases/{release}-scholar/hub/labels`, per language, concatenated); finalise prints that claims_labels waits and returns when the other set's labels are not sorted (`sort.jsonl` stage `done`). After claims_labels, figures and cards it writes `state/finalise.done`; the build directory is removed once uploaded.
- In a test of `write_groups` with two sets' labels directories (`en, fr` and `en, de`), refs from both got their labels in `en`, `fr` and `de`, and a ref in neither got none.
- `promote-release` refuses a set without `finalise.done`, and with `WIKIDATA_SCHOLAR=1` takes no previous release; a repo whose `main` has data files and no previous release given halts.
- docs/dataset_cards/scholar/ holds the scholarly set's templates, generated from dump/: `wikidata-scholar-*` in links, examples and titles ("Wikidata Scholarly Labels"), intros naming the scholarly works (linking scholarly.py), the main set's tables named in a note, no `20260507` tag paragraph, examples on Q1895685; claims_labels' card names both sets' labels. The dump/ templates gained a note naming the `wikidata-scholar-*` tables, "but its scholarly works" in the Releases paragraph, and claims_labels' card both sets' labels. `CARD_TEMPLATES_DIR` is `scholar/` with `WIKIDATA_SCHOLAR=1`; card examples match `wikidata-scholar-` paths and configs. Every table's card renders with empty metadata in the legacy, main and scholarly modes.
- The scholarly set's card samples are Q1895685 (Watson and Crick, 1953; labels en, fr, de, an en alias, en, fr and de wiki links) and Q30249683 ("Attention Is All You Need"), and claims_labels P31, Q13442814 and Q180445 (Nature).
- The Justfile recipes `run-release`, `finalise-release` and `promote-release` take a set (`main` by default, or `scholar`, any other value an error), set as `WIKIDATA_SCHOLAR`; `just --dry-run` shows the expansions. README.md's Releases section lists the order.

### Hub branch and promotion (src/wikidata/hub.py)

- Every call to the table repos in push/core.py, compact.py, sort_by_id.py, cards.py and main.py (`download-wikidata`) passes `revision=HUB_REVISION`; `push_group` creates the branch with `ensure_build_branch`, which on creation deletes every file but README.md and .gitattributes from it in one commit and leaves an existing branch as it is.
- `promote-release` (with `WIKIDATA_PREVIOUS_RELEASE`) tags each repo's `main` with the previous release where untagged, copies the branch's files to `main` with `CommitOperationCopy` (server-side for LFS files) and deletes `main`'s other data files, in commits of 1,000 operations, tags `main` with the release and deletes the branch; a repo already tagged with the release is skipped. Neither has been run against the Hub.

### claims_labels for a release (src/wikidata/claims_labels.py)

- `finalise` for a release compacts and sorts every table but claims_labels (creating each local copy directory for the sort to fill), then `build_claims_labels` collects the distinct (field, ref, id) of every snak's property (`property-labels`), item or property value (`labels`) and `http://www.wikidata.org/entity/` unit (`unit-labels`, id without the prefix) from the local claims, joins them per language with the local labels, writes each language's rows sorted by ref as `{lang}/chunks-0000-{last chunk}.parquet`, uploads them to the branch, and then compacts and sorts claims_labels; its stages (refs, written, uploaded) are in `state/claims_labels_build.jsonl`.
- On the 400 entities partitioned in the test release, the refs came to 10,111 labels, 2,456 property-labels and 39 unit-labels, and 430 languages were written (`en`: 81 labels rows and 1 unit-labels row, `kilogram`).

### Cards, recipes and the SAE

- docs/dataset_cards/dump/ holds the release templates: no `source_datasets`, a Releases section with `{{release}}` (the dump's URL, `main` as the latest, tags, the `20260507` tag for the philippesaade build), seven tables, the claims schema with every new column, `snaktype` in place of the "does not tell apart" note, deleted-property snaks kept with a null datatype, links with `badges`, claims_labels built from the release's labels, and an entities card; every table's card renders with empty metadata in release mode, and the six legacy cards render as before.
- The Justfile has `latest-dump`, `download-dump`, `split-dump`, `route-release`, `run-release`, `finalise-release`, `promote-release` and `release` recipes taking the release (and the set, see above); README.md's Releases section lists the order of `release`'s steps.
- sae/release.sh, sourced by sae/id_sets.sh, train.sh, export.sh, neighbours.sh and publish.sh, sets a release's local copy (`releases/{release}/hub`) and identifier-set folder (`sae/output/releases/{release}/`) from `RELEASE=` or the run's recorded `sae/output/$RUN/release` (written by train.sh), halting on a mismatch; unset, `hub/` and `sae/output/` as before.
- space/index.html shows each run's Wikidata date in the run note, from the run's `release` in runs.json, or 2026-05-07 without one.

## Missing

- The rest of `just release 20260928 20260507`: the scholarly set's chunks from 5 on (running), the main set's processing, both sets' compaction, sort, claims_labels and cards, and promotion (`promote-release` and the build branch have not yet run against the Hub repos).
- The run's timings, disk use and memory per chunk; route-release's counts for 20260928.
- A first chunk of the main set (10,000 rows, parallel inference) processed with a polars-genson release of 8d4f669.
- What the 714,789 old missing entities without a P31 value are, and the rule (if any) behind the partly missing classes.
- The card figures (card_stats) for the entities table and the release's cards rendered from a release's metadata.
- Run v2 of the SAE on 20260928's identifier sets, and its entry (with `release`) in sae/runs.json.
- Lexemes (a separate dump) are not read.
