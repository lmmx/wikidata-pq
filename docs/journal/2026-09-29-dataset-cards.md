# 2026-09-29: Dataset cards with subsets, sizes and language fallback

## Current State

### Cards on the Hub

- `docs/dataset_cards/{table}.md` holds one static card per table, and `_ensure_dataset_card` uploads it as README.md only when the repo has no README.md (src/wikidata/push/core.py:96-110) — edits to a template after a repo's first push do not reach the Hub.
- Each card's YAML front matter declares one config, `default`, with `data_files: "*/*.parquet"` (claims: `"all/*.parquet"`), and no `dataset_info` — the Hub shows no subsets and no per-key sizes.
- The Hub shows the total size of a repo (e.g. 17.8 GB for wikidata-claims_labels) and not the size of a key (claims_labels/en: 738 MB before compaction).
- The cards give file paths as `{language}/chunks-NNNN-NNNN.parquet`, "one file per group of source chunks" (e.g. docs/dataset_cards/labels.md:23) — compaction writes one file per run of groups of about 500 MiB (docs/journal/2026-09-29-hub-layout-and-compaction.md).
- docs/dataset_cards/claims_labels.md:47-48 says a name appears once in each file whose source chunks use it, "so take the unique rows when you read more than one file" — compaction writes each claims_labels row once per language.
- No card mentions language fallback, `mul`, or regional codes such as `en-gb`.
- The README.md of each of the six repos on the Hub equals its template in docs/dataset_cards (hub/{table}/README.md, 2026-09-30).
- README.md describes a claims table with one row per claim per language and `language=en/` directory partitioning, and DESIGN.md has no compaction step and gives `GROUP_TARGET_COUNT` a default of 100 where src/wikidata/config.py sets 30.

### Inputs

- Compaction writes `docs/dataset_cards_metadata.json`, `{table: {key: {"files", "bytes", "rows"}}}`, from the files it uploads (`write_metadata`, src/wikidata/compact.py); all six tables are in it after compaction (docs/journal/2026-09-29-hub-layout-and-compaction.md).
- Keys per table: labels 621, descriptions 594, aliases 587, links 955 (sites), claims 1 (`all`), claims_labels 618.
- `en` has the most rows of every language-split table in the audit sidecars (labels 48,749,798, aliases 14,481,185, descriptions 58,975,012, claims_labels 115,269,144 before deduplication), and `enwiki` of links (10,256,657).
- `mul` in the audit sidecars: labels 18,921,607 rows (3rd of 621 keys), aliases 723,685 (26th of 587), claims_labels 27,353,427 before deduplication (61st of 618), and no descriptions key — Wikidata's `mul` code applies to labels and aliases only (https://www.wikidata.org/wiki/Help:Default_values_for_labels_and_aliases).
- `scripts/card_stats.py` (since replaced by src/wikidata/card_stats.py) on the local copy (hub/, which matches the metadata JSON in files, bytes and rows for every key of every table):
  - Regional codes (keys with a hyphen): labels 106, descriptions 102, aliases 97, claims_labels 105 (e.g. `en-gb`, `pt-br`, `zh-hant`, `sr-el`, `de-formal`).
  - labels, Q ids: 74,429,805 with a label; 48,736,354 (65.5%) with `en`, 18,921,222 (25.4%) with `mul`, 10,020,338 (13.5%) with `mul` and no `en`, 15,673,113 (21.1%) with neither. P ids: 13,448; 13,444 with `en`, 385 with `mul`, 4 with `mul` and no `en`.
  - descriptions, Q ids: 67,735,222 with a description; 58,961,668 (87.0%) with `en`, 8,773,554 (13.0%) without; no `mul` key. P ids: 13,427; 13,344 with `en`.
  - aliases, Q ids: 15,704,236 with an alias; 8,721,660 (55.5%) with `en`, 540,446 (3.4%) with `mul`, 253,760 (1.6%) with `mul` and no `en`, 6,728,816 (42.8%) with neither. P ids: 12,328; 7,859 with `en`, 266 with `mul`, 38 with `mul` and no `en`.
- On `chunk_1237`, 135 of 10,000 entities carry a `mul` label and 2 carry only `mul` (docs/journal/2026-09-26-claims-label-maps-and-language-rule.md).
- The sort stage replaced every table's files with `{key}/part-{i}-of-{n}.parquet`, sorted by `id` (claims_labels by `ref`) in string order across a key's files, and rewrote the metadata JSON from them (docs/journal/2026-09-30-sort-by-id.md): one file per key except labels/en (2) and claims/all (34).

### Language fallback (checked 2026-09-30)

- Wikibase builds a term fallback chain as: the language, its variants (for languages with converters, such as `zh` and `sr`, with script conversion), its explicit MediaWiki fallbacks each with their variants, then `mul`, then `en` (`addImplicitFallbacksToChain` adds `mul` then `en`, lib/includes/LanguageFallbackChainFactory.php in mediawiki-extensions-Wikibase master; explicit fallbacks from `LanguageFallback::getAll(..., STRICT)`).
- The design task for `mul` gives the same order, "Translatewiki fallback chain > mul > en" (https://phabricator.wikimedia.org/T285156), and the Wikidata help page's query example requests `[AUTO_LANGUAGE],mul,en` (https://www.wikidata.org/wiki/Help:Default_values_for_labels_and_aliases).
- The Wikidata API returns each language's explicit fallbacks: `action=query&meta=languageinfo&liprop=code|fallbacks&licode=*` (1,486 codes on 2026-09-30, MediaWiki 1.47.0-wmf.21); no code lists `mul` among its fallbacks.
- Examples from that API: `en-gb` and `en-ca` to `en`; `pt-br` to `pt`; `de-formal`, `de-ch`, `gsw` and `lb` to `de`; `sr-el` to `sr-latn`, `sr`; `zh-hant` to `zh-tw`, `zh-hk`, `zh`, `zh-hans`; `en-us`, `es-419` and `simple` have none.
- Every key of labels, descriptions, aliases and claims_labels is a code in that API's list; keys with explicit fallbacks: labels 315 of 621, descriptions 312 of 594, aliases 305 of 587, claims_labels 314 of 618; seven keys' chains end in `en` explicitly (`en-gb`, `en-ca`, `sco`, `jam`, `pih`, `gpe`, `bi`).

### Renderer (src/wikidata/cards.py), 2026-09-30

- `render_card` fills a template's placeholders from the metadata JSON and docs/dataset_cards_stats.json: `{{configs}}` with one quoted config per key plus `all` (`default: true` on `en`, `enwiki` for links, `all` for claims and for a table not yet in the metadata JSON), `{{default}}` with that config, `{{key_examples}}` with the three largest keys and the largest with a hyphen, `{{sizes}}` with totals, the 10 largest keys and every key in a `<details>` block, `{{sample}}` with the stats file's sample rows, and `{{languages}}` with the coverage of `en` and `mul` among `Q` ids, the `mul` paragraph when the table has a `mul` key and a "no `mul` subset" sentence when it has none; `write_cards` writes all six to docs/dataset_cards/rendered; `render-cards` (`just cards`) runs it.
- `render_card` raises when a table's stats entry was computed from other inputs (`inputs_sha256`: sha256 of the table's metadata JSON entry and its `SAMPLES`, `STATS_COLUMN` and `PREFIXES` entries) or is missing, when a template has an unknown placeholder or one with nothing to fill it, when a table has a `mul` key and no `mul` text, and when an example path (`wikidata-{table}/{key}/*`) or `load_dataset` config names a key not in that table's metadata (tested with altered in-memory metadata: a changed row count, `labels/fr` removed, `descriptions/mul` added, `claims/all` renamed; `labels/mul` removed switches the text to "no `mul` subset").
- `card-stats` (`just card-stats`, `update_stats` in src/wikidata/card_stats.py) recomputes each table's entry (all but claims) whose `inputs_sha256` differs from the current one: it refuses a local copy whose files, bytes or rows per key differ from the metadata, reads the rows of fixed ids (`SAMPLES`: Q42 and Q5 in `en`, `fr`, `de` and `mul` for labels, in `en` for descriptions and in `en` and `mul` for aliases; Q42 in `enwiki`, `frwiki` and `dewiki` for links; P31, Q5 and Q11573 by `ref` in `en` for claims_labels) from fixed keys, and counts `Q` and `P` ids with any row, an `en` row, a `mul` row, and `mul` and no `en`; `finalise` runs it after the sort stage and before rendering.
- The templates hold no figures, sample rows or key names from the data outside the placeholders and the example subsets the renderer checks; the fallback examples (`en-gb` to `en`, ...) come from MediaWiki's fallback lists, not from the tables.
- `finalise` renders the cards after the sort stage and `push_card` uploads each one ("Update dataset card") unless it equals the repo's README.md; `_ensure_dataset_card` (first push of a new repo) uploads the rendered card instead of the template.
- `no` (Norwegian) is a key of labels, descriptions, aliases and claims_labels, and loads as the boolean false when unquoted in YAML.
- Rendered configs, read back with `yaml.safe_load`: labels 622, descriptions 595, aliases 588, links 956, claims 1, claims_labels 619, every name a string.
- `datasets` 5.0.1 on a local directory holding the rendered labels card and the hub/labels key folders: `get_dataset_config_names` returns 622 names including `no` and `all`, `load_dataset_builder` picks `en` with its two part files, and `load_dataset(path, "no")` and `"be-tarask"` load 178,302 and 665,449 rows.
- The card's fallback example on hub/labels with the chain `be-tarask`, `be` returns 1,004,522 rows, one per id of either key, with the `be-tarask` row for each of that key's 665,449 ids; `sr-latn`, a fallback of `sr-el`, has no labels key, so the card's `chain` keeps only languages with a subset, listed with `HfFileSystem().ls` (directories only), and the languageinfo API returns 403 to urllib's default User-Agent, so the example sets one.
- Lookups on the sorted local copy with Polars 1.44.2 `scan_parquet(...).filter(...)`: `id == "Q42"` on labels/en 45 ms, `ref == "P31"` on claims_labels/en 25 ms, and the 337 claims rows of Q42 from the 34 claims files in 86 ms.
- The templates give the file layout as `{key}/part-{i}-of-{n}.parquet` sorted by `id` (`ref` for claims_labels), subsets and loading by config name, the card_stats coverage figures and `mul` for labels and aliases, the fallback order for labels, descriptions and claims_labels, and no "take the unique rows" advice (claims_labels join example without `.unique()`).

## Design

Cards are rendered from the templates and the metadata JSON, as the last stage of `finalise-wikidata`, after every table is compacted and sorted.

- Front matter: one config per key, `config_name: {key}`, `data_files: "{key}/*.parquet"`, and one config `all` with `data_files: "*/*.parquet"`.
- Default config: `en` for labels, descriptions, aliases and claims_labels, and `enwiki` for links — `en` is the largest key of each language-split table and present in all four, and `mul` is absent from descriptions and holds only the labels an item shares across languages. claims keeps its single config.
- Body: a table per card of each key's files, size and rows from the metadata JSON, sorted by size, with the largest keys inline and every key in a collapsible `<details>` block.
- Body text: file layout as the sort stage writes it (`{key}/part-{i}-of-{n}.parquet`, rows sorted by `id`, or `ref` for claims_labels, in string order within and across a key's files, so a filter on the id skips row groups by their statistics), loading by config name, and no "take the unique rows" advice for claims_labels.
- Language section in labels, descriptions, aliases and claims_labels: `mul` (labels and aliases only) with the card_stats coverage figures, Wikibase's fallback order (the language, its MediaWiki fallbacks, `mul`, `en`), the languageinfo API query for a language's fallbacks, that script conversion between variants (e.g. `zh-hans` to `zh-hant`) is not applied to the stored rows, and a Polars example that coalesces a label along a chain and keeps the `language` it came from; claims notes that monolingual-text values carry their own language.
- Templates: `docs/dataset_cards/{table}.md` keep the front matter without `configs` and mark every part that depends on the data with a placeholder; `src/wikidata/cards.py` renders them, and a card renders only from figures computed against the current metadata.
- Rendered cards are written to `docs/dataset_cards/rendered/{table}.md` and committed, so a card's changes show in git; `render-cards` (`just cards`) renders them without pushing, for review.
- Push: each rendered README.md is uploaded only when it differs from the repo's current README.md, one commit per repo, as the last stage of `finalise`; `_ensure_dataset_card` uploads the rendered card, as before only when the repo has no README.md.
- Subsets are the configs above, one per key directory, so each language (or site, for links) is a subset on the Hub and loads by name; links has the most, 956 with `all`, under the dataset viewer's limit of 3,000 configs (its `DatasetWithTooManyConfigsError` message, "The maximum number of configs allowed is 3000", on datasets over it such as facebook/flores).
- Card figures are measured on a local copy of the Hub repos: `download-wikidata` (`just download`) downloads each table's repo to `hub/{table}` with `snapshot_download`, and `card-stats` computes the figures from it into docs/dataset_cards_stats.json.
- `dataset_info` (per-config `splits` with `num_examples` and `num_bytes`, and `download_size`) is not written; row counts and sizes go in the card body. `num_bytes` is the in-memory Arrow size of a config, which needs every key read into memory once, and `datasets` 5.0.1 raises `NonMatchingSplitsSizesError` on load when a split's `num_examples` differs from the rows it read, under the default `verification_mode` `BASIC_CHECKS` (datasets/builder.py:967-968, datasets/utils/info_utils.py:62-76), so the counts must match the files exactly.

## Missing

- A push of the rendered cards to the Hub (the next `finalise` run pushes all six).
