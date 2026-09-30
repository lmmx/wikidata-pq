# 2026-09-29: Dataset cards with subsets, sizes and language fallback

## Current State

### Cards on the Hub

- `docs/dataset_cards/{table}.md` holds one static card per table, and `_ensure_dataset_card` uploads it as README.md only when the repo has no README.md (src/wikidata/push/core.py:96-110) — edits to a template after a repo's first push do not reach the Hub.
- Each card's YAML front matter declares one config, `default`, with `data_files: "*/*.parquet"` (claims: `"all/*.parquet"`), and no `dataset_info` — the Hub shows no subsets and no per-key sizes.
- The Hub shows the total size of a repo (e.g. 17.8 GB for wikidata-claims_labels) and not the size of a key (claims_labels/en: 738 MB before compaction).
- The cards give file paths as `{language}/chunks-NNNN-NNNN.parquet`, "one file per group of source chunks" (e.g. docs/dataset_cards/labels.md:23) — compaction writes one file per run of groups of about 500 MiB (docs/journal/2026-09-29-hub-layout-and-compaction.md).
- docs/dataset_cards/claims_labels.md:47-48 says a name appears once in each file whose source chunks use it, "so take the unique rows when you read more than one file" — compaction writes each claims_labels row once per language.
- No card mentions language fallback, `mul`, or regional codes such as `en-gb`.
- README.md describes a claims table with one row per claim per language and `language=en/` directory partitioning, and DESIGN.md has no compaction step and gives `GROUP_TARGET_COUNT` a default of 100 where src/wikidata/config.py sets 30.

### Inputs

- Compaction writes `docs/dataset_cards_metadata.json`, `{table: {key: {"files", "bytes", "rows"}}}`, from the files it uploads (`write_metadata`, src/wikidata/compact.py); all six tables are in it after compaction (docs/journal/2026-09-29-hub-layout-and-compaction.md).
- Keys per table: labels 621, descriptions 594, aliases 587, links 955 (sites), claims 1 (`all`), claims_labels 618.
- `en` has the most rows of every language-split table in the audit sidecars (labels 48,749,798, aliases 14,481,185, descriptions 58,975,012, claims_labels 115,269,144 before deduplication), and `enwiki` of links (10,256,657).
- `mul` in the audit sidecars: labels 18,921,607 rows (3rd of 621 keys), aliases 723,685 (26th of 587), claims_labels 27,353,427 before deduplication (61st of 618), and no descriptions key — Wikidata's `mul` code applies to labels and aliases only (https://www.wikidata.org/wiki/Help:Default_values_for_labels_and_aliases).
- `scripts/card_stats.py` on the local copy (hub/, which matches the metadata JSON in files, bytes and rows for every key of every table):
  - Regional codes (keys with a hyphen): labels 106, descriptions 102, aliases 97, claims_labels 105 (e.g. `en-gb`, `pt-br`, `zh-hant`, `sr-el`, `de-formal`).
  - labels, Q ids: 74,429,805 with a label; 48,736,354 (65.5%) with `en`, 18,921,222 (25.4%) with `mul`, 10,020,338 (13.5%) with `mul` and no `en`, 15,673,113 (21.1%) with neither. P ids: 13,448; 13,444 with `en`, 385 with `mul`, 4 with `mul` and no `en`.
  - descriptions, Q ids: 67,735,222 with a description; 58,961,668 (87.0%) with `en`, 8,773,554 (13.0%) without; no `mul` key. P ids: 13,427; 13,344 with `en`.
  - aliases, Q ids: 15,704,236 with an alias; 8,721,660 (55.5%) with `en`, 540,446 (3.4%) with `mul`, 253,760 (1.6%) with `mul` and no `en`, 6,728,816 (42.8%) with neither. P ids: 12,328; 7,859 with `en`, 266 with `mul`, 38 with `mul` and no `en`.
- On `chunk_1237`, 135 of 10,000 entities carry a `mul` label and 2 carry only `mul` (docs/journal/2026-09-26-claims-label-maps-and-language-rule.md).

## Design

Cards are rendered from the static templates and the metadata JSON, as the last stage of `finalise-wikidata`, after every table is compacted.

- Front matter: one config per key, `config_name: {key}`, `data_files: "{key}/*.parquet"`, and one config `all` with `data_files: "*/*.parquet"`.
- Default config: `en` for labels, descriptions, aliases and claims_labels, and `enwiki` for links — `en` is the largest key of each language-split table and present in all four, and `mul` is absent from descriptions and holds only the labels an item shares across languages. claims keeps its single config.
- Body: a table per card of each key's files, size and rows from the metadata JSON, sorted by size, with the largest keys inline and every key in a collapsible `<details>` block.
- Body text: file layout as compaction writes it, and no "take the unique rows" advice for claims_labels.
- Language section in labels, descriptions, aliases and claims_labels: `mul` (labels and aliases only), MediaWiki language fallback chains (e.g. a regional code falling back to its base language), and a Polars example that coalesces a label along a chain and records the language it came from; claims notes that monolingual-text values carry their own language.
- Push: each rendered README.md is uploaded only when it differs from the repo's current README.md, one commit per repo, replacing the "only if absent" rule of `_ensure_dataset_card`. The rendered cards are also written locally for review.
- Subsets are the configs above, one per key directory, so each language (or site, for links) is a subset on the Hub and loads by name.
- Card figures are measured on a local copy of the Hub repos: `download-wikidata` (`just download`) downloads each table's repo to `hub/{table}` with `snapshot_download`, and `scripts/card_stats.py` reads it and prints the figures.
- `dataset_info` (per-config `splits` with `num_examples` and `num_bytes`, and `download_size`) is not written; row counts and sizes go in the card body. `num_bytes` is the in-memory Arrow size of a config, which needs every key read into memory once, and `datasets` 5.0.1 raises `NonMatchingSplitsSizesError` on load when a split's `num_examples` differs from the rows it read, under the default `verification_mode` `BASIC_CHECKS` (datasets/builder.py:967-968, datasets/utils/info_utils.py:62-76), so the counts must match the files exactly.

## Missing

- The renderer, the README.md push, and their stage in `finalise-wikidata`.
- The MediaWiki fallback chains to document, and the position of `mul` in Wikidata's label fallback, checked against MediaWiki and Wikibase sources.
- The file layout text and figures depend on whether the tables are re-sorted by id first (docs/journal/2026-09-30-sort-by-id.md).
- The README.md rewrite and the DESIGN.md compaction section.
