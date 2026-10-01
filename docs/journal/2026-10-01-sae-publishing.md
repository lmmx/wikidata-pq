# 2026-10-01: Publishing the SAE's tables for a browser

## Current State

### Tables (sae/publish.py, sae/publish.sh)

- sae/publish.py writes `sae/output/$RUN/publish/`: `features.parquet` (with `idf`, `children`, `strongest`), `items.parquet` (sorted by id in row groups of 20,000), `postings.parquet` (every (feature, item) pair, sorted by feature and id), `classes.parquet`, `members.parquet`, `names.parquet` (sorted by lowercased label), `id_properties.parquet` and `model/` (ae.pt, config.json, run.json).
- The first publish capped postings at 20,000 per feature (items.parquet 951.0 MB, postings.parquet 23,002,234 rows and 255.1 MB, model/ae.pt 254.1 MB) — feature 0's postings then stop at weight 8.23 and the Kalman filter's own weight on it is 7.74, so the card's neighbour query returned statistic replication, rate of return and personalization first; over uncapped postings (98,172,844 pairs) it returns Monte Carlo method, fixed point and random walk among the first, in 1.1 s on the local files (DuckDB 1.5.6). sae/publish.py keeps every posting by default (8ab5882).
- In the weight-ordered postings the `id` column (strings) held 379 MB of 929 MB, `rank` 255 MB, `norm` 128 MB, `weight` 110 MB and `kinds` 50 MB; postings are now Q numbers sorted within each feature (DELTA_BINARY_PACKED), with `weight / norm` as one column and no `rank` or `norm` (1.07 MB against 2.11 MB on five features), and each feature's 40 heaviest items go in `features.parquet` as `strongest`.
- sae/publish.py reads each coded item's "instance of" (P31) values and every "subclass of" (P279) edge in one pass over the claims (deprecated statements left out), gives a coded item with no "instance of" its "subclass of" parents as `kinds` (with `is_class`), walks up to every class above, and writes `classes.parquet` (`class`, `label`, `parents`, `items`) and `kinds` in `items.parquet` and `postings.parquet`.
- sae/publish.py writes `members.parquet` (`class`, `id`, `subclass`): each class's direct instances and subclasses among the coded items, sorted by class.
- sae/publish.py writes to `names.parquet` every labelled item with a non-deprecated external-ID statement, with `coded` marking those with a code, and labels items and names in English, else multilingual, else the first of de, fr, es, it, pt, nl, sv, pl, ru, ja, zh, else any language.
- The v0 publish with all of the above wrote 4,096 features, 31,682,565 items, 98,172,844 postings, 117,919 classes (190,039 classes with no "instance of" taking their superclasses as kinds), 36,176,463 members and 53,578,070 names, 31,678,084 of them coded (sae/output/v0/publish.stdout); Q118819064 and Q118823515 have their Italian labels ("crudo di Cuneo", "torrone di Bagnara").

### Dataset (sae/dataset_card.md, sae/upload.sh)

- sae/upload.sh uploads `sae/output/$RUN/publish` to the folder `$RUN/` of `permutans/wikidata-id-matryoshka-sae-features` (or `$REPO`), exits unless sae/runs.json names the run, and uploads sae/dataset_card.md (as README.md) and sae/runs.json to the top of the repo.
- The dataset repo holds v0 under `v0/` and `runs.json` at its top; the first upload's top-level copies were deleted (`items.parquet` at the top answers 404).
- sae/dataset_card.md's configs and queries read `v0/`; its two SQL queries ran on the local publish files with DuckDB 1.5.6 for the weight-ordered layout.
- The model weights sit in the dataset repo under `$RUN/model/`, not in a model repo.
- The Hub answers range requests for the dataset's files with `access-control-allow-origin` set, through a 302 to its CDN.

## Missing

- The card's neighbour query has not been run against the id-sorted, 16-bit postings (docs/journal/2026-10-01-space-transfer-and-compression.md).
