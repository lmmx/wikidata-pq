---
license: cc0-1.0
language:
- multilingual
source_datasets:
- permutans/wikidata-claims
pretty_name: Wikidata External ID Matryoshka SAE Features
tags:
- wikidata
- knowledge-graph
- sparse-autoencoder
configs:
- config_name: items
  data_files: v0/items.parquet
  default: true
- config_name: features
  data_files: v0/features.parquet
- config_name: postings
  data_files: v0/postings.parquet
- config_name: names
  data_files: v0/names.parquet
- config_name: classes
  data_files: v0/classes.parquet
- config_name: members
  data_files: v0/members.parquet
---

# Wikidata External ID Matryoshka SAE Features

Features of a Matryoshka sparse autoencoder (SAE) trained on which external identifiers each
Wikidata item has, and every item's sparse code over them.

An external identifier (VIAF, MathWorld, GBIF, IMDb, ...) records that an outside catalogue
chose to include the item, so an item's set of identifiers says what kind of thing it is in
the judgement of thousands of independent curators. The SAE learns a dictionary of 4,096
features, each a bundle of identifier systems that go together ("MathWorld + nLab +
ProofWiki", "WFO + POWO + IPNI", "IMDb + TMDB + Letterboxd"), at four nested levels of
detail (64, 256, 1,024 and 4,096 features, after Bussmann et al., "Learning Multi-Level
Features with Matryoshka Sparse Autoencoders", ICML 2025). Each item's code is the handful
of features active on it.

Each training run is a folder (`v0/`, ...) holding its own tables and model, listed with its
settings in `runs.json`; the configs below read `v0`, the first, experimental run (see
[Limitations](#limitations)).

## Files

In each run's folder:

| File | Rows | |
|---|---|---|
| `items.parquet` | 31.7M | each item's code, sorted by `id` in row groups of 20,000 |
| `features.parquet` | 4,096 | what each feature is |
| `postings.parquet` | 98.2M | every (feature, item) pair, sorted by feature and item |
| `names.parquet` | | each labelled item with an external ID, coded or not, by its lowercased label, for search by prefix |
| `classes.parquet` | | the classes the items are instances of, and every class above them |
| `members.parquet` | | each class's direct instances and subclasses among the items, sorted by class |
| `id_properties.parquet` | 7,752 | the model's input columns |
| `model/ae.pt`, `model/config.json`, `model/run.json` | | the trained SAE ([dictionary_learning](https://github.com/saprmarks/dictionary_learning)'s `MatryoshkaBatchTopKSAE`) |

Ids sort in string order (`Q10` before `Q2`), so a filter on `id` reads only the row groups
whose id range can hold it, and a filter on `feature` in the postings reads only that
feature's rows.

### `items`

| Column | Type | |
|---|---|---|
| `id` | string | Item (`Q…`) id |
| `label` | string | English label, else the multilingual (`mul`) one |
| `kinds` | list[uint32] | Its "instance of" (P31) classes, as Q numbers; for a class with none, its "subclass of" (P279) parents |
| `is_class` | bool | Whether `kinds` are "subclass of" parents |
| `features` | list[uint16] | Active features, heaviest first |
| `activations` | list[float32] | Each feature's activation |
| `weights` | list[float32] | Activation × the feature's `idf` |
| `norm` | float32 | Euclidean norm of `weights` |

### `features`

| Column | Type | |
|---|---|---|
| `feature` | uint16 | Feature index: broader features first |
| `group` | uint8 | Level, 0 (the first 64) to 3 (the last 3,072) |
| `label` | string | Its 3 identifier properties with the largest decoder weights |
| `properties`, `weights` | list | Its 10 largest, by name, and their decoder weights |
| `items`, `sets` | int64 | Items, and distinct identifier sets, it is active on |
| `parent`, `parent_label`, `parent_share` | | The feature of a broader level most often active with it, and on what share of its items |
| `children` | uint32 | Features with it as parent |
| `examples` | list[string] | Items in the most Wikipedias with it among their strongest three |
| `idf` | float64 | log(items coded / items it is active on) |
| `strongest` | list[string] | Its 40 items of greatest weight |

### `postings`

| Column | Type | |
|---|---|---|
| `feature` | uint16 | Feature |
| `id` | uint32 | Item, as a Q number (delta-encoded) |
| `unit16` | uint16 | The item's weight for the feature over its `norm` (0 to 1), times 65,535 (byte-stream-split) |
| `kinds` | list[uint32] | The item's kinds, as in `items` |

The cosine of two items is the sum, over the features they share, of one's `unit16` / 65,535
times the other's `weight`, divided by the other's `norm`.

Every file is zstd-compressed at level 19 with version 2 data pages; sorted string keys
(`items.id`, `names.key`) are delta-encoded.

### `names`

| Column | Type | |
|---|---|---|
| `key` | string | `label`, lowercased: the sort key |
| `label` | string | As in `items` |
| `id` | string | Item |
| `description` | string | English description |
| `wikipedias` | uint16 | Wikipedias with an article on the item |
| `coded` | bool | Whether the item has a code in `items` (2+ external IDs from catalogues of 50+ items) |

### `classes`

| Column | Type | |
|---|---|---|
| `class` | uint32 | Class, as a Q number |
| `label` | string | English label, else the multilingual one |
| `parents` | list[uint32] | Its "subclass of" (P279) classes |
| `items` | uint32 | Coded items with it among their `kinds` |

### `members`

| Column | Type | |
|---|---|---|
| `class` | uint32 | Class, as a Q number (delta-encoded) |
| `id` | uint32 | Item, as a Q number (delta-encoded) |
| `subclass` | bool | Whether the item is a subclass of the class, else an instance |

Statements of deprecated rank are left out of `kinds`, `parents` and `members`.

## Space

[permutans/wikidata-id-features](https://huggingface.co/spaces/permutans/wikidata-id-features)
searches items by name and shows their features and neighbours, reading these files in the
browser with DuckDB-WASM.

## Queries

With DuckDB (here, or in the dataset viewer's SQL console):

```sql
-- An item's features
SELECT i.label, f.feature, f."group", f.label AS feature_label, i.weight
FROM (
  SELECT label, unnest(features) AS feature, unnest(weights) AS weight
  FROM 'hf://datasets/permutans/wikidata-id-matryoshka-sae-features/v0/items.parquet'
  WHERE id = 'Q846780'  -- Kalman filter
) i
JOIN 'hf://datasets/permutans/wikidata-id-matryoshka-sae-features/v0/features.parquet' f
  USING (feature)
ORDER BY i.weight DESC;

-- Its neighbours: items sharing its 8 heaviest features, by cosine of the weights
WITH item AS (
  SELECT unnest(features) AS feature, unnest(weights) AS w, norm
  FROM 'hf://datasets/permutans/wikidata-id-matryoshka-sae-features/v0/items.parquet'
  WHERE id = 'Q846780'
), seed AS (SELECT * FROM item ORDER BY w DESC LIMIT 8)
SELECT 'Q' || p.id AS id, sum(p.unit16 / 65535 * s.w) / any_value(s.norm) AS similarity
FROM 'hf://datasets/permutans/wikidata-id-matryoshka-sae-features/v0/postings.parquet' p
JOIN seed s USING (feature)
WHERE p.id <> 846780
GROUP BY p.id
ORDER BY similarity DESC
LIMIT 20;
```

## How it was made

From [permutans/wikidata-claims](https://huggingface.co/datasets/permutans/wikidata-claims),
by the scripts in [sae/](https://github.com/lmmx/wikidata-pq/tree/master/sae):

1. Each item's set of external-ID properties (statements of datatype `external-id`, not
   deprecated), counted by distinct set. Properties on fewer than 50 items, and items left
   with fewer than 2 identifiers, are left out: 7,752 properties and 31.9M items, in 4.0M
   distinct sets.
2. The SAE trained on the sets as 0/1 vectors, drawn with probability proportional to
   `items ** 0.5`, with the Matryoshka BatchTopK trainer (`k = 8`, 100M sets drawn): 91% of
   the variance explained, and on held-out sets 96% of each set's identifiers among its top
   reconstructed values. 3,578 of the 4,096 features are ever active.
3. Every set encoded with the threshold the trainer settled on, and each item given its
   set's code.

## Limitations

- **The broad features follow set variety, not size.** Drawing sets by `items ** 0.5`
  favours domains with many distinct identifier combinations (people, films, libraries),
  so they fill the 64 broadest features; stars first appear at level 2 and genes at 3.
- **Well-catalogued items have many features** (Emmy Noether 61), mostly generic ones
  (national encyclopedias, library authority files). `weights` discount them by rarity.
- **Thin items stay thin:** an item with two or three identifiers gets one or two broad
  features.
- **Squared-error loss on 0/1 inputs**, as the trainer is built for language-model
  activations.
- **Items with fewer than 2 kept identifiers have no code.**
