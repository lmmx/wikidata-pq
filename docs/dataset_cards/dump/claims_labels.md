---
license: cc0-1.0
language:
- multilingual
pretty_name: Wikidata Claims Labels
tags:
- wikidata
- knowledge-graph
- multilingual
{{configs}}
---

# Wikidata Claims Labels

The names, in every language, of the things
[wikidata-claims](https://huggingface.co/datasets/permutans/wikidata-claims) statements refer
to: their properties, the items they point to, and the units of their quantities.

The dump's statements hold only ids. This table names every property, item and unit that a
statement, qualifier or reference in the release refers to, with each of its labels from
[wikidata-labels](https://huggingface.co/datasets/permutans/wikidata-labels) of the same
release, one row per name. Join them in the languages you want.

## Files

Files are at `{language}/part-{i}-of-{n}.parquet`: one folder per Wikidata language code
({{key_examples}}). Each folder's rows are sorted by `ref` across its files, in
string order (`Q10` comes before `Q2`), so a filter on `ref` reads only the row groups whose
range can hold it.

## Schema

| Column | Type | |
|---|---|---|
| `field` | string | What `ref` is: `property-labels` (a property), `labels` (an item a statement points to), or `unit-labels` (a unit) |
| `ref` | string | The property, item or unit id |
| `language` | string | Language code, as in the folder name |
| `label` | string | Its name in that language |

{{sample}}

A language has one row per (`field`, `ref`). The same id can have a row in more than one field,
such as `P31` both as a property and as an item a statement points to.

## Joining to claims

For a statement in [wikidata-claims](https://huggingface.co/datasets/permutans/wikidata-claims),
the names come from:

| Name of | `field` | `ref` |
|---|---|---|
| the property | `property-labels` | `property` |
| the item it points to | `labels` | `datavalue.id` |
| the unit of a quantity | `unit-labels` | `datavalue.unit` |

The same goes for the snaks in `qualifiers` and `references`. A statement's own subject (its
`id`) is named in [wikidata-labels](https://huggingface.co/datasets/permutans/wikidata-labels).

```python
import polars as pl

hf = "hf://datasets/permutans"
claims = pl.scan_parquet(f"{hf}/wikidata-claims/all/*.parquet")
names = (
    pl.scan_parquet(f"{hf}/wikidata-claims_labels/en/*.parquet")
    .filter(pl.col("field") == "property-labels")
    .select(pl.col("ref").alias("property"), pl.col("label").alias("property_label"))
)
claims.join(names, on="property", how="left").head().collect()
```

## Subsets

Each language is a subset named by its code, and `all` holds every language. {{default}} is the
default.

```python
from datasets import load_dataset

ds = load_dataset("permutans/wikidata-claims_labels", "fr")
```

```python
import polars as pl

names = pl.scan_parquet("hf://datasets/permutans/wikidata-claims_labels/en/*.parquet")
names.filter(pl.col("ref") == "P31").collect()
```

{{sizes}}

## Languages

{{languages}}

Wikidata shows a label in a language by trying the language, then its fallback languages in
MediaWiki (`en-gb` falls back to `en`, `pt-br` to `pt`, ...), then `mul`, then `en`. The
wikidata-labels card has code that looks up a language's fallbacks and takes one label per item
along them; here, take one row per (`field`, `ref`) instead of per `id`.

## The wikidata-pq tables

Seven tables built from the same dump by [wikidata-pq](https://github.com/lmmx/wikidata-pq),
all keyed by Wikidata id:

| Dataset | Rows | Split by |
|---|---|---|
| [wikidata-entities](https://huggingface.co/datasets/permutans/wikidata-entities) | an item's or property's type, page and last revision | not split |
| [wikidata-labels](https://huggingface.co/datasets/permutans/wikidata-labels) | its name, per language | language |
| [wikidata-descriptions](https://huggingface.co/datasets/permutans/wikidata-descriptions) | its short description, per language | language |
| [wikidata-aliases](https://huggingface.co/datasets/permutans/wikidata-aliases) | its other names, per language | language |
| [wikidata-links](https://huggingface.co/datasets/permutans/wikidata-links) | its page title and badges on each Wikimedia site | site |
| [wikidata-claims](https://huggingface.co/datasets/permutans/wikidata-claims) | its statements | not split |
| [wikidata-claims_labels](https://huggingface.co/datasets/permutans/wikidata-claims_labels) | names of the properties, items and units its statements refer to, per language | language |

## Releases

Each release is built from one of Wikidata's weekly JSON dumps, named by its date: this one is
**{{release}}**, from
[`wikidata-{{release}}-all.json.bz2`](https://dumps.wikimedia.org/wikidatawiki/entities/{{release}}/),
every item and property in it. The `main` branch holds the latest release; every release is
also a tag, so a release can be pinned:

```python
from datasets import load_dataset

ds = load_dataset("permutans/wikidata-labels", "en", revision="{{release}}")
```

The tag `20260507` holds the tables built before releases, from
[philippesaade/wikidata](https://huggingface.co/datasets/philippesaade/wikidata) (the dump of
2026-05-07 without scholarly articles, and without the fields that copy dropped).

## Source and license

Built from the Wikidata JSON dump at
[dumps.wikimedia.org](https://dumps.wikimedia.org/wikidatawiki/entities/). Wikidata is released
under [CC0](https://creativecommons.org/publicdomain/zero/1.0/), and so are these tables.
