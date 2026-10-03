---
license: cc0-1.0
language:
- multilingual
pretty_name: Wikidata Scholarly Descriptions
tags:
- wikidata
- knowledge-graph
- multilingual
{{configs}}
---

# Wikidata Scholarly Descriptions

The short description of every scholarly work in Wikidata (scholarly articles, theses, conference papers, preprints, errata, reports and the other classes in [scholarly.py](https://github.com/lmmx/wikidata-pq/blob/master/src/wikidata/scholarly.py)), in every language it has one: one row
per (id, language).

## Files

Files are at `{language}/part-{i}-of-{n}.parquet`: one folder per Wikidata language code
({{key_examples}}). Each folder's rows are sorted by `id` across its files, in string
order (`Q10` comes before `Q2`), so a filter on `id` reads only the row groups whose id range can
hold it.

## Schema

| Column | Type | |
|---|---|---|
| `id` | string | Item (`Q…`) or property (`P…`) id |
| `language` | string | Language code, as in the folder name |
| `value` | string | The description |

{{sample}}

## Subsets

Each language is a subset named by its code, and `all` holds every language. {{default}} is the
default.

```python
from datasets import load_dataset

ds = load_dataset("permutans/wikidata-scholar-descriptions", "fr")
```

```python
import polars as pl

descriptions = pl.scan_parquet("hf://datasets/permutans/wikidata-scholar-descriptions/en/*.parquet")
descriptions.filter(pl.col("id") == "Q1895685").collect()
```

{{sizes}}

## Languages

{{languages}}

Wikidata shows a description in a language by trying the language, then its fallback languages
in MediaWiki (`en-gb` falls back to `en`, `pt-br` to `pt`, `de-ch` to `de`, ...), then `mul`, then `en`. The
[wikidata-scholar-labels](https://huggingface.co/datasets/permutans/wikidata-scholar-labels) card has code that
looks up a language's fallbacks and takes one row per item along them.

## The wikidata-pq tables

Seven tables built from the same dump by [wikidata-pq](https://github.com/lmmx/wikidata-pq),
all keyed by Wikidata id:

| Dataset | Rows | Split by |
|---|---|---|
| [wikidata-scholar-entities](https://huggingface.co/datasets/permutans/wikidata-scholar-entities) | an item's or property's type, page and last revision | not split |
| [wikidata-scholar-labels](https://huggingface.co/datasets/permutans/wikidata-scholar-labels) | its name, per language | language |
| [wikidata-scholar-descriptions](https://huggingface.co/datasets/permutans/wikidata-scholar-descriptions) | its short description, per language | language |
| [wikidata-scholar-aliases](https://huggingface.co/datasets/permutans/wikidata-scholar-aliases) | its other names, per language | language |
| [wikidata-scholar-links](https://huggingface.co/datasets/permutans/wikidata-scholar-links) | its page title and badges on each Wikimedia site | site |
| [wikidata-scholar-claims](https://huggingface.co/datasets/permutans/wikidata-scholar-claims) | its statements | not split |
| [wikidata-scholar-claims_labels](https://huggingface.co/datasets/permutans/wikidata-scholar-claims_labels) | names of the properties, items and units its statements refer to, per language | language |

These tables hold the dump's scholarly works; everything else is in the same seven tables
without `scholar-` in the name, such as
[wikidata-claims](https://huggingface.co/datasets/permutans/wikidata-claims).

## Releases

Each release is built from one of Wikidata's weekly JSON dumps, named by its date: this one is
**{{release}}**, from
[`wikidata-{{release}}-all.json.bz2`](https://dumps.wikimedia.org/wikidatawiki/entities/{{release}}/),
its scholarly works. The `main` branch holds the latest release; every release is
also a tag, so a release can be pinned:

```python
from datasets import load_dataset

ds = load_dataset("permutans/wikidata-scholar-labels", "en", revision="{{release}}")
```

## Source and license

Built from the Wikidata JSON dump at
[dumps.wikimedia.org](https://dumps.wikimedia.org/wikidatawiki/entities/). Wikidata is released
under [CC0](https://creativecommons.org/publicdomain/zero/1.0/), and so are these tables.
