---
license: cc0-1.0
language:
- multilingual
source_datasets:
- philippesaade/wikidata
pretty_name: Wikidata Descriptions
tags:
- wikidata
- knowledge-graph
- multilingual
{{configs}}
---

# Wikidata Descriptions

The short description of every Wikidata item and property, in every language it has one: one row
per (id, language).

## Files

Files are at `{language}/part-{i}-of-{n}.parquet`: one folder per Wikidata language code
(`en`, `fr`, `zh-hans`, ...). Each folder's rows are sorted by `id` across its files, in string
order (`Q10` comes before `Q2`), so a filter on `id` reads only the row groups whose id range can
hold it.

## Schema

| Column | Type | |
|---|---|---|
| `id` | string | Item (`Q…`) or property (`P…`) id |
| `language` | string | Language code, as in the folder name |
| `value` | string | The description |

```
id          language  value
Q136719174  en        prize awarded by FIFA
P13897      en        identifier for a sports team in the Sofascore database
```

## Subsets

Each language is a subset named by its code, and `all` holds every language. `en` is the
default.

```python
from datasets import load_dataset

ds = load_dataset("permutans/wikidata-descriptions", "fr")
```

```python
import polars as pl

descriptions = pl.scan_parquet("hf://datasets/permutans/wikidata-descriptions/en/*.parquet")
descriptions.filter(pl.col("id") == "Q42").collect()
```

{{sizes}}

## Languages

Of the 67,735,222 items with a description, 58,961,668 (87.0%) have one in `en`.

There is no `mul` subset: Wikidata's
[default values](https://www.wikidata.org/wiki/Help:Default_values_for_labels_and_aliases), in
the `mul` code, are for labels and aliases only.

Wikidata shows a description in a language by trying the language, then its fallback languages
in MediaWiki (`en-gb` falls back to `en`, `pt-br` to `pt`, `de-ch` to `de`, ...), then `en`. The
[wikidata-labels](https://huggingface.co/datasets/permutans/wikidata-labels) card has code that
looks up a language's fallbacks and takes one row per item along them.

## The wikidata-pq tables

Six tables built from the same source by [wikidata-pq](https://github.com/lmmx/wikidata-pq), all
keyed by Wikidata id:

| Dataset | Rows | Split by |
|---|---|---|
| [wikidata-labels](https://huggingface.co/datasets/permutans/wikidata-labels) | an item's or property's name, per language | language |
| [wikidata-descriptions](https://huggingface.co/datasets/permutans/wikidata-descriptions) | its short description, per language | language |
| [wikidata-aliases](https://huggingface.co/datasets/permutans/wikidata-aliases) | its other names, per language | language |
| [wikidata-links](https://huggingface.co/datasets/permutans/wikidata-links) | its page title on each Wikimedia site | site |
| [wikidata-claims](https://huggingface.co/datasets/permutans/wikidata-claims) | its statements | not split |
| [wikidata-claims_labels](https://huggingface.co/datasets/permutans/wikidata-claims_labels) | names of the properties, items and units its statements refer to, per language | language |

## Source and license

Built from [philippesaade/wikidata](https://huggingface.co/datasets/philippesaade/wikidata), a
Parquet copy of the Wikidata dump with one JSON-valued row per item or property, by Jonathan Fraine
and Philippe Saadé at Wikimedia Deutschland. Wikidata is released under
[CC0](https://creativecommons.org/publicdomain/zero/1.0/), and so are these tables.
