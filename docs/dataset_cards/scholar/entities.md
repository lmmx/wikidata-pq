---
license: cc0-1.0
language:
- multilingual
pretty_name: Wikidata Scholarly Entities
tags:
- wikidata
- knowledge-graph
{{configs}}
---

# Wikidata Scholarly Entities

Every scholarly work in the dump (scholarly articles, theses, conference papers, preprints, errata, reports and the other classes in [scholarly.py](https://github.com/lmmx/wikidata-pq/blob/master/src/wikidata/scholarly.py)), one row each: its type, its page on Wikidata
and the revision the dump has.

## Files

Files are at `all/part-{i}-of-{n}.parquet`. Rows are sorted by `id` across the files, in string
order (`Q10` comes before `Q2`), so a filter on `id` reads only the row groups whose id range can
hold it.

## Schema

| Column | Type | |
|---|---|---|
| `id` | string | The item (`Q…`) or property (`P…`) |
| `type` | string | `item` or `property` |
| `datatype` | string | A property's value type (`wikibase-item`, `external-id`, `time`, ...); null for an item |
| `ns` | integer | Its page's namespace on wikidata.org: 0 for items, 120 for properties |
| `title` | string | Its page's title: `Q42`, or `Property:P31` |
| `pageid` | integer | Its page's id on wikidata.org |
| `lastrevid` | integer | The page revision the dump has |
| `modified` | string | When that revision was made (ISO 8601, UTC) |

`lastrevid` and `modified` say how current each item is in the release: an item edited after the
dump was made differs on Wikidata from its rows here.

## Loading

There is one subset, `all`.

```python
import polars as pl

entities = pl.scan_parquet("hf://datasets/permutans/wikidata-scholar-entities/all/*.parquet")
entities.filter(pl.col("id") == "Q1895685").collect()
```

{{sizes}}

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
