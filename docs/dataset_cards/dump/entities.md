---
license: cc0-1.0
language:
- multilingual
pretty_name: Wikidata Entities
tags:
- wikidata
- knowledge-graph
{{configs}}
---

# Wikidata Entities

Every Wikidata item and property in the dump, one row each: its type, its page on Wikidata
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

entities = pl.scan_parquet("hf://datasets/permutans/wikidata-entities/all/*.parquet")
entities.filter(pl.col("id") == "Q42").collect()
```

{{sizes}}

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

The dump's scholarly works (any item that is an instance of one of the classes in
[scholarly.py](https://github.com/lmmx/wikidata-pq/blob/master/src/wikidata/scholarly.py): scholarly articles, theses, conference papers, preprints, errata, reports, ...) are in the same seven tables
named `wikidata-scholar-*`, such as
[wikidata-scholar-claims](https://huggingface.co/datasets/permutans/wikidata-scholar-claims);
these tables hold everything else.

## Releases

Each release is built from one of Wikidata's weekly JSON dumps, named by its date: this one is
**{{release}}**, from
[`wikidata-{{release}}-all.json.bz2`](https://dumps.wikimedia.org/wikidatawiki/entities/{{release}}/),
every item and property in it but its scholarly works. The `main` branch holds the latest release; every release is
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
