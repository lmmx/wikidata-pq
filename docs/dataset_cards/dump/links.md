---
license: cc0-1.0
language:
- multilingual
pretty_name: Wikidata Sitelinks
tags:
- wikidata
- knowledge-graph
{{configs}}
---

# Wikidata Sitelinks

The page each Wikidata item has on other Wikimedia sites (Wikipedias, Wikiquote, Wikisource,
Commons, ...): one row per (id, site).

## Files

Files are at `{site}/part-{i}-of-{n}.parquet`: one folder per site code ({{key_examples}}). Each folder's rows are sorted by `id` across its files, in
string order (`Q10` comes before `Q2`), so a filter on `id` reads only the row groups whose id
range can hold it.

## Schema

| Column | Type | |
|---|---|---|
| `id` | string | Item (`Q…`) id |
| `site` | string | Site code, as in the folder name |
| `title` | string | Page title on that site |
| `badges` | list of strings | The page's badges on that site, as item ids (e.g. `Q17437796`, featured article; `Q17437798`, good article) |

{{sample}}

## Subsets

Each site is a subset named by its code, and `all` holds every site. {{default}} is the default.

```python
from datasets import load_dataset

ds = load_dataset("permutans/wikidata-links", "frwiki")
```

```python
import polars as pl

links = pl.scan_parquet("hf://datasets/permutans/wikidata-links/enwiki/*.parquet")
links.filter(pl.col("id") == "Q42").collect()
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
