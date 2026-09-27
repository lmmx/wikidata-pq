---
license: cc0-1.0
language:
- multilingual
source_datasets:
- philippesaade/wikidata
pretty_name: Wikidata Sitelinks
tags:
- wikidata
- knowledge-graph
configs:
- config_name: default
  data_files: "*/*.parquet"
---

# Wikidata Sitelinks

The page each Wikidata item has on other Wikimedia sites (Wikipedias, Wikiquote, Wikisource,
Commons, ...): one row per (id, site).

Files are at `{site}/chunks-NNNN-NNNN.parquet`: one folder per site code (`enwiki`, `frwiki`,
`commonswiki`, `enwikiquote`, ...), and one file per group of source chunks.

## Schema

| Column | Type | |
|---|---|---|
| `id` | string | Item (`Q…`) id |
| `site` | string | Site code, as in the folder name |
| `title` | string | Page title on that site |

```
id          site    title
Q136719174  enwiki  FIFA Peace Prize
Q136719174  eswiki  Premio de la Paz de la FIFA
Q136719174  hrwiki  FIFA-ina Nagrada za mir
```

## Loading

Each site is its own folder, so you can read just the ones you want:

```python
import polars as pl

df = pl.scan_parquet("hf://datasets/permutans/wikidata-links/enwiki/*.parquet").collect()
```

```python
from datasets import load_dataset

ds = load_dataset("permutans/wikidata-links", data_files="enwiki/*.parquet")
```

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
