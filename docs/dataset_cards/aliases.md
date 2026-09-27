---
license: cc0-1.0
language:
- multilingual
source_datasets:
- philippesaade/wikidata
pretty_name: Wikidata Aliases
tags:
- wikidata
- knowledge-graph
- multilingual
configs:
- config_name: default
  data_files: "*/*.parquet"
---

# Wikidata Aliases

The alternative names of every Wikidata item and property, in every language it has any: one row
per alias, so an id can have several rows in a language. The main name is in
[wikidata-labels](https://huggingface.co/datasets/permutans/wikidata-labels), not here.

Files are at `{language}/chunks-NNNN-NNNN.parquet`: one folder per Wikidata language code
(`en`, `fr`, `zh-hans`, `mul`, ...), and one file per group of source chunks.

## Schema

| Column | Type | |
|---|---|---|
| `id` | string | Item (`Q…`) or property (`P…`) id |
| `language` | string | Language code, as in the folder name |
| `value` | string | One alias |

```
id          language  value
Q136719174  en        FIFA Peace Prize – Football Unites the World
Q136719174  en        FIFA Peace Award
```

## Loading

Each language is its own folder, so you can read just the ones you want:

```python
import polars as pl

df = pl.scan_parquet("hf://datasets/permutans/wikidata-aliases/en/*.parquet").collect()
```

```python
from datasets import load_dataset

ds = load_dataset("permutans/wikidata-aliases", data_files="en/*.parquet")
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
