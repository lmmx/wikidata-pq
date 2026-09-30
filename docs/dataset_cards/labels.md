---
license: cc0-1.0
language:
- multilingual
source_datasets:
- philippesaade/wikidata
pretty_name: Wikidata Labels
tags:
- wikidata
- knowledge-graph
- multilingual
{{configs}}
---

# Wikidata Labels

The name of every Wikidata item and property, in every language it has one: one row per
(id, language).

## Files

Files are at `{language}/part-{i}-of-{n}.parquet`: one folder per Wikidata language code
(`en`, `fr`, `zh-hans`, `mul`, ...). Each folder's rows are sorted by `id` across its files, in
string order (`Q10` comes before `Q2`), so a filter on `id` reads only the row groups whose id
range can hold it.

## Schema

| Column | Type | |
|---|---|---|
| `id` | string | Item (`Q…`) or property (`P…`) id |
| `language` | string | Language code, as in the folder name |
| `value` | string | The label |

```
id          language  value
Q136719174  en        FIFA Peace Prize
Q136719174  fr        Prix FIFA pour la paix
Q136719174  de        FIFA-Friedenspreis
P13897      en        Sofascore sports team ID
```

## Subsets

Each language is a subset named by its code, and `all` holds every language. `en` is the
default.

```python
from datasets import load_dataset

ds = load_dataset("permutans/wikidata-labels", "fr")
```

```python
import polars as pl

labels = pl.scan_parquet("hf://datasets/permutans/wikidata-labels/en/*.parquet")
labels.filter(pl.col("id") == "Q42").collect()
```

{{sizes}}

## Languages

Not every item has a label in every language: of the 74,429,805 items with a label,
48,736,354 (65.5%) have one in `en`.

`mul` is Wikidata's code for a
[default label](https://www.wikidata.org/wiki/Help:Default_values_for_labels_and_aliases), one
that holds in every language, such as a person's name in the Latin alphabet. 18,921,222 items
have a `mul` label, and 10,020,338 of them (13.5% of items with a label) have no `en` label, so
reading `en` alone misses their names.

Wikidata shows a label in a language by trying, in order:

1. the language itself;
2. its fallback languages in MediaWiki (`en-gb` falls back to `en`, `pt-br` to `pt`, `de-ch` to
   `de`, `zh-hk` to `zh-hant`, `zh-tw`, `zh` and `zh-hans`, ...);
3. `mul`;
4. `en`.

The Wikidata API lists a language's fallbacks. Wikidata also converts the script between the
variants of some languages (it can show a `zh-hans` label in `zh-hant` script); these tables
hold each label as entered, so a label taken from a fallback variant stays in that variant's
script.

To get one label per item in a language, with the language each label came from:

```python
import json
import urllib.request

import polars as pl
from huggingface_hub import HfFileSystem

repo = "datasets/permutans/wikidata-labels"
subsets = {p["name"].rsplit("/", 1)[1] for p in HfFileSystem().ls(repo) if p["type"] == "directory"}


def fallbacks(lang: str) -> list[str]:
    url = (
        "https://www.wikidata.org/w/api.php?action=query&meta=languageinfo"
        f"&liprop=fallbacks&licode={lang}&format=json&formatversion=2"
    )
    request = urllib.request.Request(url, headers={"User-Agent": "wikidata-labels-example/0.1"})
    return json.load(urllib.request.urlopen(request))["query"]["languageinfo"][lang]["fallbacks"]


def chain(lang: str) -> list[str]:
    """The languages Wikidata tries for `lang`, in order, that have a subset here."""
    return [l for l in dict.fromkeys([lang, *fallbacks(lang), "mul", "en"]) if l in subsets]


langs = chain("de-ch")  # ['de-ch', 'de', 'mul', 'en']
labels = (
    pl.concat([pl.scan_parquet(f"hf://{repo}/{l}/*.parquet") for l in langs])
    .sort(pl.col("language").replace_strict(langs, range(len(langs))), maintain_order=True)
    .unique("id", keep="first", maintain_order=True)
)
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
