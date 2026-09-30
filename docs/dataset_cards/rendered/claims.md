---
license: cc0-1.0
language:
- multilingual
source_datasets:
- philippesaade/wikidata
pretty_name: Wikidata Claims
tags:
- wikidata
- knowledge-graph
configs:
- config_name: "all"
  data_files: "*/*.parquet"
  default: true
---

# Wikidata Claims

The statements about every Wikidata item and property: one row per statement, with its value,
rank, qualifiers and references.

## Files

Files are at `all/part-{i}-of-{n}.parquet`. Rows are sorted by `id` across
the files, in string order (`Q10` comes before `Q2`), so a filter on `id` reads only the row
groups whose id range can hold it. An item's statements keep their order in the source.

Unlike the other tables, claims are not split by language: a statement has no language of its
own, only the names of the things it refers to do. Those names are in
[wikidata-claims_labels](https://huggingface.co/datasets/permutans/wikidata-claims_labels), split by
language, for you to join in the languages you want. A `monolingualtext` value carries its own
language, in `datavalue.language`.

## Schema

| Column | Type | |
|---|---|---|
| `id` | string | The item (`Q…`) or property (`P…`) the statement is about |
| `property` | string | The statement's property, e.g. `P31` (instance of) |
| `datavalue` | struct | Its value: see below |
| `datatype` | string | The property's datatype, e.g. `wikibase-item`, `quantity`, `time`, `external-id` |
| `rank` | string | `preferred`, `normal` or `deprecated` |
| `qualifiers` | list of {key, value} | Qualifiers, grouped by property: `key` is the property, `value` a list of snaks |
| `references` | list of lists of {key, value} | References, each a list of snaks grouped by property, as for qualifiers |

A snak, in `qualifiers` and `references`, is a struct of `property`, `datavalue` and `datatype`,
as in the statement's own columns.

### datavalue

One struct holds the fields for every kind of value, and those not used by a value are null.
It is null where a snak has no value (Wikidata's "unknown value" and "no value", which this table
does not tell apart).

| Fields | For | |
|---|---|---|
| `id` | items, properties and other entities (`wikibase-item`, `wikibase-property`, ...) | The entity's id |
| `datavalue__string` | string values (`string`, `external-id`, `url`, `commonsMedia`, ...) | The string |
| `text`, `language` | `monolingualtext` | The text and its language |
| `amount`, `upperBound`, `lowerBound`, `unit` | `quantity` | The amount and bounds as decimal strings, and the unit |
| `time`, `timezone`, `before`, `after`, `precision`, `calendarmodel` | `time` | The timestamp, its precision and calendar model |
| `latitude`, `longitude`, `altitude`, `precision`, `globe` | `globe-coordinate` | The coordinates, their precision and globe |

`precision`, `latitude` and `longitude` are structs with an integer and a float field
(`precision__integer` and `precision__number`, and so on), since the source has both. Only one of
them is set. `altitude` is always null.

## Names

For each statement, qualifier or reference snak, the names in a language come from
[wikidata-claims_labels](https://huggingface.co/datasets/permutans/wikidata-claims_labels):

| Name of | `field` | `ref` |
|---|---|---|
| the property | `property-labels` | `property` |
| the item it points to | `labels` | `datavalue.id` |
| the unit of a quantity | `unit-labels` | `datavalue.unit` |

The subject of a statement (its `id`) is named in
[wikidata-labels](https://huggingface.co/datasets/permutans/wikidata-labels). The
wikidata-claims_labels card has a worked join.

## Loading

There is one subset, `all`.

```python
import polars as pl

claims = pl.scan_parquet("hf://datasets/permutans/wikidata-claims/all/*.parquet")
claims.filter(pl.col("id") == "Q42").collect()
```

```python
from datasets import load_dataset

ds = load_dataset("permutans/wikidata-claims", streaming=True)
```

In total: 34 files, 17.7 GB of Parquet, 774,255,243 rows.

## Snaks on deleted properties

The source has some snaks on properties since deleted from Wikidata (such as P450 and P4003),
which it could not render fully. The datavalue is left as an error message, or the whole snak
as just the property id. These are left out:

- a statement whose main value is one of these snaks is dropped;
- a qualifier or reference snak that is one is dropped, and so is a qualifier group or reference
  left empty.

There are very few of them. The pipeline keeps a record of each one it drops, but they are not
published here.

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
