---
license: cc0-1.0
language:
- multilingual
pretty_name: Wikidata Scholarly Claims
tags:
- wikidata
- knowledge-graph
{{configs}}
---

# Wikidata Scholarly Claims

The statements about every scholarly work in Wikidata (scholarly articles, theses, conference papers, preprints, errata, reports and the other classes in [scholarly.py](https://github.com/lmmx/wikidata-pq/blob/master/src/wikidata/scholarly.py)): one row per statement, with its value,
rank, qualifiers and references.

## Files

Files are at `all/part-{i}-of-{n}.parquet`. Rows are sorted by `id` across
the files, in string order (`Q10` comes before `Q2`), so a filter on `id` reads only the row
groups whose id range can hold it. An item's statements keep their order in the dump.

Unlike the other tables, claims are not split by language: a statement has no language of its
own, only the names of the things it refers to do. Those names are in
[wikidata-scholar-claims_labels](https://huggingface.co/datasets/permutans/wikidata-scholar-claims_labels), split by
language, for you to join in the languages you want. A `monolingualtext` value carries its own
language, in `datavalue.language`.

## Schema

| Column | Type | |
|---|---|---|
| `id` | string | The item (`Q…`) or property (`P…`) the statement is about |
| `snaktype` | string | `value`, `somevalue` (Wikidata's "unknown value") or `novalue` ("no value") |
| `property` | string | The statement's property, e.g. `P31` (instance of) |
| `hash` | string | The main snak's hash (null: the dump gives main snaks none) |
| `datavalue` | struct | Its value: see below; null unless `snaktype` is `value` |
| `datavalue_type` | string | The value's type: `wikibase-entityid`, `string`, `monolingualtext`, `quantity`, `time`, `globecoordinate` |
| `datatype` | string | The property's datatype, e.g. `wikibase-item`, `quantity`, `time`, `external-id`; null on a property since deleted |
| `statement_type` | string | `statement` |
| `statement_id` | string | The statement's id (GUID), e.g. `Q42$F078E5B3-F9A8-480E-B7AC-D97778CBBEF9`, stable across releases |
| `rank` | string | `preferred`, `normal` or `deprecated` |
| `references` | list of structs | References: each a `hash`, its `snaks` grouped by property (a list of {key, value}: `key` the property, `value` a list of snaks), and `snaks-order` (the properties in display order) |
| `qualifiers` | list of {key, value} | Qualifiers, grouped by property: `key` is the property, `value` a list of snaks |
| `qualifiers-order` | list of strings | The qualifiers' properties in display order |

A snak, in `qualifiers` and `references`, is a struct of `snaktype`, `property`, `hash`,
`datavalue`, `datavalue_type` and `datatype`, as in the statement's own columns.

### datavalue

One struct holds the fields for every kind of value, and those not used by a value are null.
It is null where a snak has no value: `snaktype` says which (`somevalue` or `novalue`).

| Fields | For | |
|---|---|---|
| `id`, `entity-type`, `numeric-id` | items, properties and other entities (`wikibase-item`, `wikibase-property`, ...) | The entity's id, its type (`item`, `property`, `lexeme`, `form`, `sense`) and its number (null for forms and senses) |
| `datavalue__string` | string values (`string`, `external-id`, `url`, `commonsMedia`, ...) | The string |
| `text`, `language` | `monolingualtext` | The text and its language |
| `amount`, `upperBound`, `lowerBound`, `unit` | `quantity` | The amount and bounds as decimal strings, and the unit |
| `time`, `timezone`, `before`, `after`, `precision`, `calendarmodel` | `time` | The timestamp, its precision and calendar model |
| `latitude`, `longitude`, `altitude`, `precision`, `globe` | `globe-coordinate` | The coordinates, their precision and globe |

`precision`, `latitude` and `longitude` are structs with an integer and a float field
(`precision__integer` and `precision__number`, and so on), since the dump has both. Only one of
them is set. `altitude` is always null in the dump.

## Names

For each statement, qualifier or reference snak, the names in a language come from
[wikidata-scholar-claims_labels](https://huggingface.co/datasets/permutans/wikidata-scholar-claims_labels):

| Name of | `field` | `ref` |
|---|---|---|
| the property | `property-labels` | `property` |
| the item it points to | `labels` | `datavalue.id` |
| the unit of a quantity | `unit-labels` | `datavalue.unit` |

The subject of a statement (its `id`) is named in
[wikidata-scholar-labels](https://huggingface.co/datasets/permutans/wikidata-scholar-labels). The
wikidata-claims_labels card has a worked join.

## Loading

There is one subset, `all`.

```python
import polars as pl

claims = pl.scan_parquet("hf://datasets/permutans/wikidata-scholar-claims/all/*.parquet")
claims.filter(pl.col("id") == "Q1895685").collect()
```

```python
from datasets import load_dataset

ds = load_dataset("permutans/wikidata-scholar-claims", streaming=True)
```

{{sizes}}

## Snaks on deleted properties

A snak on a property since deleted from Wikidata (such as P450) is kept as the dump has it,
with a null `datatype`.

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
