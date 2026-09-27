---
license: cc0-1.0
language:
- multilingual
source_datasets:
- philippesaade/wikidata
pretty_name: Wikidata Claims Labels
task_categories:
- text-generation
tags:
- wikidata
- knowledge-graph
- multilingual
---

# Wikidata Claims Labels

A lookup table of labels for everything a [claim](https://huggingface.co/datasets/permutans/wikidata-claims)
can refer to: the property, the entity a `wikibase-item` claim points to, and the unit a `quantity`
claim is measured in. One row per (field, ref, language, label), partitioned by language at
`{language}/*.parquet`.

## Why this table exists

The raw Wikidata dump repeats every property's, entity's and unit's full multilingual label map
inside *every claim that mentions it* — the same "instance of" label in 300+ languages, duplicated on
every one of the millions of claims using property P31. Reproducing that in
[`permutans/wikidata-claims`](https://huggingface.co/datasets/permutans/wikidata-claims) would blow
claims up by a four-figure factor for no benefit (see the pipeline's
[design journal](https://github.com/lmmx/wikidata-pq/blob/master/docs/journal/2026-09-26-claims-unsplit.md)
for the measurement). Instead, claims stay unsplit and unlabelled, and every label map claims used to
carry inline is pulled out once into this deduplicated table, which you join back in for whichever
language(s) you want.

Part of a set of six tables produced from the same source and the same pipeline — see
[Related tables](#related-tables) below, and the reprocessing pipeline's
[README](https://github.com/lmmx/wikidata-pq) / [DESIGN.md](https://github.com/lmmx/wikidata-pq/blob/master/DESIGN.md)
for how they're built.

## Schema

| Column | Type | Meaning |
|---|---|---|
| `field` | string | Which kind of label this is — `labels` (a referenced entity's own label), `property-labels`, or `unit-labels` |
| `ref` | string | The ID this label belongs to — an entity ID, a property ID, or a unit ID, depending on `field` |
| `language` | string | Wikidata language code (e.g. `en`, `fr`, `zh`) — also the partition folder |
| `label` | string | The label text in that language |

## Joining to claims

Given a row from [`permutans/wikidata-claims`](https://huggingface.co/datasets/permutans/wikidata-claims):

- Property label: join on `field = "property-labels"`, `ref = claims.property`
- Entity value label (for `datatype = "wikibase-item"`): join on `field = "labels"`, `ref = claims.datavalue.id`
- Unit label (for `datatype = "quantity"`): join on `field = "unit-labels"`, `ref = claims.datavalue.unit`

Filter to your language(s) of interest, e.g. `language = "en"`, before joining — that's the whole
point of partitioning this table by language rather than shipping every claim with every language's
labels attached.

Note: an entity's *own* label (as the subject of the `id` column, not as a claim's referenced value)
lives in [`permutans/wikidata-labels`](https://huggingface.co/datasets/permutans/wikidata-labels)
instead — this table only covers labels for things claims *point to*.

## Source and license

Derived from [philippesaade/wikidata](https://huggingface.co/datasets/philippesaade/wikidata)
(Jonathan Fraine & Philippe Saadé, Wikimedia Deutschland; funded by Wikimedia Deutschland), itself a
JSON-formatted rendering of the Wikidata dump. Wikidata content is dedicated to the public domain
under [CC0](https://creativecommons.org/publicdomain/zero/1.0/), and this reprocessing preserves
that license.

## Related tables

All produced by the same pipeline run, from the same source dump:

- [`permutans/wikidata-labels`](https://huggingface.co/datasets/permutans/wikidata-labels) — entity/property names (an ID's *own* label)
- [`permutans/wikidata-descriptions`](https://huggingface.co/datasets/permutans/wikidata-descriptions) — short descriptions
- [`permutans/wikidata-aliases`](https://huggingface.co/datasets/permutans/wikidata-aliases) — alternative names
- [`permutans/wikidata-links`](https://huggingface.co/datasets/permutans/wikidata-links) — sitelinks to Wikipedia etc.
- [`permutans/wikidata-claims`](https://huggingface.co/datasets/permutans/wikidata-claims) — the statements (property/value pairs) this table's labels belong to
- [`permutans/wikidata-claims_labels`](https://huggingface.co/datasets/permutans/wikidata-claims_labels) *(this dataset)* — labels for the properties, units and referenced entities that appear *inside* claims
