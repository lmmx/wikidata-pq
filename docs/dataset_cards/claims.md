---
license: cc0-1.0
language:
- multilingual
source_datasets:
- philippesaade/wikidata
pretty_name: Wikidata Claims
task_categories:
- text-generation
tags:
- wikidata
- knowledge-graph
---

# Wikidata Claims

One row per Wikidata claim (a property/value statement about an entity), typed and flattened from
the raw dump's nested JSON. Unlike the other five tables in this set, claims are **not** split by
language — a claim itself has no language; only its labels do (see [Why claims aren't split
by language](#why-claims-arent-split-by-language)). All rows live under `all/*.parquet`.

Part of a set of six tables produced from the same source and the same pipeline — see
[Related tables](#related-tables) below, and the reprocessing pipeline's
[README](https://github.com/lmmx/wikidata-pq) / [DESIGN.md](https://github.com/lmmx/wikidata-pq/blob/master/DESIGN.md)
for how they're built.

## Schema

| Column | Type | Meaning |
|---|---|---|
| `id` | string | Entity ID (`Q...`) the claim is about |
| `property` | string | Property ID (`P...`), e.g. `P31` for "instance of" |
| `datatype` | string | The claim's value type, e.g. `wikibase-item`, `string`, `time`, `quantity`, `monolingualtext`, `external-id`, `globe-coordinate` |
| `datavalue` | struct | The value itself — see [Datavalue fields](#datavalue-fields), which vary by `datatype` |
| `rank` | string | `normal`, `preferred`, or `deprecated` |
| `qualifiers` | list\<struct\> | Extra context on the claim (e.g. "as of" a point in time) — see [Qualifiers and references](#qualifiers-and-references) |
| `references` | list\<list\<struct\>\> | Provenance for the claim — see [Qualifiers and references](#qualifiers-and-references) |
| `mainsnak__string` | string | Set only for a small number of corrupted claims in the source dump where the whole claim collapsed to a bare property-id string instead of an object (see [Data quality notes](#data-quality-notes)); null otherwise |

### Datavalue fields

`datavalue` is one struct with every possible value field; only the ones relevant to the row's
`datatype` are non-null (this mirrors how the source dump itself unions many value shapes):

| Field | Used for `datatype` | Meaning |
|---|---|---|
| `id` | `wikibase-item` | The referenced entity's ID (`Q...`) — look up its label in [`claims_labels`](https://huggingface.co/datasets/permutans/wikidata-claims_labels) (`field="labels"`) or its own row in [`labels`](https://huggingface.co/datasets/permutans/wikidata-labels) |
| `amount`, `unit`, `upperBound`, `lowerBound` | `quantity` | The numeric amount (as a string, to preserve precision) and its unit entity ID — unit label via `claims_labels` (`field="unit-labels"`) |
| `time`, `timezone`, `before`, `after`, `calendarmodel`, `precision` | `time` | ISO-ish timestamp and precision info; `precision` is itself a struct (`precision__integer`/`precision__number`) |
| `latitude`, `longitude`, `altitude`, `globe` | `globe-coordinate` | Coordinates; `latitude`/`longitude` are structs (`{name}__number`/`{name}__integer`) since the source mixes int and float representations |
| `text`, `language` | `monolingualtext` | Text in a single fixed language (not multilingual like the other tables) |
| `datavalue__string` | `string`, `external-id`, and other scalar-string datatypes | The raw string value |
| `value`, `error` | corrupted source rows (see below) | Present only where the source dump's property/datatype lookup failed for this snak |

### Qualifiers and references

These mirror the raw Wikidata JSON's own structure (see the
[Wikibase JSON docs](https://doc.wikimedia.org/Wikibase/master/php/docs_topics_json.html#json_snaks)):

- A **qualifier** adds context to a claim, e.g. "position held" qualified by "start time". Each entry
  is `{key: property_id, value: [snak, ...]}` — a property ID paired with one or more snaks (each
  shaped like `datavalue`/`datatype` above) giving that qualifier's value(s).
- A **reference** is a list of such `{key, value}` groups (i.e. `references` is a list of "one
  reference = list of property-grouped snaks"), giving the source(s) that support the claim.

## Why claims aren't split by language

Earlier versions of this pipeline tried to give claims the same per-language partitioning as the
other tables, joining in property/entity/unit labels and exploding one row per claim into one row per
language the property or value had a label in. Since near-universal properties like P31 ("instance
of") are translated into 300+ languages, this inflated row counts by a four-figure factor for
negligible benefit — see the
[design journal](https://github.com/lmmx/wikidata-pq/blob/master/docs/journal/2026-09-26-claims-unsplit.md)
for the measurements that led to reverting it. Claims now stay as one row per claim, and you join in
whichever language(s) you want from [`claims_labels`](https://huggingface.co/datasets/permutans/wikidata-claims_labels)
yourself — see that dataset's card for join examples.

## Data quality notes

A small number of claims in the source dump are internally corrupted, most often because the claim's
property has since been deleted from Wikidata, so an id-to-datatype lookup that the source dump relies
on failed. Two forms of this are represented, both kept (not dropped) so they're auditable:

- The claim's `mainsnak` is a well-formed object, but its `datavalue` collapsed to a bare string
  instead of the expected structure — the raw value ends up in `datavalue.value`, with `datavalue.error`
  set if the source dump recorded one.
- The entire `mainsnak` collapsed to a bare string (just the property ID) instead of an object at
  all — in this case `property` and `datatype` are null, and the property ID is in `mainsnak__string`
  instead.

## Source and license

Derived from [philippesaade/wikidata](https://huggingface.co/datasets/philippesaade/wikidata)
(Jonathan Fraine & Philippe Saadé, Wikimedia Deutschland; funded by Wikimedia Deutschland), itself a
JSON-formatted rendering of the Wikidata dump. Wikidata content is dedicated to the public domain
under [CC0](https://creativecommons.org/publicdomain/zero/1.0/), and this reprocessing preserves
that license.

## Related tables

All produced by the same pipeline run, from the same source dump, joinable on `id`:

- [`permutans/wikidata-labels`](https://huggingface.co/datasets/permutans/wikidata-labels) — entity/property names
- [`permutans/wikidata-descriptions`](https://huggingface.co/datasets/permutans/wikidata-descriptions) — short descriptions
- [`permutans/wikidata-aliases`](https://huggingface.co/datasets/permutans/wikidata-aliases) — alternative names
- [`permutans/wikidata-links`](https://huggingface.co/datasets/permutans/wikidata-links) — sitelinks to Wikipedia etc.
- [`permutans/wikidata-claims`](https://huggingface.co/datasets/permutans/wikidata-claims) *(this dataset)* — the statements (property/value pairs) themselves
- [`permutans/wikidata-claims_labels`](https://huggingface.co/datasets/permutans/wikidata-claims_labels) — labels for the properties, units and referenced entities that appear *inside* claims
