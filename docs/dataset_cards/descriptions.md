---
license: cc0-1.0
language:
- multilingual
source_datasets:
- philippesaade/wikidata
pretty_name: Wikidata Descriptions
task_categories:
- text-generation
- fill-mask
tags:
- wikidata
- knowledge-graph
- multilingual
---

# Wikidata Descriptions

One row per (entity or property ID, language, description). Split out from
[philippesaade/wikidata](https://huggingface.co/datasets/philippesaade/wikidata), which packs every
language's description for an ID into a single JSON-mapping field on one row per ID. This repo
unpivots that mapping so you can download just the languages you want, at `{language}/*.parquet`.

A description is a short disambiguating phrase, e.g. Q42's English description is
"English author and humorist (1952–2001)" — distinct from its [label](https://huggingface.co/datasets/permutans/wikidata-labels)
("Douglas Adams").

Part of a set of six tables produced from the same source and the same pipeline — see
[Related tables](#related-tables) below, and the reprocessing pipeline's
[README](https://github.com/lmmx/wikidata-pq) / [DESIGN.md](https://github.com/lmmx/wikidata-pq/blob/master/DESIGN.md)
for how they're built.

## Schema

| Column | Type | Meaning |
|---|---|---|
| `id` | string | Entity ID (`Q...`) or property ID (`P...`) this description belongs to |
| `language` | string | Wikidata language code (e.g. `en`, `fr`, `zh`) — also the partition folder |
| `value` | string | The description text in that language |

## Example

```
id       language  value
Q42      en        English author and humorist (1952–2001)
Q42      fr        écrivain et humoriste anglais (1952-2001)
```

## Source and license

Derived from [philippesaade/wikidata](https://huggingface.co/datasets/philippesaade/wikidata)
(Jonathan Fraine & Philippe Saadé, Wikimedia Deutschland; funded by Wikimedia Deutschland), itself a
JSON-formatted rendering of the Wikidata dump. Wikidata content is dedicated to the public domain
under [CC0](https://creativecommons.org/publicdomain/zero/1.0/), and this reprocessing preserves
that license.

## Related tables

All produced by the same pipeline run, from the same source dump, joinable on `id`:

- [`permutans/wikidata-labels`](https://huggingface.co/datasets/permutans/wikidata-labels) — entity/property names
- [`permutans/wikidata-descriptions`](https://huggingface.co/datasets/permutans/wikidata-descriptions) *(this dataset)* — short descriptions
- [`permutans/wikidata-aliases`](https://huggingface.co/datasets/permutans/wikidata-aliases) — alternative names
- [`permutans/wikidata-links`](https://huggingface.co/datasets/permutans/wikidata-links) — sitelinks to Wikipedia etc.
- [`permutans/wikidata-claims`](https://huggingface.co/datasets/permutans/wikidata-claims) — the statements (property/value pairs) themselves
- [`permutans/wikidata-claims_labels`](https://huggingface.co/datasets/permutans/wikidata-claims_labels) — labels for the properties, units and referenced entities that appear *inside* claims
