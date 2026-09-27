---
license: cc0-1.0
language:
- multilingual
source_datasets:
- philippesaade/wikidata
pretty_name: Wikidata Labels
task_categories:
- text-generation
- fill-mask
tags:
- wikidata
- knowledge-graph
- multilingual
---

# Wikidata Labels

One row per (entity or property ID, language, label). Split out from
[philippesaade/wikidata](https://huggingface.co/datasets/philippesaade/wikidata), which packs every
language's label for an ID into a single JSON-mapping field on one row per ID. This repo unpivots
that mapping so you can download just the languages you want, at `{language}/*.parquet`.

Part of a set of six tables produced from the same source and the same pipeline — see
[Related tables](#related-tables) below, and the reprocessing pipeline's
[README](https://github.com/lmmx/wikidata-pq) / [DESIGN.md](https://github.com/lmmx/wikidata-pq/blob/master/DESIGN.md)
for how they're built.

## Schema

| Column | Type | Meaning |
|---|---|---|
| `id` | string | Entity ID (`Q...`) or property ID (`P...`) this label belongs to |
| `language` | string | Wikidata language code (e.g. `en`, `fr`, `zh`) — also the partition folder |
| `value` | string | The label text in that language |

## Example

```
id       language  value
Q42      en        Douglas Adams
Q42      fr        Douglas Adams
P31      en        instance of
```

## Source and license

Derived from [philippesaade/wikidata](https://huggingface.co/datasets/philippesaade/wikidata)
(Jonathan Fraine & Philippe Saadé, Wikimedia Deutschland; funded by Wikimedia Deutschland), itself a
JSON-formatted rendering of the Wikidata dump. Wikidata content is dedicated to the public domain
under [CC0](https://creativecommons.org/publicdomain/zero/1.0/), and this reprocessing preserves
that license.

## Related tables

All produced by the same pipeline run, from the same source dump, joinable on `id`:

- [`permutans/wikidata-labels`](https://huggingface.co/datasets/permutans/wikidata-labels) *(this dataset)* — entity/property names
- [`permutans/wikidata-descriptions`](https://huggingface.co/datasets/permutans/wikidata-descriptions) — short descriptions
- [`permutans/wikidata-aliases`](https://huggingface.co/datasets/permutans/wikidata-aliases) — alternative names
- [`permutans/wikidata-links`](https://huggingface.co/datasets/permutans/wikidata-links) — sitelinks to Wikipedia etc.
- [`permutans/wikidata-claims`](https://huggingface.co/datasets/permutans/wikidata-claims) — the statements (property/value pairs) themselves
- [`permutans/wikidata-claims_labels`](https://huggingface.co/datasets/permutans/wikidata-claims_labels) — labels for the properties, units and referenced entities that appear *inside* claims (a separate, deduplicated lookup — this `labels` table only covers each ID's *own* label)
