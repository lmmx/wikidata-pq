---
license: cc0-1.0
language:
- multilingual
source_datasets:
- philippesaade/wikidata
pretty_name: Wikidata Sitelinks
task_categories:
- text-generation
tags:
- wikidata
- knowledge-graph
- multilingual
---

# Wikidata Sitelinks

One row per (entity ID, site, page title). Split out from
[philippesaade/wikidata](https://huggingface.co/datasets/philippesaade/wikidata), which packs every
site's interwiki link for an ID into a single JSON-mapping field on one row per ID. This repo unpivots
that mapping so you can download just the sites you want, at `{site}/*.parquet`.

A sitelink connects a Wikidata entity to its page on a Wikimedia project — Wikipedia, Wikisource,
Wikivoyage, Wikiquote, etc. — in a given language. The `site` code (e.g. `enwiki`, `frwiktionary`)
identifies both the project and the language, unlike the plain language codes used to partition the
other tables in this set.

Part of a set of six tables produced from the same source and the same pipeline — see
[Related tables](#related-tables) below, and the reprocessing pipeline's
[README](https://github.com/lmmx/wikidata-pq) / [DESIGN.md](https://github.com/lmmx/wikidata-pq/blob/master/DESIGN.md)
for how they're built.

## Schema

| Column | Type | Meaning |
|---|---|---|
| `id` | string | Entity ID (`Q...`) this sitelink belongs to |
| `site` | string | Site code (e.g. `enwiki`, `dewiki`, `enwikivoyage`) — also the partition folder |
| `title` | string | The page title on that site |

## Example

```
id       site       title
Q42      enwiki     Douglas Adams
Q42      frwiki     Douglas Adams
Q42      enwikiquote Douglas Adams
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
- [`permutans/wikidata-descriptions`](https://huggingface.co/datasets/permutans/wikidata-descriptions) — short descriptions
- [`permutans/wikidata-aliases`](https://huggingface.co/datasets/permutans/wikidata-aliases) — alternative names
- [`permutans/wikidata-links`](https://huggingface.co/datasets/permutans/wikidata-links) *(this dataset)* — sitelinks to Wikipedia etc.
- [`permutans/wikidata-claims`](https://huggingface.co/datasets/permutans/wikidata-claims) — the statements (property/value pairs) themselves
- [`permutans/wikidata-claims_labels`](https://huggingface.co/datasets/permutans/wikidata-claims_labels) — labels for the properties, units and referenced entities that appear *inside* claims
