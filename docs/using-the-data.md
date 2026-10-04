# Using the data

The tables are Hugging Face datasets under [`permutans`](https://huggingface.co/permutans).
Each one's dataset card on the Hub gives its schema, its subsets and their sizes, and how
to load it. The [project README](https://github.com/lmmx/wikidata-pq#example-one-item-in-english)
has a worked example that puts all the tables together for one item. This page covers what
those leave out.

## Which build `main` holds

Until the release 20260928 is promoted, `main` holds the six tables built from the
philippesaade copy of the dump of 2026-05-07 (see [Home](index.md)). Promotion replaces
them with the release and tags them `20260507`. The two builds' schemas differ: see
[Builds compared](#builds-compared).

## Which table

| To get | Read |
|---|---|
| an item's or property's name in a language | `wikidata-labels` |
| its short description, or its other names | `wikidata-descriptions`, `wikidata-aliases` |
| its page on Wikipedia or another Wikimedia site | `wikidata-links`, one subset per site (`enwiki`, `frwikivoyage`, `commonswiki`, ...) |
| its statements (facts) | `wikidata-claims` |
| the names of the properties, items and units in statements | `wikidata-claims_labels` |
| whether an id is an item or a property, its page and last revision | `wikidata-entities` (releases only) |
| a scholarly article, thesis, preprint and so on | the same tables named `wikidata-scholar-*` (releases only) |

Only items and properties are included. Lexemes are in a separate Wikidata dump, which is
not read.

## Subsets and `all`

A language-split table has one subset per language code, and `all` holds every language's
rows together, so in `all` an id has one row per language. claims and entities are not
split, and `all` is their only subset. To list a table's subsets:

```python
from huggingface_hub import HfFileSystem

[p["name"].rsplit("/", 1)[1] for p in HfFileSystem().ls("datasets/permutans/wikidata-labels")
 if p["type"] == "directory"]
```

`load_dataset` takes one subset at a time. For several languages at once, pass their
folders as `data_files`:

```python
from datasets import load_dataset

ds = load_dataset("permutans/wikidata-labels", data_files=["en/*.parquet", "mul/*.parquet"])
```

## Looking things up

Rows are sorted by `id` (by `ref` in claims_labels), so a filter on it reads only the row
groups that can hold that id. A filter on anything else (a label's text, an external
identifier in the claims) reads the whole subset. For many such lookups, download the
subset once and query the local copy.

## Languages and `mul`

`mul` is Wikidata's language code for a label that holds in every language, such as a
person's name. Wikidata shows a label in a language by trying the language, its fallback
languages, then `mul`, then `en`, so most lookups want `mul` as well as the language. The
wikidata-labels card has the fallback chain and code that takes one label per item along
it.

## Statements

### Choosing by rank

`rank` is `preferred`, `normal` or `deprecated`. Wikidata's own "best" values for a
property are its preferred statements if it has any, otherwise its normal ones:

```python
import polars as pl

claims = pl.scan_parquet("hf://datasets/permutans/wikidata-claims/all/*.parquet")
best = claims.filter(pl.col("id") == "Q42").filter(
    pl.col("rank")
    == pl.when((pl.col("rank") == "preferred").any().over("id", "property"))
    .then(pl.lit("preferred"))
    .otherwise(pl.lit("normal"))
)
```

### Dates

A `time` value is a string such as `+1952-03-11T00:00:00Z`, with its `precision` an
integer (in `precision.precision__integer`): 11 is a day, 10 a month, 9 a year, 8 a decade,
7 a century, down to 0 for a billion years. Coarser than a day, the unused parts are zeros
(`+1974-00-00T00:00:00Z` at precision 9), which a datetime parser refuses. A leading `-` is
a year BCE (`-0044` is 44 BCE). `calendarmodel` is the Gregorian (`Q1985727`) or Julian
(`Q1985786`) calendar. To take the year:

```python
dv = pl.col("datavalue").struct
years = claims.filter(pl.col("datatype") == "time").select(
    "id",
    "property",
    year=dv.field("time").str.extract(r"^([+-]\d+)-").cast(pl.Int64),
    precision=dv.field("precision").struct.field("precision__integer"),
)
```

### Quantities

`amount`, `upperBound` and `lowerBound` are decimal strings with a sign (`+1.96`), which
cast to a float. `unit` is `1` for a number without a unit, and otherwise the unit's URI,
`http://www.wikidata.org/entity/Q…`. That URI is the unit's `ref` in claims_labels, under
`unit-labels`.

### Qualifiers and references

`qualifiers` is a list of `{key, value}`, `key` a property and `value` a list of snaks,
each with its own `property`, `datavalue` and `datatype`. To get one row per qualifier
snak:

```python
qualifiers = (
    claims.select("id", pl.col("property").alias("statement_property"), "qualifiers")
    .explode("qualifiers")
    .select("id", "statement_property", snak=pl.col("qualifiers").struct.field("value"))
    .explode("snak")
    .unnest("snak")
)
```

In a release, each reference is a struct with its `hash` and its `snaks`, grouped by
property in the same way:

```python
references = (
    claims.select("statement_id", "references")
    .explode("references")
    .select(
        "statement_id",
        reference=pl.col("references").struct.field("hash"),
        snaks=pl.col("references").struct.field("snaks"),
    )
    .explode("snaks")
    .select("statement_id", "reference", snak=pl.col("snaks").struct.field("value"))
    .explode("snak")
    .unnest("snak")
)
```

In the 20260507 build a reference is the list of `{key, value}` itself, so the first
`explode("references")` is followed by a second one in place of the `snaks` field.

### Names of properties, items and units

Join claims_labels on `property`, `datavalue.id` and `datavalue.unit`. The
wikidata-claims_labels card gives the `field` for each and a worked join; the README
example names all three.

## Builds compared

| | 20260507 (philippesaade copy) | A release |
|---|---|---|
| Tables | six | seven: adds `wikidata-entities` |
| Scholarly works | left out | in `wikidata-scholar-*` |
| Unknown value vs no value | not told apart (`datavalue` null) | `snaktype`: `somevalue`, `novalue` |
| Statement id, type | none | `statement_id`, `statement_type` |
| Snak hash, value type | none | `hash`, `datavalue_type` |
| Qualifier and reference order | none | `qualifiers-order`; each reference's `snaks-order` |
| `references` | list of lists of `{key, value}` | list of `{hash, snaks, snaks-order}` |
| Entity values | `id` | `id`, `entity-type`, `numeric-id` |
| Sitelink badges | none | `badges` in links |
| Snaks on deleted properties | dropped, with a statement whose main value is one | kept, with a null `datatype` |
| claims_labels | from the label maps in the source | from the release's own labels, of both sets |

A query that names only the columns the 20260507 build has runs on a release too, except
one that reads `references`.

## Pinning a release

`main` changes when a release is promoted. Every release is a tag, created at promotion, so
the first tags (`20260507` and `20260928`) exist once 20260928 is promoted:

```python
from datasets import load_dataset
from huggingface_hub import snapshot_download
import polars as pl

ds = load_dataset("permutans/wikidata-labels", "en", revision="20260507")
lf = pl.scan_parquet("hf://datasets/permutans/wikidata-labels@20260507/en/*.parquet")
snapshot_download("permutans/wikidata-labels", repo_type="dataset", revision="20260507",
                  allow_patterns=["en/*"], local_dir="wikidata-labels")
```

A release's files are named `part-{i}-of-{n}.parquet` within each folder, and `n` can
change from one release to the next, so read a folder by `*.parquet` rather than by file
name.

## Reading from the Hub

The datasets are public, so reading needs no token. The Hub limits how many requests a
client makes, more tightly without one: log in (`hf auth login`, or set `HF_TOKEN`) to
raise the limit. A scan over `hf://` fetches parts of files on demand, which suits a few
lookups. For repeated queries, or a whole subset, download it first with `snapshot_download`
(as above, or as in the README) and read the local files.

## License and citation

Wikidata is released under [CC0](https://creativecommons.org/publicdomain/zero/1.0/), and
so are these tables. There is no formal citation: cite Wikidata, the release (its dump
date), and this repository.

## Terms

- **Item**: a thing described, with an id `Q` and a number.
- **Property**: the relation in a statement, such as "instance of" (`P31`): `P` and a
  number.
- **Statement** (claim): a property and a value about an item, with a rank, qualifiers
  (context, such as a point in time) and references (provenance).
- **Snak**: one property-value pair, the statement's own (its main snak) or one of its
  qualifiers or references.
- **Datavalue**: a snak's value, of a datatype such as `wikibase-item` (another item),
  `external-id`, `quantity`, `time` or `monolingualtext`.
- **Sitelink**: an item's page title on a Wikimedia site, such as `enwiki`.
- **`mul`**: the language code for a label that holds in every language.
- **Row group**: the unit a Parquet reader reads at once. Each records its minimum and
  maximum id, which lets a filter on id skip it.
- **Release**: a build from one of Wikidata's weekly JSON dumps, named by its date, such
  as `20260928`.
- **Set**: a release's main tables (`wikidata-*`) or its scholarly ones
  (`wikidata-scholar-*`).
- **Chunk**: 10,000 entities of a dump, the unit the pipeline processes.

The [Wikibase JSON format](https://doc.wikimedia.org/Wikibase/master/php/docs_topics_json.html)
describes the dump these come from.
