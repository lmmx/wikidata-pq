# wikidata-pq

Wikidata as six Parquet datasets on the Hugging Face Hub, split by language: 35.5 GB in all,
built from the 959 GB [philippesaade/wikidata](https://huggingface.co/datasets/philippesaade/wikidata)
dump, 7,449 Parquet files with every language and statement of an item in JSON columns (see
[totals](https://github.com/lmmx/wikidata-pq/blob/master/scripts/source_size/chunk_totals.csv)).

## Datasets

| Dataset | Rows | Split by | Size |
|---|---|---|--:|
| [wikidata-labels](https://huggingface.co/datasets/permutans/wikidata-labels) | an item's or property's name, per language | language | 6.5 GB |
| [wikidata-descriptions](https://huggingface.co/datasets/permutans/wikidata-descriptions) | its short description, per language | language | 5.5 GB |
| [wikidata-aliases](https://huggingface.co/datasets/permutans/wikidata-aliases) | its other names, per language | language | 1.4 GB |
| [wikidata-links](https://huggingface.co/datasets/permutans/wikidata-links) | its page title on each Wikimedia site | site | 1.3 GB |
| [wikidata-claims](https://huggingface.co/datasets/permutans/wikidata-claims) | its statements, one row per statement | not split | 17.7 GB |
| [wikidata-claims_labels](https://huggingface.co/datasets/permutans/wikidata-claims_labels) | names of the properties, items and units its statements refer to, per language | language | 3.2 GB |
| | | | **35.5 GB** |

Sizes are of the Parquet files on the Hub, in decimal units as the Hub shows them (33.1 GiB on
disk). The source is 27 times larger: among other things, it repeats the full multilingual
names of every property, item and unit in each statement that mentions them (see [Why](#why)).

Each language (or site) is a folder and a subset of its own, so you download only the ones
you want; `all` holds every one. Within a folder, rows are sorted by id, so a filter on the id
reads only the row groups that can hold it.

```python
import polars as pl

labels = pl.scan_parquet("hf://datasets/permutans/wikidata-labels/en/*.parquet")
labels.filter(pl.col("id") == "Q42").collect()
```

```python
from datasets import load_dataset

ds = load_dataset("permutans/wikidata-labels", "fr")
```

Each dataset's card gives its schema, the size of every subset, and how languages fall back
(including Wikidata's `mul` code for labels that hold in every language). The cards are
rendered from [docs/dataset_cards](docs/dataset_cards) by the pipeline.

## Example: one item in English

All six tables together, for one language: an item's label, description, aliases and Wikipedia
page, and its statements with every property, item and unit named. Wikidata shows English as
`en`, then `mul` (its code for names that hold in every language, such as a person's name), so
`langs` holds both, in that order; monolingual text values are kept to the same languages.
Every table is sorted by id, so each lookup reads only the row groups that can hold it.

```python
import polars as pl

hf = "hf://datasets/permutans"
langs = ["en", "mul"]  # English, then Wikidata's default for every language
item = "Q42"


def scan(table: str, *keys: str) -> pl.LazyFrame:
    return pl.concat(
        [pl.scan_parquet(f"{hf}/wikidata-{table}/{key}/*.parquet") for key in keys]
    )


def first(lf: pl.LazyFrame, *by: str) -> pl.LazyFrame:
    """One row per `by`, from the first language in `langs` that has one."""
    rank = pl.col("language").replace_strict(langs, range(len(langs)))
    return lf.sort(rank, maintain_order=True).unique(by, keep="first", maintain_order=True)


is_item = pl.col("id") == item
label = first(scan("labels", *langs).filter(is_item), "id").collect()
description = scan("descriptions", "en").filter(is_item).collect()
aliases = scan("aliases", *langs).filter(is_item).collect()
wikipedia = scan("links", "enwiki").filter(is_item).collect()

# Statements, with monolingual text kept to the same languages
dv = pl.col("datavalue").struct
claims = (
    scan("claims", "all")
    .filter(is_item)
    .filter((pl.col("datatype") != "monolingualtext") | dv.field("language").is_in(langs))
    .collect()
)

# Names of the properties, items and units they refer to
refs = pl.concat([claims["property"], claims["datavalue"].struct.field("id"),
                  claims["datavalue"].struct.field("unit")]).drop_nulls().unique()
names = first(scan("claims_labels", *langs).filter(pl.col("ref").is_in(refs.implode())),
              "field", "ref").collect()


def name(field: str, alias: str) -> pl.DataFrame:
    return names.filter(pl.col("field") == field).select(
        pl.col("ref").alias(alias), pl.col("label").alias(f"{alias}_label")
    )


statements = (
    claims.with_columns(item=dv.field("id"), unit=dv.field("unit"))
    .join(name("property-labels", "property"), on="property", how="left")
    .join(name("labels", "item"), on="item", how="left")
    .join(name("unit-labels", "unit"), on="unit", how="left")
    .select(
        "property",
        "property_label",
        pl.coalesce(
            "item_label",
            dv.field("text"),
            dv.field("datavalue__string"),
            pl.when(dv.field("amount").is_not_null()).then(
                pl.concat_str(dv.field("amount"), "unit_label", separator=" ", ignore_nulls=True)
            ),
            dv.field("time"),
        ).alias("value"),
        "rank",
    )
)
```

What it finds for Q42, abridged:

```
label        Douglas Adams (en)
description  British science fiction writer and humorist (1952–2001) (en)
aliases      Douglas Noël Adams, Douglas Noel Adams, Douglas N. Adams (mul)
wikipedia    enwiki: Douglas Adams

property  property_label          value                  rank
P31       instance of             human                  normal
P106      occupation              novelist               normal
P569      date of birth           +1952-03-11T00:00:00Z  normal
P27       country of citizenship  United Kingdom         normal
P2048     height                  +1.96 metre            normal
P1477     birth name              Douglas Noël Adams     normal
...       (337 statements)
```

Q42's aliases are all `mul`, so with `en` alone it would have none. For another language, put
its code first (and its fallbacks, as the wikidata-labels card shows): `["de", "mul", "en"]`.
Qualifier and reference snaks are named the same way, from their `property` and `datavalue`.

## Downloading only what you need

Each language is its own folder, so a local copy of one language is a download of those
folders. For the example above (claims, not split by language, are the bulk of it):

```python
from huggingface_hub import snapshot_download

folders = {
    "labels": ["en/*", "mul/*"],
    "descriptions": ["en/*"],
    "aliases": ["en/*", "mul/*"],
    "links": ["enwiki/*"],
    "claims": ["all/*"],
    "claims_labels": ["en/*", "mul/*"],
}
for table, patterns in folders.items():
    snapshot_download(
        f"permutans/wikidata-{table}",
        repo_type="dataset",
        allow_patterns=patterns,
        local_dir=f"wikidata/wikidata-{table}",
    )
```

Then set `hf = "wikidata"` in the example to read the local copy. Without the `is_item`
filter, the same code gives the English tables for every item, reading each subset in full.

[demos/item.py](demos/item.py) is the example as a script, for any item and language, on a
local copy: `python demos/item.py Q64 --lang de --data wikidata`.
[demos/ancestors.py](demos/ancestors.py) walks an item's family tree through the claims, a
batch of ids per generation: `python demos/ancestors.py Q517 --lang fr`.
[demos/divisions.py](demos/divisions.py) tabulates a country's states or regions with their
capital, population, area and head of government, using ranks, qualifiers and units:
`python demos/divisions.py Q183 --lang de`.
[demos/timeline.py](demos/timeline.py) lays out London's history from its dated statements,
with its population counts as a bar chart: `python demos/timeline.py`.
[demos/wikiprojects.py](demos/wikiprojects.py) collects the items linked to WikiProjects, and
shows how the concepts of one (Mathematics by default) fit together: `python demos/wikiprojects.py`.
[demos/concepts.py](demos/concepts.py) and [demos/algorithms.py](demos/algorithms.py) find every
member of a class through its subclasses, then show what Wikidata knows about them: who
concepts are named after, when they were discovered, and algorithms' time complexities.
[demos/properties.py](demos/properties.py) ranks the properties by how many entities have them,
overall and among the items in the most Wikipedias.
[demos/bearers.py](demos/bearers.py) digs into the items bearing one property (what they are, the
best known, a random sample, what else they have), run by `demos/bbc_things.sh`, `demos/bbc_news_topics.sh`,
`demos/knowledge_graph.sh` and `demos/wordnet.sh`.
[demos/news_ids.py](demos/news_ids.py) finds the properties for news websites' topic pages, by
the class Wikidata gives them: `demos/news_ids.sh`; `demos/news_topics.sh` passes them to
`demos/bearers.py` to show the topics they cover.
[demos/export_bearers.py](demos/export_bearers.py) writes such items to Parquet, one row per item
with its name, description, kinds, country and outlets, to explore in an embedding viewer:
`demos/news_topics_export.sh`, then `embedding-atlas demos/output/news_topics.parquet --text text`.
`demos/formulas_export.sh` and `demos/formulas_view.sh` do the same for the items with a defining
formula, coloured by the field that studies them.
[sae/](sae/README.md) trains a Matryoshka sparse autoencoder on which external identifiers
each item has, for features (catalogue domains, broad to specific) and item neighbours.

## Why

The source has one row per item or property, with every language and every statement packed
into JSON columns:

1. **Massive JSON objects**: claims can exceed 1M characters in a single field, breaking Polars'
   JSON decoding ([bug report](https://github.com/pola-rs/polars/issues/23891)).
2. **Nested multilingual structures**: labels are nested in per-language maps, awkward to fish
   out despite being simple scalar values.
3. **Repeated label maps**: every statement carries the full multilingual label map of each
   property, item and unit it mentions, about 98% of the claims JSON in a measured chunk. These
   tables keep each name once, in wikidata-claims_labels, and the claims keep only the ids.
4. **Complex schema**: mixed datatypes (strings, timestamps, entity references, quantities)
   within the same fields.

These tables give each kind of data its own flat schema, and each language its own files.

## Pipeline

1. **Pull**: download the source files, a chunk at a time, with prefetching ahead.
2. **Process**: normalise the JSON columns to a flat schema with
   [polars-genson](https://github.com/lmmx/polars-genson), one table per kind of data, and
   take the label maps out of the claims into claims_labels.
3. **Partition**: split each table by language (links by site; claims are not split).
4. **Push**: merge a group of chunks into one file per language, upload it, and verify it by
   sha256.
5. **Finalise**, once every chunk is uploaded:
   - **Compact**: rewrite each language's group files into files of about 500 MB.
   - **Sort**: sort each language's rows by id across its files, into `part-{i}-of-{n}.parquet`.
   - **Cards**: compute the cards' figures from a local copy, render the cards, and push those
     that changed.

Every step resumes from its state after an interruption. See [DESIGN.md](DESIGN.md) for the
details.

## Running

```sh
just run         # process-wikidata: pull, process, partition and push every chunk
just download    # download-wikidata: a local copy of the six Hub repos in hub/
just finalise    # finalise-wikidata: compact, sort, and push the dataset cards
just card-stats  # the cards' figures, from hub/
just cards       # render the cards to docs/dataset_cards/rendered without pushing
```

Source chunks are at most 1.1 GB each (see `scripts/source_size`); local disk is bounded by the
prefetch budget (60 GB) plus one upload group's partitions and staging (at most 25 GB of
partitions), and prefetch pauses below 100 GB free (see DESIGN.md). The finalise stages read the local copy in
`hub/`, 35.5 GB.

## Releases from the official dumps

A release is built from one of Wikidata's weekly JSON dumps
([dumps.wikimedia.org/wikidatawiki/entities](https://dumps.wikimedia.org/wikidatawiki/entities/)),
named by its date. `WIKIDATA_RELEASE` selects it: its working files are under
`releases/{release}/`, and its card figures under `docs/releases/{release}/`. A release keeps
every field of the dump, which the philippesaade copy had dropped: each snak's `snaktype`
(telling "unknown value" from "no value"), hash and datavalue type; each statement's id, type
and qualifier order; each reference's hash and snak order; sitelink badges; and each entity's
type, page and last revision, in a seventh table, wikidata-entities. claims_labels is built at
the end from the release's own claims and labels.

A release is published as two sets of the seven tables: its scholarly works (any item whose
"instance of" is one of the classes in `src/wikidata/scholarly.py`: scholarly articles, theses,
conference papers, ...) as `wikidata-scholar-{table}`, from `releases/{release}-scholar/`, and
everything else as `wikidata-{table}`, from `releases/{release}/`. `WIKIDATA_SCHOLAR=1` selects
the scholarly set (the recipes take `main` or `scholar`). Each set's claims_labels names refs
from both sets' labels.

```sh
just latest-dump                         # the newest release with a full JSON dump
just download-dump 20260928              # wikidata-20260928-all.json.bz2 (103 GB), md5-checked
just split-dump 20260928                 # into chunks of 10,000 entities (needs lbzip2)
rm releases/20260928/dump/*.bz2          # once split.done is written, to free the disk
just route-release 20260928              # scholarly works to releases/20260928-scholar/data
just release 20260928 20260507           # everything else, in order (below); rerun to resume
```

`just release` runs, in order (each step resumable, so rerun it after an interruption):

1. `run-release {release} scholar`, then `run-release {release} main`: each set's chunks are
   processed, partitioned and uploaded in groups to the branch `build-{release}` of each repo;
   each chunk is deleted once processed and its partitions once uploaded (`CLEAN_UP_LOCAL`).
   Three chunks are processed at once, each in its own process; `WIKIDATA_WORKERS=N just
   release …` changes that. A group uploads in the background while the next chunks run.
2. `finalise-release {release} main`: each table is compacted on the Hub (downloaded, rewritten
   into ~500 MB files, committed), then sorted by id (a local copy in `releases/{release}/hub`,
   sorted through id-range buckets for claims, committed), claims first as the largest. Then
   claims' refs for claims_labels are collected and claims' local copy deleted. It stops
   before claims_labels, which needs the scholarly set's labels.
3. `finalise-release {release} scholar`: the same for the scholarly set, then its
   claims_labels (from both sets' labels), its card figures and cards.
4. `finalise-release {release} main` again: the main set's claims_labels, figures and cards.
5. `promote-release {release} {previous} main`, then `... scholar`: tag `main` with `previous`
   (the main set's repos only: the scholarly repos are new), copy the branch's files to `main`,
   tag `main` with the release, delete the branch. Promotion refuses a set not finalised.

The release is uploaded, compacted and sorted on a branch (`build-20260928`) of each repo, so
`main` keeps the previous release until it is promoted; every release stays available as a tag
(`revision="20260928"`). Splitting holds the bz2 and the chunks together, about 210 GB for
20260928, so the bz2 is removed by hand once the split is complete. Disk peaks while a set's
claims are sorted: its local copy, its buckets and the sorted files, about three times the
claims; nothing else of that set is local yet, and the other set's chunks are gone by then.

## Notes on coverage

- About 10% of the source aliases are null ("the alias for a given ID in the given language is
  null") and are dropped.
- Snaks on properties since deleted from Wikidata, which the source could not render, are
  dropped, and so is a statement whose main value is one (see the claims card). A release
  from an official dump keeps them, with a null `datatype`.

## Terminology

- An **item** id is the thing being described: `Q` and a number.
- A **property** id is the relation in a statement, such as "instance of" (`P31`): `P` and a
  number.
- A **statement** (claim) says something about an item: a property and a value, with a rank,
  qualifiers (context, such as a point in time) and references (provenance).
- A **snak** is one property-value pair: the statement's own (its "mainsnak"), or one of its
  qualifiers or references.
- A **datavalue** is a snak's value, of a datatype such as `wikibase-item` (another item),
  `external-id` (an identifier string), `quantity`, `time` or `monolingualtext`.
- A **sitelink** is an item's page title on a Wikimedia site, such as `enwiki`.

For more, see the [Wikibase JSON format](https://doc.wikimedia.org/Wikibase/master/php/docs_topics_json.html#json_snaks)
and a [full example](https://doc.wikimedia.org/Wikibase/master/php/docs_topics_json.html#json_example).

## Background

I originally wanted only a very small part of Wikidata, to produce synthetic data for OCR
training from realistic nested data (I had used DBpedia for this before), but one thing led to
another and I decided to extract this Wikidata dataset, which had recently been uploaded with
multi-language labels to Hugging Face.
