# The philippesaade build

Before releases from the official dumps, the datasets were built from
[philippesaade/wikidata](https://huggingface.co/datasets/philippesaade/wikidata): 7,449
Parquet files (959 GB) of the dump of 2026-05-07, one row per entity with its labels,
descriptions, aliases, sitelinks and claims as JSON columns. That copy left out the
scholarly works and dropped several fields of the dump (each snak's type and hash,
statement ids, reference hashes, sitelink badges, the entity's own fields). It also
repeated the full multilingual labels of every property, item and unit inside each
statement that mentions them, which the pipeline moved out into `claims_labels`.

The pipeline still builds from it when `WIKIDATA_RELEASE` is unset:

```sh
just run         # process-wikidata: pull, process, partition and push every chunk, then finalise
just download    # download-wikidata: a local copy of the repos in hub/
just finalise    # finalise-wikidata: compact, sort, and push the dataset cards
just card-stats  # the cards' figures, from hub/
just cards       # render the cards without pushing
```

In this mode:

- Working directories are at the repository root (`state/`, `data/`, `results/`, `audit/`,
  `staging/`, `compact/`, `hub/`).
- The pull step downloads each chunk from the source repo, and a background prefetch
  downloads the chunks ahead within a disk budget. See [Pull](../reference/pull.md).
- There are six tables (no `entities`), and claims_labels is built from the label maps in
  each chunk's claims during processing.
- Uploads go to `main` directly, with no build branch, and `run` calls `finalise` itself at
  the end.

Its published files are kept as the tag `20260507` of each main-set repo once the first
release is promoted.
