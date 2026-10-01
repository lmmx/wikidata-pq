# Features of Wikidata's external identifiers

A Matryoshka sparse autoencoder (SAE) trained on which external identifiers each Wikidata
item has. It learns a dictionary of features, each a bundle of identifier systems that go
together ("MathWorld + nLab + ProofWiki", "WFO + POWO + IPNI", "IMDb + TMDB + Letterboxd"),
at four nested levels of detail, and gives every item a sparse code over them: a handful of
named features. Items with similar codes are catalogued by the same kinds of outside
sources, which makes them similar kinds of thing, whatever their "instance of" says.

## The idea

An external identifier (an "external-id" property: VIAF, MathWorld, GBIF, IMDb, ...) records
that some outside catalogue chose to include the item. Each catalogue has its own scope, so
an item's set of identifiers says what kind of thing it is in the judgement of thousands of
independent curators. As data, it is a binary matrix of items by identifier properties,
very sparse and very skewed: a few hubs (Google Knowledge Graph, Freebase, VIAF) are on
millions of items, most properties on a few thousand.

A sparse autoencoder compresses each row into a few active features out of a large
dictionary, and reconstructs the row from them. Each feature is interpretable by its
decoder row: the identifier properties it switches on. The Matryoshka variant (Bussmann et
al., "Learning Multi-Level Features with Matryoshka Sparse Autoencoders", ICML 2025) orders
the dictionary and trains nested prefixes of it (here the first 64, 256, 1,024 and 4,096
features) to each reconstruct the input on their own, so the early features are broad and
the later ones specific. A later feature's parent is the broader feature most
often active with it, which makes the levels a tree.

## Pipeline

| Step | Script | Writes |
|---|---|---|
| Each item's set of external-ID properties (non-deprecated), counted by distinct set; properties on fewer than 50 items and items with fewer than 2 identifiers are left out | `id_sets.sh` | `id_sets.parquet`, `id_properties.parquet` |
| Train the SAE on the distinct sets, drawn by `items ** alpha`, with the Matryoshka BatchTopK trainer of [dictionary_learning](https://github.com/saprmarks/dictionary_learning) | `train.sh` | `sae/trainer_0/ae.pt`, `sae/run.json` |
| Encode every set and item; describe the features | `export.sh` | `features.parquet`, `codes.parquet` |
| The items most like a seed item | `neighbours.sh` | `neighbours_<seed>.stdout` |
| Tables to publish and query from a browser: items with labels and weights, postings by feature, names by lowercased label | `publish.sh` | `publish/` |
| Upload them to the run's folder in the dataset, with the card (`dataset_card.md`) and the list of runs (`runs.json`) | `upload.sh` | [permutans/wikidata-id-matryoshka-sae-features](https://huggingface.co/datasets/permutans/wikidata-id-matryoshka-sae-features) |

Everything is written to `sae/output/`: the identifier sets there, and each training run's
model, tables and logs in a folder of its own, `sae/output/$RUN`, named by `RUN=` on every
script after `id_sets.sh`. `train.sh` refuses a run that already has a model. Each
published run is listed with its settings in `runs.json`, and the dataset and the Space
hold one folder per run. The Parquet files and weights are not committed; the `.stdout`
logs are. Training uses the `sae` dependency group
(`uv add --group sae dictionary-learning`) and a GPU.

A release built from an official Wikidata dump (see the repository README) is named with
`RELEASE=` on `id_sets.sh` and `train.sh`: its identifier sets go to
`sae/output/releases/$RELEASE/`, read from `releases/$RELEASE/hub/`, and the run records its
release (in `sae/output/$RUN/release`) for its later steps. Add `"release"` to the run's entry
in `runs.json` for the Space to show it.

```sh
RELEASE=20260928 sae/id_sets.sh # a release's identifier sets
RUN=v2 RELEASE=20260928 sae/train.sh
sae/id_sets.sh                  # one pass over the claims, about a minute
RUN=v1 sae/train.sh             # 1.5 hours per 100M sets on an RTX 3090
RUN=v1 sae/export.sh            # one more pass; --no-items reuses codes.parquet
RUN=v1 sae/neighbours.sh        # several seeds; or a name: sae/neighbours.sh "Hilbert space"
RUN=v1 sae/publish.sh           # sae/output/v1/publish/, about 2.3 GB
RUN=v1 sae/upload.sh            # to v1/ in the dataset; add v1 to runs.json first
just default-run v1             # the Space opens v1 (last in runs.json); uploads runs.json
space/upload.sh                 # the Space (space/index.html), with a picker of the runs
```

## First run (v0)

- **Data:** 53.6M items have an external identifier, over 9,778 properties. Kept: 7,752
  properties and 31.9M items, in 4.0M distinct sets (one set, SIMBAD + Gaia, is 14% of the
  items: stars).
- **Training:** 100M sets drawn, `k = 8` active features per set on average. 91% of the
  variance explained; on held-out sets, 96% of each set's identifiers are among its top
  reconstructed values ([train.stdout](output/train.stdout)).
- **Features:** 3,578 of the 4,096 are ever active (64, 189, 738 and 2,587 by level). They
  read as catalogue domains: plants, taxa, chemicals, genes, proteins, films, video games,
  music, artists, journals, researchers, national libraries, places, listed buildings
  ([export.stdout](output/export.stdout)). Feature 2479 is MathWorld + nLab + ProofWiki,
  on 6,321 items (number, triangle, Hilbert space, Fourier transform, group, Kalman filter),
  beside features for mathematicians (MacTutor, Mathematics Genealogy Project, MR Author).
- **Neighbours:** the Kalman filter's nearest items are random walk, Monte Carlo method,
  control theory, martingale, dynamical system, and economics topics (it has an STW
  economics thesaurus ID) ([neighbours_Q846780.stdout](output/neighbours_Q846780.stdout)).

## Known weaknesses

- **Broad features follow set variety, not size.** Drawing sets by `items ** 0.5` favours
  domains with many distinct identifier combinations (people, films), so they fill the 64
  broadest features, while stars first appear at the third level and genes at the fourth.
  A higher `--alpha` (1 draws by item) would give the big domains (stars, places, taxa,
  genes) broad features of their own.
- **Well-catalogued items have many features** (Emmy Noether 61, Hilbert space 22): the
  fixed threshold used after training lets rich sets switch on many, mostly generic ones
  (national encyclopedias, library authority files). Neighbours weight features by rarity
  (log of items over the feature's items), which helps but does not remove them.
- **Thin items stay thin.** An item with two or three identifiers gets one or two broad
  features (image moment: only "academic topic").
- **Squared-error loss on 0/1 inputs**, as the trainer is built for language-model
  activations; a binary cross-entropy loss would fit the data better.
- **Some features are single hubs** (Freebase, Google Knowledge Graph), presence flags
  rather than domains.
