# 2026-10-01: A Matryoshka SAE over items' external identifiers

## Current State

### Identifier sets (sae/id_sets.py, sae/id_sets.sh)

- sae/id_sets.py makes two passes over the claims, a file at a time, over non-deprecated statements of datatype `external-id`: the first counts the items holding each property, the second writes each file's sets of kept property indexes, counted by distinct set, to `sae/output/parts/`, merged by the streaming engine into `id_sets.parquet`.
- The first version built every distinct set of all properties in memory and filtered and regrouped it after the pass — the process was killed after the 34-file pass ("Terminated"), and the two-pass version replaced it (de75f07).
- The run on hub/ found 53,585,955 items with an external identifier over 9,778 properties, and kept 7,752 properties (held by 50+ items) and 31,907,221 items with 2+ of them, in 4,007,832 distinct sets (sae/output/id_sets.stdout).
- The most common set, SIMBAD ID + Gaia ID, holds 4,567,864 items (14.3% of the kept items); the top 100 sets hold 46.5%.
- DOI (P356) is on 359,109 items in the claims, so scholarly articles take little of the sets.

### Training (sae/train.py, sae/train.sh)

- sae/train.py feeds `dictionary_learning`'s `trainSAE` with `MatryoshkaBatchTopKTrainer` (PyPI `dictionary-learning` 0.1.0, the `sae` dependency group in pyproject.toml), with groups 64/192/768/3072, `k = 8`, learning-rate decay from 80% of the steps, and placeholder `layer`/`lm_name` arguments the trainer asserts.
- Batches are dense 0/1 vectors built on the GPU from CSR arrays of `id_sets.parquet`, each set drawn with probability proportional to `items ** alpha` (default 0.5), with 1% of the sets held out (sae/train.py:`batch`, `draws`).
- An earlier hand-written trainer (22c2d99) differed from `dictionary_learning`'s in decoder normalisation, the aux-k loss for dead features, gradient clipping, threshold start and decay, and learning rate, and was replaced by the library trainer (32e9351).
- The trainer's loss is squared error, applied here to 0/1 inputs.
- The first run (100M sets, 24,415 steps of 4,096, 93 minutes at 4.38 steps/s on an RTX 3090) reached 0.907 fraction of variance explained at step 24,000; on held-out sets, recall at set size was 0.958, with 8.0 features active per set and 1,607 of 4,096 features never active across 102,400 held-out draws (sae/output/train.stdout).
- The trained model is at `sae/output/sae/trainer_0/ae.pt` with `config.json`, untracked.

### Export (sae/export.py, sae/export.sh)

- sae/export.py encodes all 4,007,832 sets with the trainer's threshold: 9.0 features active per set, and 5,803 sets with none (sae/output/export.stdout).
- Live features by group, counted over items: 64 of 64, 189 of 192, 738 of 768, 2,587 of 3,072 (3,578 of 4,096).
- A feature's parent is the broader-group feature with the largest share of co-active items; the first export gave never-active features feature 0 (Gran Enciclopèdia Catalana) as parent, 518 of the 667 features with parent 0 — never-active features now have no parent (c77ca37).
- Examples were first the lowest id of each feature's most common sets, which gave obscure `Q100…` items in string order; examples are now the items in the most Wikipedias among those with the feature in their strongest three (c77ca37).
- `codes.parquet` holds 31,682,565 items (`id`, `features`, `activations`, strongest first), joined by a third pass over the claims; `--no-items` reuses it.
- Feature 2479 (MathWorld ID, nLab ID, ProofWiki ID) is active on 6,321 items, including Hilbert space, Fourier transform, wavelet, Kalman filter and group (Q83478); its parent is a Microsoft Academic ID, Freebase ID, Encyclopedia of Life ID feature.
- The number of features per item varies with the item's identifiers under the fixed threshold: Emmy Noether (Q7099) 61, Hilbert space (Q190056) 22, Kalman filter (Q846780) 13, image moment (Q841934) 1.
- The 64 group-0 features include about 15 film/TV and about 15 national-library features, while SIMBAD/Gaia stars first appear in group 2 (feature 1020) and genes in group 3 (feature 1652).

### Neighbours (sae/neighbours.py, sae/neighbours.sh)

- sae/neighbours.py weights each item's activations by the feature's idf (log of coded items over the feature's items), takes as candidates the items with any of the seed's 8 heaviest features, and ranks them by cosine of the weighted codes, grouped by the shared feature contributing most.
- The Kalman filter's nearest items include random walk, Monte Carlo method, control theory, martingale and dynamical system, and economics topics through feature 2144 (STW Thesaurus for Economics ID) (sae/output/neighbours_Q846780.stdout).
- sae/neighbours.sh with no arguments runs 8 seeds (Kalman filter, Hilbert space, image moment, Emmy Noether, Casablanca, red fox, caffeine, Tetris), each to `sae/output/neighbours_<QID>.stdout`.

### Publishing (sae/publish.py, sae/publish.sh, sae/dataset_card.md, sae/upload.sh)

- sae/publish.py writes `sae/output/publish/`: `items.parquet` (with `label`, `weights`, `norm`, sorted by id in row groups of 20,000), `postings.parquet` (feature, rank, id, weight, norm, sorted by feature and rank), `features.parquet` (with `idf`, `children`), `id_properties.parquet` and `model/`.
- The run with postings capped at 20,000 per feature wrote items.parquet 951.0 MB, postings.parquet 23,002,234 rows and 255.1 MB, model/ae.pt 254.1 MB (sae/output/publish.stdout).
- With the 20,000 cap, feature 0's postings stop at weight 8.23 and the Kalman filter's own weight on it is 7.74, so the card's neighbour query on the capped files returned statistic replication, rate of return and personalization first — the same query over uncapped postings (98,172,844 (feature, item) pairs) returns Monte Carlo method, fixed point and random walk among the first, in 1.1 s on the local files (DuckDB 1.5.6).
- sae/publish.py keeps every posting by default (8ab5882).
- The two SQL queries in sae/dataset_card.md run on the local publish files with DuckDB 1.5.6, read with the `hf://datasets/permutans/wikidata-id-matryoshka-sae-features/` prefix removed.
- sae/upload.sh copies sae/dataset_card.md to `sae/output/publish/README.md` and runs `hf upload` to `permutans/wikidata-id-matryoshka-sae-features` (or `$REPO`) as a dataset.

## Missing

- sae/publish.sh has not been rerun since postings were uncapped: `sae/output/publish/postings.parquet` holds the capped 23,002,234 rows.
- The dataset has not been uploaded.
- No Space reads the published files (sae/publish.py's docstring names `space/`, which holds no files).
- Items with fewer than 2 kept identifiers have no code in `codes.parquet` or `items.parquet`.

## Divergence

- sae/dataset_card.md gives `postings.parquet` 98.2M rows, and the files in `sae/output/publish/` hold 23,002,234.
