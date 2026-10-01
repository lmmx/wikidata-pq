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
- sae/train.py defaults to `--alpha 0.75` and `--samples 200e6` after the first run (0.5, 100M).
- The runners (sae/train.sh, sae/export.sh, sae/neighbours.sh, sae/publish.sh, sae/upload.sh) exit unless `RUN` names a run, and read and write `sae/output/$RUN`; sae/train.sh exits when `$RUN/sae/trainer_0/ae.pt` exists.
- The first run's model, codes, features, publish folder and logs moved from `sae/output/` to `sae/output/v0/` (`git mv` for the logs); `id_sets.parquet`, `id_properties.parquet` and `id_sets.stdout` stay in `sae/output/`, shared by the runs.
- sae/train.py writes the run's settings (alpha, samples, k, groups, batch, seed) to `--out`/run.json, and sae/publish.py copies it to `model/run.json`; `sae/output/v0/sae/run.json` was written by hand from the first run's settings (alpha 0.5, 100M sets).
- sae/runs.json lists the published runs with their settings and a note, starting with v0.
- sae/upload.sh uploads `sae/output/$RUN/publish` to the folder `$RUN/` in the dataset repo, exits unless sae/runs.json names the run, and uploads sae/dataset_card.md (as README.md) and sae/runs.json to the top of the repo.
- sae/dataset_card.md's configs and queries read `v0/`.
- space/index.html reads `runs.json` from the dataset, takes the run from `?run=` (else the last listed), reads that run's folder, shows a run picker with the run's alpha, samples and k, and labels the levels with the run's group sizes; without `runs.json` it reads the top of the repo.

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
- The uncapped publish (38 s) and the upload of 2.08 GB to `permutans/wikidata-id-matryoshka-sae-features` ran (commit 795646ec on the Hub), with the model weights in the dataset repo under `model/`.
- The Hub answers range requests for the dataset's files with `access-control-allow-origin` set, through a 302 to its CDN (curl with an `Origin` header).
- sae/publish.py writes `names.parquet` (`key` = lowercased label, `label`, `id`, English `description`, `wikipedias`), sorted by key then Wikipedias in row groups of 20,000; a prefix search `key >= q AND key < q || chr(65535)` on a 300-label sample returns the matching labels by Wikipedias (DuckDB 1.5.6).

### Space (space/index.html, space/data.js, space/README.md, space/upload.sh)

- The first space/index.html loaded DuckDB-WASM 1.33.1-dev57.0 and defined views over the Hub files of `items`, `names` and `postings` before reading `features.parquet` — on the deployed Space it stayed at "Loading the features" on an iPad and a desktop browser.
- The Parquet footers measure items.parquet 0.91 MB (1,585 row groups), postings.parquet 2.01 MB (4,909), names.parquet 0.80 MB (1,470), features.parquet under 0.01 MB (pyarrow `serialized_size`).
- space/data.js reads the Hub files with hyparquet 1.31.2 and hyparquet-compressors 1.1.2 (zstd): each file's footer once (`cachedAsyncBuffer`, 1 MiB initial fetch), then the row groups whose footer min/max statistics can hold the key looked up, read in parallel, with 64-bit integers converted to numbers.
- space/data.js `neighbours` scores the items in the postings of an item's 8 heaviest features by the dot product of weights over those features, divided by both norms, and records the feature contributing most.
- space/index.html imports hyparquet and hyparquet-compressors from jsDelivr (`+esm`) and space/data.js, loads `features.parquet` at start, and fetches the `names` and `items` footers in the background.
- space/data.js run from Node 20 (undici through the container's proxy) against the Hub files: features 4.3 s; searches for "red fox", "emmy noether" and "tetris" return Q8332, Q7099 and Q71910 first, with descriptions; the red fox lookup 2.9 s; its 30 neighbours with labels 7.3 s (muskrat, coypu, European rabbit, raccoon, brown rat via feature 5); feature 2479's 30 strongest items with labels 2.3 s (simple group, abelian group, algorithm, sine).
- Reading row groups one after another took 11.1 s for the Kalman filter's neighbours and 20.8 s for 30 labels, against 5.4 s and 2.3 s in parallel.
- space/index.html lays out an item as a header (label, description, QID link) over two panels: "What it is", the item's features in four bands by level (Broad, General, Specific, Niche, each with its count of features) as cards named by their first catalogue (the property label without " ID"), with the next catalogues as chips, the item count and a weight bar, and features under 40% of the item's top weight folded under "lighter ones"; and "Most like it", the neighbours as linked pills grouped by the feature linking them most.
- space/index.html lists an item's neighbours as rows, each with 8 dots for the item's 8 heaviest features (numbered in a key above, coloured by level), filled where the neighbour has the feature — grouping the neighbours by their exact set of shared features gave 21 groups for the Kalman filter's 36 nearest, 12 for the red fox's and 6 for Tetris's.
- Scoring neighbours over the item's 8 heaviest features only put Lockheed Martin F-22 Raptor first for French Bulldog (Q29149) and fiscal federalism first for the Kalman filter; scoring over all the item's features (read from `postings`, against the v0 files on the Hub from Node) gives Golden Retriever, Persian cat, Rottweiler and Chow Chow for French Bulldog, and random walk 0.79, urban economics 0.78, elliptic function 0.78 for the Kalman filter, as sae/neighbours.py does (sae/output/v0/neighbours_Q846780.stdout) — reading all the Kalman filter's 13 features took 11.8 s and 36 MB, and its 10 features on 250k items or fewer 1.6 s.
- space/data.js `neighbours` takes a `use` filter and a `top` count over the item's features; space/index.html scores over up to 16 of the item's features on 250k items or fewer, and shows each neighbour's cosine similarity as a number and a bar, and its dots shaded by the item's weight on each feature.
- French Bulldog (Q29149) has 20 external-ID properties, and the American Kennel Club ID (P13890, 283 items) is its only dog-specific one; none of its features in v0 is specific to dogs.
- space/data.js `description` finds an item's description in `names` by its lowercased label, so an item reached by a link shows its description as one reached by search does.
- space/index.html lays out a feature as its chain of parent features, its level, its catalogues with decoder-weight bars, its narrower features (12, then folded), its examples and its 40 strongest items as pills; the home view shows 7 example items, a "How this works" section on the nested levels, and the 64 broadest features.
- space/index.html run in jsdom 29.1.1 from Node against the Hub files renders `#item=Q846780` and `#feature=2479` with no errors.
- space/upload.sh runs `hf repos create --repo-type space --sdk static --exist-ok` and `hf upload` of space/ (without upload.sh) to `permutans/wikidata-id-features` (or `$REPO`).

## Missing

- The v1 run (`--alpha 0.75`, 200M sets) has not been trained.
- The dataset repo holds the first run's files at its top level, not under `v0/`, and has no `runs.json`.

- The hyparquet version of space/index.html has not run in a browser.
- Items with fewer than 2 kept identifiers have no code in `codes.parquet` or `items.parquet`.

