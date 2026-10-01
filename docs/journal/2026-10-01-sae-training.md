# 2026-10-01: Training a Matryoshka SAE on items' external identifiers

## Current State

### Identifier sets (sae/id_sets.py, sae/id_sets.sh)

- sae/id_sets.py makes two passes over the claims, a file at a time, over non-deprecated statements of datatype `external-id`: the first counts the items holding each property, the second writes each file's sets of kept property indexes, counted by distinct set, to `sae/output/parts/`, merged by the streaming engine into `id_sets.parquet`.
- The first version built every distinct set of all properties in memory and filtered and regrouped it after the pass — the process was killed after the 34-file pass ("Terminated"), and the two-pass version replaced it (de75f07).
- The run on hub/ found 53,585,955 items with an external identifier over 9,778 properties, and kept 7,752 properties (held by 50+ items) and 31,907,221 items with 2+ of them, in 4,007,832 distinct sets (sae/output/id_sets.stdout).
- The most common set, SIMBAD ID + Gaia ID, holds 4,567,864 items (14.3% of the kept items); the top 100 sets hold 46.5%.
- DOI (P356) is on 359,109 items in the claims, so scholarly articles take little of the sets.

### Training (sae/train.py, sae/train.sh)

- sae/train.py feeds `dictionary_learning`'s `trainSAE` with `MatryoshkaBatchTopKTrainer` (PyPI `dictionary-learning` 0.1.0, the `sae` dependency group in pyproject.toml), with groups 64/192/768/3072, `k = 8`, learning-rate decay from 80% of the steps, and placeholder `layer`/`lm_name` arguments the trainer asserts.
- Batches are dense 0/1 vectors built on the GPU from CSR arrays of `id_sets.parquet`, each set drawn with probability proportional to `items ** alpha`, with 1% of the sets held out (sae/train.py:`batch`, `draws`).
- An earlier hand-written trainer (22c2d99) differed from `dictionary_learning`'s in decoder normalisation, the aux-k loss for dead features, gradient clipping, threshold start and decay, and learning rate, and was replaced by the library trainer (32e9351).
- The trainer's loss is squared error, applied here to 0/1 inputs.
- The first run, v0 (alpha 0.5, 100M sets, 24,415 steps of 4,096, 93 minutes at 4.38 steps/s on an RTX 3090), reached 0.907 fraction of variance explained at step 24,000; on held-out sets, recall at set size was 0.958, with 8.0 features active per set and 1,607 of 4,096 features never active across 102,400 held-out draws (sae/output/v0/train.stdout).
- sae/train.py defaults to `--alpha 0.75` and `--samples 200e6` after v0, and writes the run's settings (alpha, samples, k, groups, batch, seed) to `--out`/run.json.

### Runs (sae/*.sh, sae/runs.json)

- The runners (sae/train.sh, sae/export.sh, sae/neighbours.sh, sae/publish.sh, sae/upload.sh) exit unless `RUN` names a run, and read and write `sae/output/$RUN`; sae/train.sh exits when `$RUN/sae/trainer_0/ae.pt` exists.
- v0's model, codes, features, publish folder and logs moved from `sae/output/` to `sae/output/v0/` (`git mv` for the logs); `id_sets.parquet`, `id_properties.parquet` and `id_sets.stdout` stay in `sae/output/`, shared by the runs; `sae/output/v0/sae/run.json` was written by hand from v0's settings.
- sae/runs.json lists the published runs with their settings and a note, starting with v0.

### Export (sae/export.py, sae/export.sh)

- sae/export.py encodes all 4,007,832 sets with the trainer's threshold: 9.0 features active per set, and 5,803 sets with none (sae/output/v0/export.stdout).
- Live features by group, counted over items: 64 of 64, 189 of 192, 738 of 768, 2,587 of 3,072 (3,578 of 4,096).
- A feature's parent is the broader-group feature with the largest share of co-active items; the first export gave never-active features feature 0 (Gran Enciclopèdia Catalana) as parent, 518 of the 667 features with parent 0 — never-active features now have no parent (c77ca37).
- Examples were first the lowest id of each feature's most common sets, which gave obscure `Q100…` items in string order; examples are now the items in the most Wikipedias among those with the feature in their strongest three (c77ca37).
- `codes.parquet` holds 31,682,565 items (`id`, `features`, `activations`, strongest first), joined by a third pass over the claims; `--no-items` reuses it.
- Feature 2479 (MathWorld ID, nLab ID, ProofWiki ID) is active on 6,321 items, including Hilbert space, Fourier transform, wavelet, Kalman filter and group (Q83478); its parent is a Microsoft Academic ID, Freebase ID, Encyclopedia of Life ID feature.
- The number of features per item varies with the item's identifiers under the fixed threshold: Emmy Noether (Q7099) 61, Hilbert space (Q190056) 22, Kalman filter (Q846780) 13, image moment (Q841934) 1.
- The 64 group-0 features include about 15 film/TV and about 15 national-library features, while SIMBAD/Gaia stars first appear in group 2 (feature 1020) and genes in group 3 (feature 1652).

### Neighbours from the command line (sae/neighbours.py, sae/neighbours.sh)

- sae/neighbours.py weights each item's activations by the feature's idf (log of coded items over the feature's items), takes as candidates the items with any of the seed's 8 heaviest features, and ranks them by cosine of the weighted codes over all shared features, grouped by the shared feature contributing most.
- The Kalman filter's nearest items include random walk, Monte Carlo method, control theory, martingale and dynamical system, and economics topics through feature 2144 (STW Thesaurus for Economics ID) (sae/output/v0/neighbours_Q846780.stdout).
- sae/neighbours.sh with no arguments runs 8 seeds (Kalman filter, Hilbert space, image moment, Emmy Noether, Casablanca, red fox, caffeine, Tetris), each to `sae/output/$RUN/neighbours_<QID>.stdout`.

### v1 against v0 (sae/output/v1/)

- v1 (alpha 0.75, 200M sets, 48,829 steps, 3 h 26 min at 3.95 steps/s) reached 0.953 fraction of variance explained at step 48,000 (v0: 0.907 at 24,000); held out, recall at set size 0.976, 8.3 features active, 1,270 never active — not comparable with v0's held-out figures, since the held-out draws are weighted by each run's own alpha.
- Over all sets (sae/output/v1/export.stdout): 11.5 features active per set (v0 9.0), 2,422 sets with none (5,803), 31,887,664 coded items (31,682,565).
- Live features by group: 64, 192, 741, 2,281 (3,278 of 4,096; v0 3,578); median items per feature 216,686 / 48,098 / 12,152 / 3,514 (v0 131,902 / 26,675 / 6,694 / 652) — broader features, fewer live fine ones.
- Group 0 shifts from films and libraries toward places and taxa: GeoNames (feature 22, 3.0M items), Encyclopedia of Life (5), WoRMS (28) and OpenStreetMap relation (53) are among the 20 broadest.
- Maths splits into two features: 1281 (nLab, ProofWiki, Encyclopedia of Mathematics; 4,221 items) and 1237 (MathWorld, ProofWiki, Encyclopedia of Mathematics; 6,230 items, numbers among its examples); v0 had one, 2479.
- Seed neighbours (sae/output/v1/neighbours_*.stdout):
  - Kalman filter: robust statistics, evolutionary algorithm, Voronoi diagram, computer algebra, Gaussian process via feature 3590 (GitHub topic, GitLab topic); v0 gave random walk, urban economics, elliptic function, neuroeconomics.
  - red fox: brown rat, Rattus rattus, European rabbit, red deer, stoat; v0 had house mouse and muskrat in place of the deer and stoat.
  - Hilbert space: metric space, topological space, hyperbolic geometry, algebraic topology, harmonic analysis; v0 had Diophantine equation fourth.
  - Emmy Noether: Riemann, Minkowski, Weierstraß, Élie Cartan, Cantor (0.70–0.68); v0 led with Nicolas Carnot (0.65).
  - caffeine: aspirin, urea, formaldehyde, carbon dioxide, citric acid via a hazardous-chemicals feature (203: ZVG, HSDB, CAMEO); v0 gave aspirin, folic acid, progesterone, salicylic acid, theophylline via a drugs feature (274: ATC, RxNorm, DrugCentral).
  - Casablanca and Tetris: much the same films and games in both; image moment: still a near-empty code (3 features), every neighbour at 1.0.
- sae/runs.json lists v1 after v0, so the Space opens v1 and keeps v0 under `?run=v0`.

## Missing

- v1 is not yet published or uploaded (`RUN=v1 sae/publish.sh && RUN=v1 sae/upload.sh`); its postings will be about a quarter larger than v0's, with 11.5 features per set against 9.0.
- Items with fewer than 2 kept identifiers have no code in `codes.parquet` or `items.parquet`.
