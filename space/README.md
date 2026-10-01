---
title: Wikidata ID Features
emoji: 🪆
colorFrom: indigo
colorTo: pink
sdk: static
app_file: index.html
pinned: false
license: cc0-1.0
short_description: Wikidata items by the catalogues that cover them
datasets:
- permutans/wikidata-id-matryoshka-sae-features
---

Search a Wikidata item by name to see its features, learned by a Matryoshka sparse
autoencoder from which external identifiers each item has, and the items most like it; or
browse the features from broad to specific. The page reads
[permutans/wikidata-id-matryoshka-sae-features](https://huggingface.co/datasets/permutans/wikidata-id-matryoshka-sae-features)
in the browser with DuckDB-WASM, over HTTP range requests: no server.

Code: [lmmx/wikidata-pq/sae](https://github.com/lmmx/wikidata-pq/tree/master/sae).
