#!/usr/bin/env bash
# Upload the published tables (from sae/publish.sh) and the card (sae/dataset_card.md) to the
# dataset repo on the Hub. Needs `hf auth login` first.
#
#   sae/upload.sh                                     # permutans/wikidata-id-matryoshka-sae-features
#   REPO=someone/other-name sae/upload.sh
set -euo pipefail

cd "$(dirname "$0")/.."
repo=${REPO:-permutans/wikidata-id-matryoshka-sae-features}
cp sae/dataset_card.md sae/output/publish/README.md
hf upload "$repo" sae/output/publish . --repo-type dataset \
  --commit-message "Upload Matryoshka SAE features and item codes"
echo "https://huggingface.co/datasets/$repo"
