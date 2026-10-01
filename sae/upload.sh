#!/usr/bin/env bash
# Upload a run's published tables (from sae/publish.sh, run v1 by default) and the card
# (sae/dataset_card.md) to the dataset repo on the Hub. Needs `hf auth login` first.
#
#   sae/upload.sh                                     # permutans/wikidata-id-matryoshka-sae-features
#   REPO=someone/other-name sae/upload.sh
set -euo pipefail

cd "$(dirname "$0")/.."
repo=${REPO:-permutans/wikidata-id-matryoshka-sae-features}
dir=sae/output/${RUN-v1}/publish
cp sae/dataset_card.md "$dir/README.md"
hf upload "$repo" "$dir" . --repo-type dataset \
  --commit-message "Upload Matryoshka SAE features and item codes"
echo "https://huggingface.co/datasets/$repo"
