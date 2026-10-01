#!/usr/bin/env bash
# Create the static Space (if need be) and upload space/ to it. Needs `hf auth login` first.
#
#   space/upload.sh                                   # permutans/wikidata-id-features
#   REPO=someone/other-name space/upload.sh
set -euo pipefail

cd "$(dirname "$0")/.."
repo=${REPO:-permutans/wikidata-id-features}
hf repos create "$repo" --repo-type space --sdk static --exist-ok
hf upload "$repo" space . --repo-type space --exclude upload.sh \
  --commit-message "Update the Space"
echo "https://huggingface.co/spaces/$repo"
