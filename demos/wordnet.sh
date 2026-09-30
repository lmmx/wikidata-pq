#!/usr/bin/env bash
# The items with a WordNet 3.1 Synset ID (P8814): what they are, the best known, a random sample, and the
# properties over-represented among them (one pass over the claims), to
# demos/output/wordnet.stdout. Extra arguments go to demos/bearers.py.
#
#   demos/wordnet.sh                     # local copy in hub/, `python` on the PATH
#   DATA=wikidata PYTHON="uv run python" demos/wordnet.sh --lang fr
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p demos/output
start=$SECONDS
${PYTHON:-python} demos/bearers.py P8814 --pattern='-(\w)$' --data "${DATA:-hub}" "$@" >demos/output/wordnet.stdout
echo "wordnet: $((SECONDS - start)) s -> demos/output/wordnet.stdout"
