#!/usr/bin/env bash
# The items with a BBC Things ID (P1617): what they are, the best known, a random sample, and the
# properties over-represented among them (one pass over the claims), to
# demos/output/bbc_things.stdout. Extra arguments go to demos/bearers.py.
#
#   demos/bbc_things.sh                     # local copy in hub/, `python` on the PATH
#   DATA=wikidata PYTHON="uv run python" demos/bbc_things.sh --lang fr
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p demos/output
start=$SECONDS
${PYTHON:-python} demos/bearers.py P1617 --data "${DATA:-hub}" "$@" >demos/output/bbc_things.stdout
echo "bbc_things: $((SECONDS - start)) s -> demos/output/bbc_things.stdout"
