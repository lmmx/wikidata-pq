#!/usr/bin/env bash
# Each item's set of external-ID properties, counted by distinct set (one pass over the
# claims), to sae/output/id_sets.parquet and id_properties.parquet, with the counts in
# sae/output/id_sets.stdout. Extra arguments go to sae/id_sets.py.
#
#   sae/id_sets.sh                     # local copy in hub/, `python` on the PATH
#   DATA=wikidata PYTHON="uv run python" sae/id_sets.sh --min-items 100
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p sae/output
start=$SECONDS
${PYTHON:-python} sae/id_sets.py --data "${DATA:-hub}" "$@" >sae/output/id_sets.stdout
echo "id_sets: $((SECONDS - start)) s -> sae/output/id_sets.stdout"
