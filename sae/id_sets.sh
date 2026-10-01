#!/usr/bin/env bash
# Each item's set of external-ID properties, counted by distinct set (one pass over the
# claims), to sae/output/id_sets.parquet and id_properties.parquet, with the counts in
# sae/output/id_sets.stdout; for a release (RELEASE=, see sae/release.sh), from its local
# copy to sae/output/releases/$RELEASE/. Extra arguments go to sae/id_sets.py.
#
#   sae/id_sets.sh                     # local copy in hub/, `python` on the PATH
#   RELEASE=20260928 sae/id_sets.sh    # releases/20260928/hub/
#   DATA=wikidata PYTHON="uv run python" sae/id_sets.sh --min-items 100
set -euo pipefail

cd "$(dirname "$0")/.."
source sae/release.sh
mkdir -p "$inputs"
start=$SECONDS
${PYTHON:-python} sae/id_sets.py --data "$data" --out "$inputs" "$@" >"$inputs/id_sets.stdout"
echo "id_sets: $((SECONDS - start)) s -> $inputs/id_sets.stdout"
