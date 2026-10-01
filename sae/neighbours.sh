#!/usr/bin/env bash
# The items most like a seed by the autoencoder's features (from sae/export.sh), to
# sae/output/neighbours_<seed>.stdout. The seed is a QID or English label (default the
# Kalman filter); extra arguments go to sae/neighbours.py.
#
#   sae/neighbours.sh                          # Kalman filter (Q846780)
#   sae/neighbours.sh "Hilbert space" --top 50
set -euo pipefail

cd "$(dirname "$0")/.."
seed=${1:-Q846780}
shift || true
name=$(echo "$seed" | tr -c 'A-Za-z0-9\n' '_')
start=$SECONDS
${PYTHON:-uv run --group sae python} sae/neighbours.py "$seed" --data "${DATA:-hub}" "$@" \
  >"sae/output/neighbours_$name.stdout"
echo "neighbours: $((SECONDS - start)) s -> sae/output/neighbours_$name.stdout"
