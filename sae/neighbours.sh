#!/usr/bin/env bash
# The items most like a seed by the autoencoder's features (from sae/export.sh), each seed
# to sae/output/neighbours_<seed>.stdout. With no arguments, a set of seeds from several
# domains; otherwise one seed (QID or English label), with extra arguments going to
# sae/neighbours.py.
#
#   sae/neighbours.sh                          # the seeds below
#   sae/neighbours.sh "Hilbert space" --top 50
set -euo pipefail

cd "$(dirname "$0")/.."

run() {
  local seed=$1 name start=$SECONDS
  shift
  name=$(echo "$seed" | tr -c 'A-Za-z0-9\n' '_')
  ${PYTHON:-uv run --group sae python} sae/neighbours.py "$seed" --data "${DATA:-hub}" "$@" \
    >"sae/output/neighbours_$name.stdout"
  echo "neighbours $seed: $((SECONDS - start)) s -> sae/output/neighbours_$name.stdout"
}

if [ $# -gt 0 ]; then
  run "$@"
  exit
fi

seeds=(
  Q846780  # Kalman filter
  Q190056  # Hilbert space
  Q841934  # image moment (one feature: a thin item)
  Q7099    # Emmy Noether
  Q132689  # Casablanca (film)
  Q8332    # red fox
  Q60235   # caffeine
  Q71910   # Tetris
)
for seed in "${seeds[@]}"; do
  run "$seed"
done
