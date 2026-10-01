#!/usr/bin/env bash
# The items most like a seed by a run's autoencoder features (from sae/export.sh), each
# seed to sae/output/$RUN/neighbours_<seed>.stdout. With no arguments, a set
# of seeds from several domains; otherwise one seed (QID or English label), with extra
# arguments going to sae/neighbours.py.
#
#   RUN=v0 sae/neighbours.sh                   # the seeds below
#   RUN=v0 sae/neighbours.sh "Hilbert space" --top 50
set -euo pipefail

cd "$(dirname "$0")/.."
dir=sae/output/${RUN:?name the run, e.g. RUN=v0}

run() {
  local seed=$1 name start=$SECONDS
  shift
  name=$(echo "$seed" | tr -c 'A-Za-z0-9\n' '_')
  ${PYTHON:-uv run --group sae python} sae/neighbours.py "$seed" --out "$dir" --data "${DATA:-hub}" "$@" \
    >"$dir/neighbours_$name.stdout"
  echo "neighbours $seed: $((SECONDS - start)) s -> $dir/neighbours_$name.stdout"
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
