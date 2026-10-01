#!/usr/bin/env bash
# Train the Matryoshka sparse autoencoder on sae/output/id_sets.parquet (from
# sae/id_sets.sh), to sae/output/sae/trainer_0/ae.pt, with the progress and a first look at the
# features in sae/output/train.stdout. Extra arguments go to sae/train.py.
#
#   sae/train.sh                     # the `sae` dependency group (dictionary-learning)
#   sae/train.sh --k 12 --alpha 0.3
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p sae/output
start=$SECONDS
${PYTHON:-uv run --group sae python} sae/train.py "$@" | tee sae/output/train.stdout
echo "train: $((SECONDS - start)) s -> sae/output/train.stdout"
