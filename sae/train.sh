#!/usr/bin/env bash
# Train the Matryoshka sparse autoencoder on sae/output/id_sets.parquet (from
# sae/id_sets.sh), into the run's folder, sae/output/$RUN (default v1): the model at
# sae/trainer_0/ae.pt, and the progress and a first look at the features in train.stdout.
# Extra arguments go to sae/train.py. The first run (v0) is in sae/output itself (RUN=).
#
#   sae/train.sh                     # the `sae` dependency group, sae/train.py's defaults
#   RUN=v2 sae/train.sh --k 12
set -euo pipefail

cd "$(dirname "$0")/.."
dir=sae/output/${RUN-v1}
mkdir -p "$dir"
start=$SECONDS
${PYTHON:-uv run --group sae python} sae/train.py --out "$dir/sae" "$@" | tee "$dir/train.stdout"
echo "train: $((SECONDS - start)) s -> $dir/train.stdout"
