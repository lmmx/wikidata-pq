#!/usr/bin/env bash
# Train the Matryoshka sparse autoencoder on sae/output/id_sets.parquet (from
# sae/id_sets.sh) as a new run, in sae/output/$RUN: the model at sae/trainer_0/ae.pt with
# the run's settings in sae/run.json, and the progress and a first look at the features in
# train.stdout. Refuses a run that has a model. Extra arguments go to sae/train.py; add the
# run to sae/runs.json to publish it.
#
#   RUN=v1 sae/train.sh                # the `sae` dependency group, sae/train.py's defaults
#   RUN=v2 sae/train.sh --k 12
set -euo pipefail

cd "$(dirname "$0")/.."
dir=sae/output/${RUN:?name the run, e.g. RUN=v0}
if [ -e "$dir/sae/trainer_0/ae.pt" ]; then
  echo "$dir already has a model; name a new run" >&2
  exit 1
fi
mkdir -p "$dir"
start=$SECONDS
${PYTHON:-uv run --group sae python} sae/train.py --out "$dir/sae" "$@" | tee "$dir/train.stdout"
echo "train: $((SECONDS - start)) s -> $dir/train.stdout"
