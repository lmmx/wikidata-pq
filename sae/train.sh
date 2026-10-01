#!/usr/bin/env bash
# Train the Matryoshka sparse autoencoder on sae/output/id_sets.parquet (from
# sae/id_sets.sh) as a new run, in sae/output/$RUN: the model at sae/trainer_0/ae.pt with
# the run's settings in sae/run.json, and the progress and a first look at the features in
# train.stdout. Refuses a run that has a model. Extra arguments go to sae/train.py; add the
# run to sae/runs.json to publish it. With RELEASE=, on that release's identifier sets (see
# sae/release.sh), recorded in the run's folder for its later steps.
#
#   RUN=v1 sae/train.sh                # the `sae` dependency group, sae/train.py's defaults
#   RUN=v2 RELEASE=20260928 sae/train.sh
#   RUN=v3 sae/train.sh --k 12
set -euo pipefail

cd "$(dirname "$0")/.."
dir=sae/output/${RUN:?name the run, e.g. RUN=v0}
if [ -e "$dir/sae/trainer_0/ae.pt" ]; then
  echo "$dir already has a model; name a new run" >&2
  exit 1
fi
source sae/release.sh
mkdir -p "$dir"
if [ -n "$release" ]; then echo "$release" >"$dir/release"; fi
start=$SECONDS
${PYTHON:-uv run --group sae python} sae/train.py --out "$dir/sae" \
  --sets "$inputs/id_sets.parquet" --properties "$inputs/id_properties.parquet" "$@" | tee "$dir/train.stdout"
echo "train: $((SECONDS - start)) s -> $dir/train.stdout"
