#!/usr/bin/env bash
# A run's trained autoencoder (from sae/train.sh) as tables, in its folder sae/output/$RUN:
# features.parquet (what each feature is) and codes.parquet (each item's
# features, one pass over the claims), with the features that raise MathWorld (P2812) and
# nLab (P4215) most, in export.stdout. Extra arguments go to sae/export.py.
#
#   RUN=v1 sae/export.sh             # the `sae` dependency group, local copy in hub/
#   RUN=v0 sae/export.sh --no-items  # reuse the run's codes.parquet
set -euo pipefail

cd "$(dirname "$0")/.."
dir=sae/output/${RUN:?name the run, e.g. RUN=v0}
mkdir -p "$dir"
start=$SECONDS
${PYTHON:-uv run --group sae python} sae/export.py --find P2812 P4215 \
  --model "$dir/sae/trainer_0/ae.pt" --out "$dir" --data "${DATA:-hub}" "$@" >"$dir/export.stdout"
echo "export: $((SECONDS - start)) s -> $dir/export.stdout"
