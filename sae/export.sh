#!/usr/bin/env bash
# A run's trained autoencoder (from sae/train.sh) as tables, in its folder sae/output/$RUN
# (default v1): features.parquet (what each feature is) and codes.parquet (each item's
# features, one pass over the claims), with the features that raise MathWorld (P2812) and
# nLab (P4215) most, in export.stdout. Extra arguments go to sae/export.py.
#
#   sae/export.sh                    # the `sae` dependency group, local copy in hub/
#   sae/export.sh --no-items         # reuse the run's codes.parquet
set -euo pipefail

cd "$(dirname "$0")/.."
dir=sae/output/${RUN-v1}
mkdir -p "$dir"
start=$SECONDS
${PYTHON:-uv run --group sae python} sae/export.py --find P2812 P4215 \
  --model "$dir/sae/trainer_0/ae.pt" --out "$dir" --data "${DATA:-hub}" "$@" >"$dir/export.stdout"
echo "export: $((SECONDS - start)) s -> $dir/export.stdout"
