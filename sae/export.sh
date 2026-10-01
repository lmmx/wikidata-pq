#!/usr/bin/env bash
# The trained autoencoder (from sae/train.sh) as tables: sae/output/features.parquet (what
# each feature is) and codes.parquet (each item's features, one pass over the claims), with
# the features that raise MathWorld (P2812) and nLab (P4215) most, in
# sae/output/export.stdout. Extra arguments go to sae/export.py.
#
#   sae/export.sh                    # the `sae` dependency group, local copy in hub/
#   sae/export.sh --no-items         # features only
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p sae/output
start=$SECONDS
${PYTHON:-uv run --group sae python} sae/export.py --find P2812 P4215 \
  --data "${DATA:-hub}" "$@" >sae/output/export.stdout
echo "export: $((SECONDS - start)) s -> sae/output/export.stdout"
