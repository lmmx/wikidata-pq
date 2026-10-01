#!/usr/bin/env bash
# A run's results as tables to publish (from sae/export.sh), to sae/output/$RUN/publish/
# (default run v1), with their sizes in its publish.stdout. Extra arguments go
# to sae/publish.py.
#
#   sae/publish.sh                     # local copy in hub/, `python` on the PATH
#   sae/publish.sh --postings 50000
set -euo pipefail

cd "$(dirname "$0")/.."
dir=sae/output/${RUN-v1}
start=$SECONDS
${PYTHON:-python} sae/publish.py --sae "$dir" --out "$dir/publish" --data "${DATA:-hub}" "$@" \
  >"$dir/publish.stdout"
echo "publish: $((SECONDS - start)) s -> $dir/publish.stdout"
