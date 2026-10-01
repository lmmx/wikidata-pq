#!/usr/bin/env bash
# The autoencoder's results as tables to publish (from sae/export.sh), to
# sae/output/publish/, with their sizes in sae/output/publish.stdout. Extra arguments go
# to sae/publish.py.
#
#   sae/publish.sh                     # local copy in hub/, `python` on the PATH
#   sae/publish.sh --postings 50000
set -euo pipefail

cd "$(dirname "$0")/.."
start=$SECONDS
${PYTHON:-python} sae/publish.py --data "${DATA:-hub}" "$@" >sae/output/publish.stdout
echo "publish: $((SECONDS - start)) s -> sae/output/publish.stdout"
