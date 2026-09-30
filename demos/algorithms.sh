#!/usr/bin/env bash
# Run demos/algorithms.py (two passes over the claims), writing each output to
# demos/output/algorithms[-{args}].stdout, and the time each took to the terminal.
#
#   demos/algorithms.sh                     # local copy in hub/, `python` on the PATH
#   DATA=wikidata PYTHON="uv run python" demos/algorithms.sh
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python}
DATA=${DATA:-hub}
OUT=demos/output
mkdir -p "$OUT"

run() {
  local name=algorithms arg
  for arg in "$@"; do name+="-${arg#--}"; done
  local start=$SECONDS
  $PYTHON demos/algorithms.py "$@" --data "$DATA" >"$OUT/$name.stdout"
  echo "$name: $((SECONDS - start)) s -> $OUT/$name.stdout"
}

run
