#!/usr/bin/env bash
# Run demos/concepts.py (two passes over the claims), writing each output to
# demos/output/concepts[-{args}].stdout, and the time each took to the terminal.
#
#   demos/concepts.sh                     # local copy in hub/, `python` on the PATH
#   DATA=wikidata PYTHON="uv run python" demos/concepts.sh
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python}
DATA=${DATA:-hub}
OUT=demos/output
mkdir -p "$OUT"

run() {
  local name=concepts arg
  for arg in "$@"; do name+="-${arg#--}"; done
  local start=$SECONDS
  $PYTHON demos/concepts.py "$@" --data "$DATA" >"$OUT/$name.stdout"
  echo "$name: $((SECONDS - start)) s -> $OUT/$name.stdout"
}

run
