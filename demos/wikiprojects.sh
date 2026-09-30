#!/usr/bin/env bash
# Run demos/wikiprojects.py (one pass over the claims), writing each output to
# demos/output/wikiprojects[-{args}].stdout, and the time each took to the terminal.
#
#   demos/wikiprojects.sh                     # local copy in hub/, `python` on the PATH
#   DATA=wikidata PYTHON="uv run python" demos/wikiprojects.sh
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python}
DATA=${DATA:-hub}
OUT=demos/output
mkdir -p "$OUT"

run() {
  local name=wikiprojects arg
  for arg in "$@"; do name+="-${arg#--}"; done
  local start=$SECONDS
  $PYTHON demos/wikiprojects.py "$@" --data "$DATA" >"$OUT/$name.stdout"
  echo "$name: $((SECONDS - start)) s -> $OUT/$name.stdout"
}

run
