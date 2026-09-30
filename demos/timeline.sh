#!/usr/bin/env bash
# Run demos/timeline.py on the queries in its docstring, writing each output to
# demos/output/timeline[-{args}].stdout, and the time each took to the terminal.
#
#   demos/timeline.sh                     # local copy in hub/, `python` on the PATH
#   DATA=wikidata PYTHON="uv run python" demos/timeline.sh
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python}
DATA=${DATA:-hub}
OUT=demos/output
mkdir -p "$OUT"

run() {
  local name=timeline arg
  for arg in "$@"; do name+="-${arg#--}"; done
  local start=$SECONDS
  $PYTHON demos/timeline.py "$@" --data "$DATA" >"$OUT/$name.stdout"
  echo "$name: $((SECONDS - start)) s -> $OUT/$name.stdout"
}

run
run --lang fr
run Q1741
