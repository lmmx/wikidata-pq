#!/usr/bin/env bash
# Run demos/divisions.py on the queries in its docstring, writing each output to
# demos/output/divisions[-{args}].stdout, and the time each took to the terminal.
#
#   demos/divisions.sh                     # local copy in hub/, `python` on the PATH
#   DATA=wikidata PYTHON="uv run python" demos/divisions.sh
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python}
DATA=${DATA:-hub}
OUT=demos/output
mkdir -p "$OUT"

run() {
  local name=divisions arg
  for arg in "$@"; do name+="-${arg#--}"; done
  local start=$SECONDS
  $PYTHON demos/divisions.py "$@" --data "$DATA" >"$OUT/$name.stdout"
  echo "$name: $((SECONDS - start)) s -> $OUT/$name.stdout"
}

run
run Q183 --lang de
run Q30
run Q142 --lang fr
