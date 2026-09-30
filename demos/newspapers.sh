#!/usr/bin/env bash
# The items bearing any newspaper's id: the properties from demos/news_ids.py
# (--ids-only), passed to demos/bearers.py, to demos/output/newspapers.stdout. Extra
# arguments go to demos/bearers.py.
#
#   demos/newspapers.sh                     # local copy in hub/, `python` on the PATH
#   DATA=wikidata PYTHON="uv run python" demos/newspapers.sh --top 30
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p demos/output
start=$SECONDS
read -ra props < <(${PYTHON:-python} demos/news_ids.py --ids-only --data "${DATA:-hub}")
echo "newspapers: ${#props[@]} properties"
${PYTHON:-python} demos/bearers.py "${props[@]}" --data "${DATA:-hub}" "$@" \
  >demos/output/newspapers.stdout
echo "newspapers: $((SECONDS - start)) s -> demos/output/newspapers.stdout"
