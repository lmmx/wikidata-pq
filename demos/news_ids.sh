#!/usr/bin/env bash
# The properties that hold a news or media outlet's ids, with the outlet, its country,
# how many items use each and its URL pattern, to demos/output/news_ids.stdout. Extra
# arguments go to demos/news_ids.py.
#
#   demos/news_ids.sh                     # local copy in hub/, `python` on the PATH
#   DATA=wikidata PYTHON="uv run python" demos/news_ids.sh --lang fr
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p demos/output
start=$SECONDS
${PYTHON:-python} demos/news_ids.py --data "${DATA:-hub}" "$@" >demos/output/news_ids.stdout
echo "news_ids: $((SECONDS - start)) s -> demos/output/news_ids.stdout"
