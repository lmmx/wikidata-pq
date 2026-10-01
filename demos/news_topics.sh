#!/usr/bin/env bash
# The topics of news websites' coverage: the items holding any news website topic ID,
# the properties from demos/news_ids.py (--ids-only) passed to demos/bearers.py, to
# demos/output/news_topics.stdout. Extra arguments go to demos/bearers.py.
#
#   demos/news_topics.sh                     # local copy in hub/, `python` on the PATH
#   DATA=wikidata PYTHON="uv run python" demos/news_topics.sh --top 30
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p demos/output
start=$SECONDS
read -ra props < <(${PYTHON:-python} demos/news_ids.py --ids-only --data "${DATA:-hub}")
echo "news_topics: ${#props[@]} properties"
${PYTHON:-python} demos/bearers.py "${props[@]}" --data "${DATA:-hub}" "$@" \
  >demos/output/news_topics.stdout
echo "news_topics: $((SECONDS - start)) s -> demos/output/news_topics.stdout"
