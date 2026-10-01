#!/usr/bin/env bash
# The topics of news websites' coverage as demos/output/news_topics.parquet, one row per
# item holding any news website topic ID (demos/news_ids.py --ids-only), to explore in
# an embedding viewer. Extra arguments go to demos/export_bearers.py.
#
#   demos/news_topics_export.sh                     # local copy in hub/, `python` on the PATH
#   pip install embedding-atlas
#   embedding-atlas demos/output/news_topics.parquet --text text
set -euo pipefail

cd "$(dirname "$0")/.."
start=$SECONDS
read -ra props < <(${PYTHON:-python} demos/news_ids.py --ids-only --data "${DATA:-hub}")
echo "news_topics_export: ${#props[@]} properties"
${PYTHON:-python} demos/export_bearers.py "${props[@]}" --data "${DATA:-hub}" \
  --out demos/output/news_topics.parquet "$@"
echo "news_topics_export: $((SECONDS - start)) s -> demos/output/news_topics.parquet"
