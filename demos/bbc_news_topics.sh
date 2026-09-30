#!/usr/bin/env bash
# The items with a BBC News topic ID (P6200): what they are, the best known, a random sample, and the
# properties over-represented among them (one pass over the claims), to
# demos/output/bbc_news_topics.stdout. Extra arguments go to demos/bearers.py.
#
#   demos/bbc_news_topics.sh                     # local copy in hub/, `python` on the PATH
#   DATA=wikidata PYTHON="uv run python" demos/bbc_news_topics.sh --lang fr
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p demos/output
start=$SECONDS
${PYTHON:-python} demos/bearers.py P6200 --data "${DATA:-hub}" "$@" >demos/output/bbc_news_topics.stdout
echo "bbc_news_topics: $((SECONDS - start)) s -> demos/output/bbc_news_topics.stdout"
