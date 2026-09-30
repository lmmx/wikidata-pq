#!/usr/bin/env bash
# The items with a Google Knowledge Graph ID (P2671): what they are, the best known, a random sample, and the
# properties over-represented among them (one pass over the claims), to
# demos/output/knowledge_graph.stdout. Extra arguments go to demos/bearers.py.
#
#   demos/knowledge_graph.sh                     # local copy in hub/, `python` on the PATH
#   DATA=wikidata PYTHON="uv run python" demos/knowledge_graph.sh --lang fr
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p demos/output
start=$SECONDS
${PYTHON:-python} demos/bearers.py P2671 --pattern '^/(\w+)/' --data "${DATA:-hub}" "$@" >demos/output/knowledge_graph.stdout
echo "knowledge_graph: $((SECONDS - start)) s -> demos/output/knowledge_graph.stdout"
