#!/usr/bin/env bash
# Upload a run's published tables (from sae/publish.sh) to the folder of that name in the
# dataset repo on the Hub, with the card (sae/dataset_card.md) and the list of runs
# (sae/runs.json, which must name the run) at the top. Needs `hf auth login` first.
#
#   RUN=v0 sae/upload.sh                              # permutans/wikidata-id-matryoshka-sae-features
#   RUN=v1 REPO=someone/other-name sae/upload.sh
set -euo pipefail

cd "$(dirname "$0")/.."
run=${RUN:?name the run, e.g. RUN=v0}
repo=${REPO:-permutans/wikidata-id-matryoshka-sae-features}
dir=sae/output/$run/publish
if ! python3 -c "import json, sys; sys.exit(sys.argv[1] not in [r['run'] for r in json.load(open('sae/runs.json'))])" "$run"; then
  echo "add $run to sae/runs.json first" >&2
  exit 1
fi
hf upload "$repo" "$dir" "$run" --repo-type dataset --commit-message "Upload run $run"
top=$(mktemp -d)
cp sae/dataset_card.md "$top/README.md"
cp sae/runs.json "$top/runs.json"
hf upload "$repo" "$top" . --repo-type dataset --commit-message "Update the card and runs.json"
rm -r "$top"
echo "https://huggingface.co/datasets/$repo/tree/main/$run"
