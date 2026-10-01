#!/usr/bin/env bash
# Make a published run the Space's default: the Space opens the last run in sae/runs.json, so
# this moves the run to the end of it and uploads only runs.json to the dataset repo (the
# run's tables must be uploaded already, by sae/upload.sh). The other runs stay listed, at
# ?run=. Needs `hf auth login` first; commit sae/runs.json afterwards.
#
#   RUN=v1 sae/default.sh                             # permutans/wikidata-id-matryoshka-sae-features
#   RUN=v1 REPO=someone/other-name sae/default.sh
set -euo pipefail

cd "$(dirname "$0")/.."
run=${RUN:?name the run, e.g. RUN=v1}
repo=${REPO:-permutans/wikidata-id-matryoshka-sae-features}
python3 - "$run" <<'PY'
import json
import sys

run = sys.argv[1]
runs = json.load(open("sae/runs.json"))
if run not in [r["run"] for r in runs]:
    sys.exit(f"{run} is not in sae/runs.json: {', '.join(r['run'] for r in runs)}")
runs.sort(key=lambda r: r["run"] == run)  # stable: the others keep their order
with open("sae/runs.json", "w") as f:
    f.write(json.dumps(runs, indent=2) + "\n")
print("sae/runs.json:", ", ".join(r["run"] for r in runs), f"(default {run})")
PY
hf upload "$repo" sae/runs.json runs.json --repo-type dataset --commit-message "Default run: $run"
