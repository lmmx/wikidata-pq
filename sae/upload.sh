#!/usr/bin/env bash
# Upload a run's published tables (from sae/publish.sh) to the folder of that name in the
# dataset repo on the Hub, with the card (sae/dataset_card.md) and the list of runs
# (sae/runs.json, which must name the run) at the top. Refuses to upload anything if a
# Parquet file in the folder is missing or incomplete (a publish cut short), or the model is
# missing. Needs `hf auth login` first.
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
# Every table the Space reads, whole (a footer that parses, its row groups' rows adding up),
# and the model, before anything is uploaded
${PYTHON:-python} - "$dir" <<'PY'
import sys
from pathlib import Path

import pyarrow.parquet as pq

folder = Path(sys.argv[1])
tables = ["features", "items", "postings", "names", "classes", "members", "id_properties"]
bad = []
for name in tables:
    path = folder / f"{name}.parquet"
    try:
        meta = pq.ParquetFile(path).metadata
        rows = sum(meta.row_group(i).num_rows for i in range(meta.num_row_groups))
        if rows != meta.num_rows or meta.num_rows == 0:
            bad.append(f"{path}: {meta.num_rows:,} rows, {rows:,} in its row groups")
        else:
            print(f"{path}: {meta.num_rows:,} rows")
    except Exception as e:
        bad.append(f"{path}: {e}")
for name in ["ae.pt", "config.json"]:
    if not (folder / "model" / name).is_file():
        bad.append(f"{folder / 'model' / name}: missing")
if bad:
    print("Not uploading; publish again first:", *bad, sep="\n  ", file=sys.stderr)
    sys.exit(1)
PY
hf upload "$repo" "$dir" "$run" --repo-type dataset --commit-message "Upload run $run"
top=$(mktemp -d)
cp sae/dataset_card.md "$top/README.md"
cp sae/runs.json "$top/runs.json"
hf upload "$repo" "$top" . --repo-type dataset --commit-message "Update the card and runs.json"
rm -r "$top"
echo "https://huggingface.co/datasets/$repo/tree/main/$run"
