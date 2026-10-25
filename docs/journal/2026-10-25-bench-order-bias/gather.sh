#!/usr/bin/env bash
# A benchmark run's six round jobs (each artifact's results.json) as one JSON list, the
# form of bench-results' runs/<tier>/<sha>.json, for order.py: OUT/<run id>.json.
# Downloads one artifact at a time and keeps only its results.json.
# Usage: gather.sh OUT RUN_ID...
set -euo pipefail
out=$1; shift
mkdir -p "$out"
for run in "$@"; do
  [ -s "$out/$run.json" ] && continue
  tmp=$(mktemp -d)
  names=$(gh api "repos/lmmx/wikidata-pq/actions/runs/$run/artifacts" --jq '.artifacts[] | select(.expired | not) | .name' | grep '^round-' || true)
  for n in $names; do
    gh run download "$run" -R lmmx/wikidata-pq -n "$n" -D "$tmp/$n" >/dev/null 2>&1 || continue
  done
  python3 -I -c 'import json, sys
from pathlib import Path
jobs = [json.loads(p.read_text()) for p in sorted(Path(sys.argv[1]).glob("round-*/results.json"))]
Path(sys.argv[2]).write_text(json.dumps(jobs))
print(sys.argv[2], len(jobs))' "$tmp" "$out/$run.json"
  rm -rf "$tmp"
done
