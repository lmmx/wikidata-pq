#!/usr/bin/env bash
# The items with a defining formula (P2534): laws, equations and quantities, with the
# fields that study them ("studied by", P2579), less the numbers (with a "numeric value",
# P1181), as demos/output/formulas.parquet. Extra arguments go to demos/export_bearers.py.
#
#   demos/formulas_export.sh                     # local copy in hub/, `python` on the PATH
#   demos/formulas_view.sh
set -euo pipefail

cd "$(dirname "$0")/.."
start=$SECONDS
${PYTHON:-python} demos/export_bearers.py P2534 --column P2579 --without P1181 \
  --data "${DATA:-hub}" --out demos/output/formulas.parquet "$@"
echo "formulas_export: $((SECONDS - start)) s -> demos/output/formulas.parquet"
