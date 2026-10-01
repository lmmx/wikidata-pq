#!/usr/bin/env bash
# The items with a defining formula (P2534): laws, equations and quantities, with the
# fields that study them ("studied by", P2579), less the numbers (with a "numeric value",
# P1181, or instances of "integer", Q12503, "prime number", Q49008, or a class below
# them: prime number is not below integer, its "subclass of odd number" being deprecated,
# since 2 is even), as demos/output/formulas.parquet. Extra arguments go to
# demos/export_bearers.py.
#
#   demos/formulas_export.sh                     # local copy in hub/, `python` on the PATH
#   demos/formulas_view.sh
set -euo pipefail

cd "$(dirname "$0")/.."
start=$SECONDS
${PYTHON:-python} demos/export_bearers.py P2534 --column P2579 --without P1181 --not-a Q12503 --not-a Q49008 \
  --data "${DATA:-hub}" --out demos/output/formulas.parquet "$@"
echo "formulas_export: $((SECONDS - start)) s -> demos/output/formulas.parquet"
