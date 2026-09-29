"""Casting a struct with a null-typed field, nested in a list or sliced, to its own type
returns an invalid array, without raising.

Run: python repro.py (needs only pyarrow). Exits non-zero while the bug is present.
"""

import sys

import pyarrow as pa
import pyarrow.compute as pc

print("pyarrow", pa.__version__)

s = pa.struct([("a", pa.int64()), ("n", pa.null())])
failed = False

# 1. A list of 1 row holding 2 structs
arr = pa.array([[{"a": 1}, {"a": 2}]], type=pa.list_(s))
arr.validate(full=True)  # the input is valid
out = pc.cast(arr, arr.type)  # does not raise
print(
    f"list: {len(arr)} row, struct length {len(out.values)},"
    f" child a length {len(out.values.field(0))}, child n length {len(out.values.field(1))}"
)
try:
    out.validate(full=True)
except pa.ArrowInvalid as e:
    print("  BUG: the cast result is invalid:", e)
    failed = True

# 2. A struct array sliced to its second row
arr = pa.array([{"a": 0}, {"a": 1}], type=s).slice(1)
arr.validate(full=True)
out = pc.cast(arr, s)
print(f"sliced struct: offset {out.offset}, length {len(out)}")
try:
    out.validate(full=True)
except pa.ArrowInvalid as e:
    print("  BUG: the cast result is invalid:", e)
    failed = True

sys.exit(1 if failed else 0)
