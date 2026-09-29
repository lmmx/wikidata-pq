"""Which casts fail, and the length of the null-typed child in each result.

Run: python matrix.py (needs only pyarrow). Prints one line per case, ok or FAIL.
"""

import pyarrow as pa
import pyarrow.compute as pc

print("pyarrow", pa.__version__)

s = pa.struct([("a", pa.int64()), ("n", pa.null())])
s_int32 = pa.struct([("a", pa.int32()), ("n", pa.null())])
s_no_null = pa.struct([("a", pa.int64()), ("n", pa.int64())])
L = pa.list_

cases = [
    # Lists, not sliced: fails when the list has fewer rows than structs
    ("list [[x]]: 1 row, 1 struct", pa.array([[{"a": 1}]], L(s)), L(s)),
    ("list [[x, x]]: 1 row, 2 structs", pa.array([[{"a": 1}, {"a": 2}]], L(s)), L(s)),
    ("list [[x], [x]]: 2 rows, 2 structs", pa.array([[{"a": 1}], [{"a": 2}]], L(s)), L(s)),
    ("list [None, [x, x]]: 2 rows, 2 structs", pa.array([None, [{"a": 1}, {"a": 2}]], L(s)), L(s)),
    ("list [[x, x], [x]]: 2 rows, 3 structs", pa.array([[{"a": 1}, {"a": 2}], [{"a": 3}]], L(s)), L(s)),
    ("large_list [[x, x]]", pa.array([[{"a": 1}, {"a": 2}]], pa.large_list(s)), pa.large_list(s)),
    ("list_view [[x, x]]", pa.array([[{"a": 1}, {"a": 2}]], pa.list_view(s)), pa.list_view(s)),
    ("list [[x, x]] with n: int64 (control)", pa.array([[{"a": 1}, {"a": 2}]], L(s_no_null)), L(s_no_null)),
    # Casts to another type
    ("list [[x, x]] -> large_list<same struct>", pa.array([[{"a": 1}, {"a": 2}]], L(s)), pa.large_list(s)),
    ("list [[x, x]] -> list<struct<a: int32, n: null>>", pa.array([[{"a": 1}, {"a": 2}]], L(s)), L(s_int32)),
    # Structs, not in a list
    ("struct, 2 rows, not sliced", pa.array([{"a": 0}, {"a": 1}], s), s),
    ("struct, 2 rows, slice(1)", pa.array([{"a": 0}, {"a": 1}], s).slice(1), s),
    ("struct, 2 rows, slice(1) -> struct<a: int32, n: null>", pa.array([{"a": 0}, {"a": 1}], s).slice(1), s_int32),
]


for label, arr, to in cases:
    arr.validate(full=True)
    out = pc.cast(arr, to)
    try:
        out.validate(full=True)
        print(f"ok    {label}")
    except pa.ArrowInvalid as e:
        print(f"FAIL  {label}: input length {len(arr)}, {e}")

# Array.cast and ChunkedArray.cast give the same invalid result, also without raising
arr = pa.array([[{"a": 1}, {"a": 2}]], L(s))
for label, out in [
    ("Array.cast", arr.cast(arr.type)),
    ("ChunkedArray.cast", pa.chunked_array([arr]).cast(arr.type)),
]:
    try:
        out.validate(full=True)
        print(f"ok    {label} of list [[x, x]]")
    except pa.ArrowInvalid as e:
        print(f"FAIL  {label} of list [[x, x]]: {e}")
