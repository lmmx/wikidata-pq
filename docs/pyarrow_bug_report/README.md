# pyarrow: casting a struct with a null-typed field to its own type gives an invalid array

Take a struct with a field of type `null`, nested in a list holding more structs than the list has
rows, or sliced. Cast it to its own type, and the result is invalid. The cast does not raise: the
result's null-typed child comes back with the length of the array passed to `cast`, not the
struct's own length. It fails `validate()`, and `to_pylist()` on it can raise.

- **Silent:** `Array.cast`, `ChunkedArray.cast` and `pyarrow.compute.cast` return the invalid
  array without raising.
- **Raises:** `RecordBatch.cast` raises `ArrowInvalid`, because it validates its result.

Reproduced on pyarrow **25.0.1**, the latest release on PyPI on 2026-09-29, on Linux x86_64 with
Python 3.11. A search of apache/arrow issues on 2026-09-29 found no report of it. The nearest,
[GH-50515](https://github.com/apache/arrow/pull/50546) ("Respect parent validity bitmap when
casting nested structs with non-nullable fields"), is a different struct-cast bug.

## Minimal repro

[`repro.py`](repro.py), which needs only pyarrow:

```python
import pyarrow as pa

s = pa.struct([("a", pa.int64()), ("n", pa.null())])
arr = pa.array([[{"a": 1}, {"a": 2}]], type=pa.list_(s))  # 1 row, 2 structs
arr.validate(full=True)  # the input is valid

out = arr.cast(arr.type)  # does not raise
len(out.values)           # 2
len(out.values.field(0))  # 2, the int64 child
len(out.values.field(1))  # 1, the null child: the list's length, not the struct's
out.validate(full=True)
# pyarrow.lib.ArrowInvalid: List child array invalid: Invalid: Struct child array #1 has length
# smaller than expected for struct array (1 < 2)
```

A sliced struct array, not in a list, fails the same way:

```python
arr = pa.array([{"a": 0}, {"a": 1}], type=s).slice(1)
pa.compute.cast(arr, s).validate(full=True)
# pyarrow.lib.ArrowInvalid: Struct child array #1 has length smaller than expected for struct array (1 < 2)
```

Output of `python repro.py` (exits 1 while the bug is present):

```
pyarrow 25.0.1
list: 1 row, struct length 2, child a length 2, child n length 1
  BUG: the cast result is invalid: List child array invalid: Invalid: Struct child array #1 has length smaller than expected for struct array (1 < 2)
sliced struct: offset 1, length 1
  BUG: the cast result is invalid: Struct child array #1 has length smaller than expected for struct array (1 < 2)
```

## Which casts fail

[`matrix.py`](matrix.py) validates each cast's result. Output on 25.0.1:

```
ok    list [[x]]: 1 row, 1 struct
FAIL  list [[x, x]]: 1 row, 2 structs: input length 1, List child array invalid: Invalid: Struct child array #1 has length smaller than expected for struct array (1 < 2)
ok    list [[x], [x]]: 2 rows, 2 structs
ok    list [None, [x, x]]: 2 rows, 2 structs
FAIL  list [[x, x], [x]]: 2 rows, 3 structs: input length 2, List child array invalid: Invalid: Struct child array #1 has length smaller than expected for struct array (2 < 3)
FAIL  large_list [[x, x]]: input length 1, List child array invalid: Invalid: Struct child array #1 has length smaller than expected for struct array (1 < 2)
FAIL  list_view [[x, x]]: input length 1, List-view child array is invalid: Invalid: Struct child array #1 has length smaller than expected for struct array (1 < 2)
ok    list [[x, x]] with n: int64 (control)
ok    list [[x, x]] -> large_list<same struct>
ok    list [[x, x]] -> list<struct<a: int32, n: null>>
ok    struct, 2 rows, not sliced
FAIL  struct, 2 rows, slice(1): input length 1, Struct child array #1 has length smaller than expected for struct array (1 < 2)
ok    struct, 2 rows, slice(1) -> struct<a: int32, n: null>
FAIL  Array.cast of list [[x, x]]: List child array invalid: Invalid: Struct child array #1 has length smaller than expected for struct array (1 < 2)
FAIL  ChunkedArray.cast of list [[x, x]]: In chunk 0: Invalid: List child array invalid: Invalid: Struct child array #1 has length smaller than expected for struct array (1 < 2)
```

- Every failure is a struct whose length differs from the length of the array passed to `cast`:
  a list with more structs than rows (`list`, `large_list` and `list_view` alike), or a sliced
  struct. The error's first number is the input's length, the second the length the struct needs.
- Lists whose row count equals their struct count cast correctly, including one with a null row.
- The same struct with the field as `int64` casts correctly.
- Casting to a different type, one that changes another field (`a` to `int32`) or the list type
  (`list` to `large_list`), gives a valid result. Only a cast to the array's own type fails.

## Through Parquet

[`repro_parquet.py`](repro_parquet.py) shows how the bug arises in ordinary use. It writes
`[None, [x, x]]` to Parquet and reads it back in 1-row batches. The second batch is `[[x, x]]`, a
list with 1 row and 2 structs. So `RecordBatch.cast` to the file's own schema fails on it, although
the batch passes `validate(full=True)`:

```
pyarrow 25.0.1
batch 0: RecordBatch.cast to the file's schema ok
batch 1: RecordBatch.cast to the file's schema FAILS: In column 0: Invalid: List child array invalid: Invalid: Struct child array #1 has length smaller than expected for struct array (1 < 2)
read_table(...).cast(schema) ok
```

Read whole, the file has 2 rows and 2 structs, and the cast succeeds.

## Where it was found

The bug surfaced while rewriting the
[wikidata-claims](https://huggingface.co/datasets/permutans/wikidata-claims) Parquet files, which
were written by Polars, with pyarrow's `ParquetWriter`. Their `references` column is
`large_list<large_list<struct<key, value: large_list<struct<property, datavalue: struct<...>, datatype>>>>>`.
The `datavalue` struct has 17 fields, and its 17th, `altitude`, has type `null` because it is null
in every row of the source.

Each batch from `ParquetFile.iter_batches(batch_size=65536)` was cast to the file's schema, with the
schema metadata removed, before being written. On `all/chunks-7263-7448.parquet` (276 MB), the
fifth batch failed:

```
pyarrow.lib.ArrowInvalid: In column 5: Invalid: List child array invalid: Invalid: List child array invalid:
Invalid: Struct child array #1 invalid: Invalid: List child array invalid: Invalid: Struct child array #1 invalid:
Invalid: Struct child array #16 has length smaller than expected for struct array (65536 < 85150)
```

Child #16 is `altitude`, and 65,536 is the batch's row count. The first four batches of the file
cast without error. Why those four pass has not been established. Narrowing it down to the minimal
repro:
- The column's rows, rebuilt with `pa.array(rows, type=...)` and written to Parquet, still failed
  when read back in 1-row batches, down to two rows: the first null, the second a reference with
  two snaks.
- Replacing the real type with `list<struct<a: int64, n: null>>` gives the repro above.

## Workaround

Don't cast when the schemas are already equal: compare them with `Schema.equals` instead. The
wikidata-pq compaction step (`src/wikidata/compact.py`, `_source_batches`) checks every input file
has the same schema and writes the batches uncast.
