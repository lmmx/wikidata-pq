# 2026-10-24: The sort's checks on row fingerprints

## Why

On the claims key's bucket path, about 37 of finalise's roughly 80 CPU-seconds per 73 MB
of claims went to the sort's integrity checks. The sources were hashed as bucketed, the
fragments as sorted, and each sorted file read back and hashed. Each pass was a Polars
read plus Polars' struct hash, which walks the nested structs row by row, with two seeds
for the additive sums.

Before making the checks cheaper, the question was whether they are needed at all.

## What the checks are for

| failure | caught by | cheaper check enough? |
|---|---|---|
| rows lost or duplicated in bucketing, sort or pack | row counts | yes |
| wrong order, or an id's rows reordered | the order of `(id, row key)` | yes |
| values changed while counts, ids and schema stay right: a nested array out of step with its parent, a take or concat bug (the family of pyarrow's cast corrupting the claims' structs, see `compact._source_batches`) | each row's content, compared in order | no |
| a writer or reader encoding or decoding wrongly | the output read back by another decoder, its content compared | no |
| rows lost upstream, in processing or partitioning (the entities table of 2026-10-06) | no check in finalise: it compares finalise's outputs with its inputs | (partitioning now checks its own rows) |
| storage or transfer corruption | the files' sha256, already checked when the sort sources a table | yes |

Counts, schemas and sortedness alone would drop the one guarantee that matters for the
claims: that no value inside the nested statements changed while everything around it
still looked right. So the checks stay, and what changes is their cost.

## Row fingerprints

`fingerprint.py` gives each row one 64-bit value from its whole content:

- each leaf value is hashed (Polars), salted with its column path, and given a fixed
  value if null;
- each element of a list is mixed with its position in the list, then summed into its
  list, along with the list's length;
- a struct's fields are summed into the struct, and a row's columns into the row, each
  under its own salt;
- the result goes through splitmix64's finaliser at every level.

It is computed a column at a time over the Arrow buffers (numpy folding leaves up through
list offsets), and a batch at a time through Polars' streaming reader, which hands each
batch to Arrow without a copy (`CompatLevel.newest`).

Measured on the benchmark's claims key (73 MB, 1,175,112 rows, 1,200 leaves), CPU seconds
on this VM:

| | CPU s |
|---|---|
| Polars scan with two seeds of the struct hash (the checks' pass before) | 21.2 |
| Polars read | 6.0 |
| Arrow export of it (`CompatLevel.newest`, no copy) | 0.01 |
| row fingerprints | 2.65 |
| Polars read and fingerprints, a batch at a time | 9.0 |

So a check pass is about 2.4 times cheaper, and it still reads with its own decoder. The
rows sorted are read by pyarrow or pq-glue, and the checks by Polars, as before, so a
reader that decodes wrongly is still caught.

The guarantees are the ones the checks had, and slightly more:

- a row's fingerprint depends on the order of the elements in each list, which the struct
  hash's two seeds did too;
- one 64-bit fingerprint replaces two seeds in the additive sums, which together with the
  row count are 96 bits;
- the exact-order checks compare a 64-bit value per row, as they compared seed 0 before.

## Checked

`scripts/test_fingerprint.py`, on synthetic rows (nested lists of structs, a map, nulls,
empty lists) and on the claims file:

- pyarrow, Polars and pq-glue give the same fingerprints, however the rows are chunked,
  and a sort permutes them;
- each corruption changes its own row's fingerprint and no other's: a deep value changed,
  list elements swapped, a value set to null, two fields' values swapped, an empty list
  made null, an element added, and a value moved to another row (both rows change);
- `_check_stable_sort` passes a correct stable sort and fails on each of those, on two
  rows of one id swapped, and on a dropped row.

`scripts/test_sort.py` passes: the bucket path's sums and its end check, which catches a
row changed in a bucket. `scripts/test_finalise.py` passes.

Buckets recorded with the old sums (`buckets.json` without `"check": "fingerprint"`) are
bucketed again from their sources, as the two kinds of sum cannot be compared.
