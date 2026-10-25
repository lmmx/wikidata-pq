#!/usr/bin/env python3
"""Test the claims partition's sort by id (partitioning.claims.by_id), which process()
applies to each chunk's statements before writing them.

On synthetic statements in the dump's order (ids of several lengths, so not in string
order, a property among items, a local inversion, each entity's statements numbered in
their order): `by_id` is a stable sort by string id (each id's rows keep their order), and
returns the table itself, no copy, when it is already sorted. And finalise's global sort,
stable by id, of chunks sorted this way is row for row its sort of the chunks as they
were: for chunks whose entities are each in one chunk (a release), and for the bench's run
on three times the rows, which repeats every chunk's ids in three groups. Given processed
claims files (one per chunk, in chunk order), it checks those too.

Usage: python scripts/test_claims_sorted.py [processed claims file ...]
"""

import sys

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from wikidata.partitioning.claims import by_id


def global_sort(chunks: list[pa.Table]) -> pa.Table:
    """finalise's sort of a key (sort_by_id._sorted): its files in order, stably by id."""
    t = pa.concat_tables(chunks)
    return t.take(pc.sort_indices(t, [("id", "ascending")]))


def check_stable(t: pa.Table, st: pa.Table, what: str) -> None:
    assert st.num_rows == t.num_rows, what
    ids = st.column("id").combine_chunks()
    n = len(ids)
    if n > 1:
        assert pc.all(pc.less_equal(ids.slice(0, n - 1), ids.slice(1))).as_py(), f"{what}: not sorted"
    # each id's rows in their order in `t`
    seen: dict[str, list[int]] = {}
    for i, s in zip(st.column("id").to_pylist(), st.column("seq").to_pylist()):
        seen.setdefault(i, []).append(s)
    want: dict[str, list[int]] = {}
    for i, s in zip(t.column("id").to_pylist(), t.column("seq").to_pylist()):
        want.setdefault(i, []).append(s)
    assert seen == want, f"{what}: an id's rows changed their order"


def chunk(entities: list[str], start: int) -> pa.Table:
    """Each entity's statements (one to four, by its number), numbered in order, with a
    nested column as the claims have."""
    ids, seq, quals = [], [], []
    for e in entities:
        for k in range(1 + int(e[1:]) % 4):
            ids.append(e)
            seq.append(start + len(seq))
            quals.append([{"p": f"P{k}", "v": [e, str(k)]}])
    return pa.table({"id": ids, "seq": seq, "qualifiers": quals})


def main() -> None:
    # The dump's order: chunk 0's ids of 1 to 5 digits, a chunk crossing Q99999 -> Q100000
    # with a property and a local inversion, and a chunk of one length already sorted
    c0 = chunk([f"Q{i}" for i in range(8, 1200, 7)], 0)
    c1 = chunk(["Q99990", "Q99995", "P31", "Q99999", "Q100000", "Q100003", "Q100001", "Q100010"], 10_000)
    c2 = chunk([f"Q{i}" for i in range(4_200_000, 4_200_400, 3)], 20_000)
    chunks = [c0, c1, c2]
    for name, c in zip(["c0", "c1", "c2"], chunks):
        check_stable(c, by_id(c), name)
    assert by_id(c2) is c2, "an already sorted table is copied"
    assert by_id(c0) is not c0
    # Multi-chunk and sliced input
    multi = pa.concat_tables([c0.slice(0, 100), c0.slice(100)])
    assert by_id(multi).equals(by_id(c0)), "a multi-chunk table sorts otherwise"
    sliced = c0.slice(37, 200)
    check_stable(sliced, by_id(sliced), "sliced")
    assert by_id(c0.slice(0, 0)).num_rows == 0 and by_id(c0.slice(0, 1)).num_rows == 1

    # finalise's result: the same rows in the same order
    assert global_sort([by_id(c) for c in chunks]).equals(global_sort(chunks)), "published rows differ"
    # The bench's run on three times the rows: every chunk's ids again in groups 2 and 3
    x3 = chunks * 3
    assert global_sort([by_id(c) for c in x3]).equals(global_sort(x3)), "published rows differ (x3)"

    # Processed claims files, in chunk order
    files = sys.argv[1:]
    if files:
        tables = [pq.read_table(f) for f in files]
        for f, t in zip(files, tables):
            st = by_id(t)
            ids = st.column("id").combine_chunks()
            assert pc.all(pc.less_equal(ids.slice(0, len(ids) - 1), ids.slice(1))).as_py(), f
            print(f"{f}: {t.num_rows:,} rows, {'already sorted' if st is t else 'sorted'}")
        assert global_sort([by_id(t) for t in tables]).equals(global_sort(tables)), "published rows differ (files)"
    print("ok")


if __name__ == "__main__":
    main()
