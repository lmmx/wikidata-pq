#!/usr/bin/env python3
"""Test the row fingerprints the sort's checks compare (fingerprint.py), and that the
checks still catch what they are for.

On synthetic rows with nested lists of structs, a map, nulls and empty lists (and on a
claims file, if given): the same rows give the same fingerprints whichever reader decodes
them (pyarrow, Polars, pq-glue) and however they are chunked, and a sort permutes them.
Each kind of corruption changes the fingerprint of the row it is made in and of no other:
a deep value changed, list elements swapped, a value set to null, a value moved to another
row, two fields' values swapped, an empty list made null, an element added. Then
sort_by_id's in-memory check (_check_stable_sort) passes on a correct sort and fails on
outputs corrupted in each of those ways, and on rows of one id swapped. Runs in a
temporary directory.

Usage: python scripts/test_fingerprint.py [claims file]
"""

import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import polars as pl
import pq_glue
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("WIKIDATA_RELEASE", "20261012")

from wikidata.fingerprint import frame_fingerprints, row_fingerprints  # noqa: E402
from wikidata import sort_by_id as sb  # noqa: E402

CLAIMS = Path(sys.argv[1]) if len(sys.argv) > 1 else None
FAILED = []


def check(name: str, ok: bool) -> None:
    print(("ok  " if ok else "FAIL"), name)
    if not ok:
        FAILED.append(name)


SNAK = pa.struct([("property", pa.string()), ("value", pa.string()), ("n", pa.int64())])
SCHEMA = pa.schema([
    ("id", pa.string()),
    ("rank", pa.string()),
    ("other", pa.string()),
    ("refs", pa.list_(pa.struct([("hash", pa.string()), ("snaks", pa.list_(SNAK))]))),
    ("order", pa.list_(pa.string())),
    ("labels", pa.map_(pa.string(), pa.string())),
    ("x", pa.float64()),
])


def synthetic(n: int = 3000, seed: int = 0) -> pa.Table:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        ident = f"Q{rng.integers(1, n // 3)}"  # several rows an id
        refs = None if i % 11 == 0 else [
            {"hash": None if j % 5 == 4 else f"h{i}-{j}",
             "snaks": [{"property": f"P{k}", "value": f"v{i}{j}{k}", "n": int(rng.integers(0, 9))}
                       for k in range(int(rng.integers(0, 4)))]}
            for j in range(int(rng.integers(0, 4)))
        ]
        rows.append({
            "id": ident, "rank": "normal" if i % 3 else "preferred", "other": "normal" if i % 4 else "x",
            "refs": refs, "order": [] if i % 7 == 0 else [f"P{k}" for k in range(i % 3)],
            "labels": None if i % 13 == 0 else [("en", f"l{i}"), ("fr", f"m{i}")][: i % 3],
            "x": None if i % 9 == 0 else float(i) / 7,
        })
    return pa.Table.from_pylist(rows, schema=SCHEMA)


def with_row(t: pa.Table, k: int, mutate) -> pa.Table:
    row = t.slice(k, 1).to_pylist()[0]
    mutate(row)
    new = pa.Table.from_pylist([row], schema=t.schema)
    return pa.concat_tables([t.slice(0, k), new, t.slice(k + 1)])


def first(t: pa.Table, cond) -> int:
    return next(i for i, row in enumerate(t.to_pylist()) if cond(row))


def deep_value(row):
    row["refs"][0]["snaks"][0]["value"] += "!"


def swap_elements(row):
    row["refs"][0], row["refs"][1] = row["refs"][1], row["refs"][0]


def set_null(row):
    row["refs"][0]["hash"] = None


def swap_fields(row):
    row["rank"], row["other"] = row["other"], row["rank"]


def empty_to_null(row):
    row["order"] = None


def add_element(row):
    row["order"] = row["order"] + ["P9"]


MUTATIONS = [
    ("a deep value changed (refs[0].snaks[0].value)", deep_value,
     lambda r: r["refs"] and r["refs"][0]["snaks"]),
    ("two list elements swapped (refs 0 and 1)", swap_elements,
     lambda r: r["refs"] and len(r["refs"]) >= 2 and r["refs"][0] != r["refs"][1]),
    ("a value set to null (refs[0].hash)", set_null, lambda r: r["refs"] and r["refs"][0]["hash"]),
    ("two fields' values swapped (rank, other)", swap_fields, lambda r: r["rank"] != r["other"]),
    ("an empty list made null (order)", empty_to_null, lambda r: r["order"] == []),
    ("an element added to a list (order)", add_element, lambda r: r["order"] is not None),
]


def readers_agree(path: Path) -> bool:
    a = row_fingerprints(pq.read_table(path))
    b = row_fingerprints(pq_glue.read_parquet(path))
    c = frame_fingerprints(pl.scan_parquet(path))["fp"].to_numpy()
    return np.array_equal(a, b) and np.array_equal(a, c)


def properties(t: pa.Table, name: str, tmp: Path) -> None:
    fp = row_fingerprints(t)
    path = tmp / f"{name}.parquet"
    pq.write_table(t, path, row_group_size=max(1, t.num_rows // 5))
    check(f"{name}: pyarrow, Polars and pq-glue reads give the same fingerprints", readers_agree(path))
    check(f"{name}: the same, rechunked", np.array_equal(row_fingerprints(pa.Table.from_batches(
        t.to_batches(max_chunksize=97))), fp))
    idx = pc.sort_indices(t, [("id", "ascending")])
    check(f"{name}: a sort permutes them", np.array_equal(row_fingerprints(t.take(idx)), fp[np.asarray(idx)]))
    distinct = len({tuple(map(str, r.values())) for r in t.select(["id", "rank", "refs", "order", "x"]).to_pylist()}) \
        if name == "synthetic" else None
    if distinct is not None:
        check(f"{name}: distinct rows, distinct fingerprints", len(np.unique(fp)) >= distinct)


def corruptions(t: pa.Table, name: str) -> None:
    fp = row_fingerprints(t)
    for what, mutate, where in MUTATIONS:
        try:
            k = first(t, where)
        except (StopIteration, KeyError, TypeError):
            continue
        got = row_fingerprints(with_row(t, k, mutate))
        check(f"{name}: {what}: row {k} alone differs", list(np.flatnonzero(got != fp)) == [k])
    j = np.arange(t.num_rows)
    a, b = next((a, a + 1) for a in range(t.num_rows - 1)
                if t.column("refs")[a].as_py() != t.column("refs")[a + 1].as_py())
    j[[a, b]] = [b, a]
    i = t.schema.get_field_index("refs")
    moved = t.set_column(i, "refs", t.column("refs").take(pa.array(j)))
    check(f"{name}: a value moved to another row: rows {a} and {b} alone differ",
          set(np.flatnonzero(row_fingerprints(moved) != fp)) == {a, b})


def stable_sort_check(t: pa.Table, tmp: Path) -> None:
    """_check_stable_sort on a correct sort, then on outputs corrupted after it."""
    src = tmp / "src.parquet"
    pq.write_table(t, src)
    good = t.take(pc.sort_indices(t, [("id", "ascending")]))  # stable
    schema = t.schema.remove_metadata()

    def run(out_table: pa.Table) -> str | None:
        out = tmp / "out.parquet"
        pq.write_table(out_table, out)
        try:
            sb._check_stable_sort("test", [out], [src], "id", schema)
        except RuntimeError as e:
            return str(e)
        return None

    check("the sort check passes a correct stable sort", run(good) is None)
    for what, mutate, where in MUTATIONS:
        k = first(good, where)
        err = run(with_row(good, k, mutate))
        check(f"the sort check fails on {what}", err is not None and "differ" in err)
    ids = good.column("id").to_pylist()
    k = next(i for i in range(len(ids) - 1)
             if ids[i] == ids[i + 1] and good.slice(i, 1).to_pylist() != good.slice(i + 1, 1).to_pylist())
    j = np.arange(good.num_rows)
    j[[k, k + 1]] = [k + 1, k]
    err = run(good.take(pa.array(j)))
    check("the sort check fails on two rows of one id swapped", err is not None and "differ" in err)
    err = run(good.slice(1))
    check("the sort check fails on a row dropped", err is not None)


def main() -> None:
    with TemporaryDirectory() as d:
        tmp = Path(d)
        t = synthetic()
        properties(t, "synthetic", tmp)
        corruptions(t, "synthetic")
        stable_sort_check(t, tmp)
        empty = t.slice(0, 0)
        check("no rows: no fingerprints", len(row_fingerprints(empty)) == 0)
        if CLAIMS is not None:
            c = pq_glue.read_parquet(CLAIMS).replace_schema_metadata()
            c = c.slice(0, min(c.num_rows, 200_000))
            properties(c, "claims", tmp)
    if FAILED:
        raise SystemExit(f"{len(FAILED)} failed")
    print("all passed")


if __name__ == "__main__":
    main()
