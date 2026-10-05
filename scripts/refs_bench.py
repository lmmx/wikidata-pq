#!/usr/bin/env python3
"""Ways to take claims_labels' refs from one claims file, each in its own process, with
its time, its peak memory, and whether its refs are the same as the first's
(docs/journal/2026-10-05-sort-speed.md):

1. every column, Polars streaming engine (file_refs before 2026-10-05)
2. the needed leaves, pyarrow `ParquetFile.read`, snaks exploded by Polars (file_refs now),
   with its read and its explode timed apart
3. the needed leaves, `pl.read_parquet(columns=..., use_pyarrow=True)` (pyarrow's
   `read_table` with memory mapping, then `from_arrow`), snaks exploded by Polars
4. the needed leaves, pyarrow, snaks flattened by pyarrow (`list_flatten`, `struct_field`:
   the nested lists' values without copying), then Polars on the flat snaks only

Usage: python scripts/refs_bench.py CLAIMS_FILE [METHOD ...]  (default all, e.g. 2 4)
"""

import multiprocessing
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from wikidata.claims_labels import _refs, _refs_leaves, _snak_refs


def every_column(path: Path) -> tuple[pl.DataFrame, str]:
    return _refs(pl.scan_parquet(path)).collect(engine="streaming"), ""


def leaves_pyarrow(path: Path) -> tuple[pl.DataFrame, str]:
    t0 = time.time()
    f = pq.ParquetFile(path)
    t = f.read(columns=_refs_leaves(f))
    t1 = time.time()
    df = _refs(pl.from_arrow(t).lazy()).collect()
    return df, f"read {t1 - t0:.0f} s, explode and unique {time.time() - t1:.0f} s"


def leaves_polars_use_pyarrow(path: Path) -> tuple[pl.DataFrame, str]:
    columns = _refs_leaves(pq.ParquetFile(path))
    # read_parquet passes its own `columns` to read_table: the leaf paths go there, not in
    # pyarrow_options (which would give read_table `columns` twice)
    df = pl.read_parquet(path, columns=columns, use_pyarrow=True)
    return _refs(df.lazy()).collect(), ""


def _snak_table(snaks: pa.ChunkedArray) -> pa.Table:
    """A flat array of snaks as the (property, datavalue) table _snak_refs takes."""
    return pa.table(
        {
            "property": pc.struct_field(snaks, "property"),
            "datavalue": pc.struct_field(snaks, "datavalue"),
        }
    )


def leaves_flattened(path: Path) -> tuple[pl.DataFrame, str]:
    t0 = time.time()
    f = pq.ParquetFile(path)
    t = f.read(columns=_refs_leaves(f))
    t1 = time.time()
    # qualifiers: [{key, value: [snak]}]; references: [{snaks: [{key, value: [snak]}]}]
    qualifiers = pc.list_flatten(pc.struct_field(pc.list_flatten(t["qualifiers"]), "value"))
    groups = pc.list_flatten(pc.struct_field(pc.list_flatten(t["references"]), "snaks"))
    references = pc.list_flatten(pc.struct_field(groups, "value"))
    parts = [t.select(["property", "datavalue"]), _snak_table(qualifiers), _snak_table(references)]
    t2 = time.time()
    df = pl.concat([_snak_refs(pl.from_arrow(p).lazy()) for p in parts]).unique().collect()
    times = f"read {t1 - t0:.0f} s, flatten {t2 - t1:.0f} s, refs and unique {time.time() - t2:.0f} s"
    return df, times


METHODS = {
    "1": ("every column, Polars", every_column),
    "2": ("needed leaves, pyarrow, Polars explode (file_refs)", leaves_pyarrow),
    "3": ("needed leaves, read_parquet(use_pyarrow=True), Polars explode", leaves_polars_use_pyarrow),
    "4": ("needed leaves, pyarrow, pyarrow flatten", leaves_flattened),
}


def run(key: str, path: Path):
    t0 = time.time()
    df, detail = METHODS[key][1](path)
    secs = time.time() - t0
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2  # KiB to GiB
    return secs, peak, detail, df.sort(df.columns)


if __name__ == "__main__":
    path = Path(sys.argv[1])
    keys = sys.argv[2:] or list(METHODS)
    print(f"{path.name}: {path.stat().st_size / 1e6:.0f} MB; the needed leaves:")
    for p in _refs_leaves(pq.ParquetFile(path)):
        print(f"  {p}")
    first = None
    for key in keys:
        name = METHODS[key][0]
        with ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn")) as pool:
            secs, peak, detail, df = pool.submit(run, key, path).result()
        same = "" if first is None else ("; same refs" if df.equals(first) else "; REFS DIFFER")
        first = df if first is None else first
        detail = f" ({detail})" if detail else ""
        print(f"{key}. {name}: {secs:.0f} s{detail}, peak {peak:.1f} GiB, {df.height:,} refs{same}", flush=True)
