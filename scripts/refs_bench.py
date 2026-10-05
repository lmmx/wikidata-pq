#!/usr/bin/env python3
"""Refs of one claims file: the pipeline's file_refs (the whole file, Polars streaming)
against only the leaves refs need, read by pyarrow. Each runs in its own process; prints
its time, its peak memory and whether the refs are the same (docs/journal/2026-10-05-sort-speed.md).

Usage: python scripts/refs_bench.py CLAIMS_FILE
"""

import multiprocessing
import re
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

from wikidata.claims_labels import _refs, file_refs

TOP = ("property", "datavalue", "qualifiers", "references")
# A snak's property, and its datavalue's id and unit, at any depth
LEAF = re.compile(r"(^|\.)(property|datavalue\.(id|unit))$")


def leaves(path: Path) -> list[str]:
    schema = pq.ParquetFile(path).schema
    paths = [schema.column(i).path for i in range(len(schema))]
    return [p for p in paths if p.split(".")[0] in TOP and LEAF.search(p)]


def needed_fields(path: Path) -> pl.DataFrame:
    t = pq.ParquetFile(path).read(columns=leaves(path))
    return _refs(pl.from_arrow(t).lazy()).collect()


METHODS = {"whole file, Polars (file_refs)": file_refs, "needed leaves, pyarrow": needed_fields}


def run(name: str, path: Path):
    t0 = time.time()
    df = METHODS[name](path)
    secs = time.time() - t0
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2  # KiB to GiB
    return secs, peak, df.sort(df.columns)


if __name__ == "__main__":
    path = Path(sys.argv[1])
    print(f"{path.name}: {path.stat().st_size / 1e6:.0f} MB; leaves read by the second method:")
    for p in leaves(path):
        print(f"  {p}")
    got = {}
    for name in METHODS:
        with ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn")) as pool:
            secs, peak, df = pool.submit(run, name, path).result()
        print(f"{name}: {secs:.0f} s, peak {peak:.1f} GiB, {df.height:,} refs", flush=True)
        got[name] = df
    a, b = got.values()
    print("same refs" if a.equals(b) else "REFS DIFFER")
