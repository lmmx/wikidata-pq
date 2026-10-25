"""claims_labels' refs from one claims file: the cost of finalise's re-read (`file_refs`:
the 9 leaves decoded, then the refs taken) against the refs taken from arrays already in
memory, as a split that has just built the claims would have them.

Usage: python -I refs_cost.py CLAIMS_FILE SRC_DIR  (SRC_DIR: the repo's src/)
"""

import sys
import time
from pathlib import Path

import polars as pl
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, sys.argv[2])
from wikidata.claims_labels import _refs_leaves, _snak_refs, _snak_table, file_refs  # noqa: E402

path = Path(sys.argv[1])
f = pq.ParquetFile(path)
print(f"claims: {f.metadata.num_rows:,} rows, {path.stat().st_size:,} bytes")
t = time.process_time()
refs = file_refs(path)
print(f"file_refs (finalise today): {time.process_time() - t:.2f} CPU-s, {refs.height:,} refs")
for _ in range(3):
    t0 = time.process_time()
    leaves = f.read(columns=_refs_leaves(f))
    t1 = time.process_time()
    q = pc.list_flatten(pc.struct_field(pc.list_flatten(leaves["qualifiers"]), "value"))
    g = pc.list_flatten(pc.struct_field(pc.list_flatten(leaves["references"]), "snaks"))
    r = pc.list_flatten(pc.struct_field(g, "value"))
    parts = [leaves.select(["property", "datavalue"]), _snak_table(q), _snak_table(r)]
    pl.concat([_snak_refs(pl.from_arrow(p).lazy()) for p in parts]).unique().collect()
    t2 = time.process_time()
    print(f"leaves decoded: {t1 - t0:.2f} CPU-s; refs from the arrays: {t2 - t1:.2f} CPU-s")
out = path.with_name("refs.parquet")
refs.write_parquet(out)
print(f"refs as Parquet: {out.stat().st_size:,} bytes ({out.stat().st_size / path.stat().st_size:.1%})")
