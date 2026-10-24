"""On the claims key: today's bucket check (Polars row hashes of the sources, stably
sorted, against the written file's, read back) against an exact one without hashing (the
permutation and ids checked, the written file read back and compared with the sorted
table). Each timed, wall and CPU."""
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from wikidata import sort_by_id as s
from wikidata.compact import _write_file_parallel

src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(exist_ok=True)
dst = out / "sorted.parquet"


def timed(label, f):
    c, w = time.process_time(), time.perf_counter()
    r = f()
    print(f"{label:44s} wall {time.perf_counter() - w:6.2f}s  cpu {time.process_time() - c:6.2f}s", flush=True)
    return r


schema = pq.read_schema(src).remove_metadata()
rg = s._row_group_rows([src])
for rnd in range(2):
    print(f"round {rnd}")
    t = s._read_file(src)
    idx = pc.sort_indices(t, [("id", "ascending")])
    st = t.take(idx)
    _write_file_parallel(dst, schema, st.to_batches(max_chunksize=rg), rg, s._sorting(schema, "id"))

    def today():
        h = s._row_hashes(pl.scan_parquet(src), "id")
        want = h.sort("id", maintain_order=True, nulls_last=True)["h0"]
        got = pl.scan_parquet(dst).select(pl.col("id"), pl.struct(pl.all()).hash(seed=0).alias("h0")).collect(engine="streaming")
        assert got["h0"].equals(want)

    def exact():
        order = idx.to_numpy()
        assert (np.bincount(order, minlength=len(order)) == 1).all()  # a permutation
        ids = st.column("id")
        assert ids.equals(t.column("id").take(idx))  # the ids taken
        assert pc.all(pc.greater_equal(ids.combine_chunks()[1:], ids.combine_chunks()[:-1])).as_py()  # ascending
        back = s._read_file(dst)
        assert back.equals(st)  # the file holds exactly the sorted rows

    timed("today: hashes, sources and the file read back", today)
    timed("exact: permutation, ids, file read back, equals", exact)
    del t, st, idx
