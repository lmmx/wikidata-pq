"""The claims key's sort on this machine, master's way (take) against #54's (runs
sliced, each row group's slices combined), each step timed: wall and CPU."""
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from wikidata import sort_by_id as s
from wikidata.compact import _write_file_parallel

src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(exist_ok=True)


def timed(label, f):
    c, w = time.process_time(), time.perf_counter()
    r = f()
    print(f"{label:38s} wall {time.perf_counter() - w:6.2f}s  cpu {time.process_time() - c:6.2f}s", flush=True)
    return r


def runs_sorted(t, col):  # #54's _sorted
    idx = pc.sort_indices(t, [(col, "ascending")])
    order = idx.to_numpy()
    starts = np.flatnonzero(np.diff(order) != 1) + 1
    print(f"  runs {len(starts) + 1} over {t.num_rows} rows")
    bounds = [0, *starts.tolist(), t.num_rows]
    runs = (t.slice(int(order[a]), b - a) for a, b in zip(bounds, bounds[1:]))
    return pa.Table.from_batches([b for run in runs for b in run.to_batches()], t.schema)


def rg_batches(t, rows):  # #54's _row_group_batches (threaded, as for a large key)
    from concurrent.futures import ThreadPoolExecutor
    parts = [t.slice(i, rows) for i in range(0, t.num_rows, rows)]
    with ThreadPoolExecutor(min(len(parts), pa.cpu_count())) as pool:
        return [b for bs in pool.map(lambda p: p.combine_chunks().to_batches(), parts) for b in bs]


schema = pq.read_schema(src).remove_metadata()
rg = s._row_group_rows([src])
sorting = s._sorting(schema, "id")
for rnd in range(2):
    print(f"round {rnd}")
    t = timed("read", lambda: s._read_file(src))
    print(f"  {t.num_rows} rows, {t.nbytes / 1e6:.0f} MB in memory, rg_rows {rg}")
    a = timed("master: _sorted (take, threaded)", lambda: s._sorted(t, "id"))
    timed("master: write", lambda: _write_file_parallel(out / "a.parquet", schema, a.to_batches(max_chunksize=rg), rg, sorting))
    b = timed("#54: _sorted (runs sliced)", lambda: runs_sorted(t, "id"))
    bb = timed("#54: _row_group_batches", lambda: rg_batches(b, rg))
    timed("#54: write", lambda: _write_file_parallel(out / "b.parquet", schema, bb, rg, sorting))
    print("  same rows in order:", a.equals(pa.Table.from_batches(bb, schema)))
    del t, a, b, bb
print("max RSS MB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)
