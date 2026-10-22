"""claims_base with explode variants, each in its own fresh process: `cur` (as is),
`chunk` (no combine_chunks copy of a single chunk), `views` (Polars' own Arrow types, so no
conversion either way, others gathered in Polars). Per variant: claims_base(...).collect()
3 times in one process (the first cold, as a fresh worker; then warm), wall, CPU, RSS peak.
Usage: explode_vars.py <src> <claims.parquet> <variant> [check]"""
import resource, sys, time
sys.path.insert(0, sys.argv[1])
import numpy as np, polars as pl, pyarrow as pa
from wikidata.partitioning import claims as C

def kb(f): return next(int(l.split()[1]) for l in open("/proc/self/status") if l.startswith(f)) // 1024
def cpu():
    r = resource.getrusage(resource.RUSAGE_SELF); return r.ru_utime + r.ru_stime

def explode_chunk(df, col):
    table = df.to_arrow()
    idx = table.schema.get_field_index(col)
    field = table.schema.field(idx)
    column = table.column(idx)
    lists = column.chunk(0) if column.num_chunks == 1 else column.combine_chunks()
    lengths = lists.value_lengths().fill_null(0).to_numpy(zero_copy_only=False)
    values = lists.flatten()
    counts = np.maximum(lengths, 1)
    if (lengths > 0).all():
        exploded = values
    else:
        out_starts = np.cumsum(counts) - counts
        value_starts = np.cumsum(lengths) - lengths
        take = np.arange(counts.sum()) - np.repeat(out_starts - value_starts, counts)
        exploded = values.take(pa.array(take, mask=np.repeat(lengths == 0, counts)))
    others = table.remove_column(idx).take(np.repeat(np.arange(len(lists)), counts))
    return pl.DataFrame(others.add_column(idx, field.with_type(exploded.type), exploded))

def explode_views(df, col):
    lists = df.get_column(col).to_arrow(compat_level=pl.CompatLevel.newest())
    lengths = lists.value_lengths().fill_null(0).to_numpy(zero_copy_only=False)
    values = pl.Series(col, lists.flatten())  # null and empty lists contribute no values
    counts = np.maximum(lengths, 1)
    if not (lengths > 0).all():
        out_starts = np.cumsum(counts) - counts
        value_starts = np.cumsum(lengths) - lengths
        take = np.arange(counts.sum()) - np.repeat(out_starts - value_starts, counts)
        values = values.gather(pl.Series(take).set(pl.Series(np.repeat(lengths == 0, counts)), None))
    others = df.drop(col).gather(np.repeat(np.arange(len(lists)), counts))
    return others.insert_column(df.get_column_index(col), values)

V = {"chunk": explode_chunk, "views": explode_views}
w = sys.argv[3]
if w in V:
    C.explode = V[w]
lf = pl.scan_parquet(sys.argv[2]).drop_nulls()
for r in range(3):
    open("/proc/self/clear_refs", "w").write("5"); r0 = kb("VmRSS")
    t, c = time.perf_counter(), cpu()
    out = C.claims_base(lf).collect()
    print(f"{w:6s} {r}: {time.perf_counter()-t:6.2f}s wall {cpu()-c:6.2f}s cpu, RSS {r0} -> peak {kb('VmHWM')} -> {kb('VmRSS')} MB, {out.shape}", flush=True)
    if len(sys.argv) > 4 and r == 0:
        from importlib import reload
        orig = reload(C)
        ref = orig.claims_base(lf).collect()
        print("   equal to cur:", ref.schema == out.schema and ref.equals(out, null_equal=True), flush=True)
        C.explode = V[w]; del ref
    del out
