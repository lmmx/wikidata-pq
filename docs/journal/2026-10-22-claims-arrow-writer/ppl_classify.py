"""Collapsed perf samples of pp_loop.py + its MARK lines -> busy samples by phase (process /
partition) and component. Usage: ppl_classify.py <collapsed> <marks>"""
import collections, re, sys
marks = [tuple(map(float, l.split()[2:])) for l in open(sys.argv[2]) if l.startswith("MARK")]
RULES = [("genson", ("genson_core", "polars_genson")),
         ("parquet write", ("polars_parquet::arrow::write", "polars_parquet::parquet::write", "ZSTD_compress", "polars_io::parquet::write", "polars_stream::nodes::io_sinks")),
         ("parquet read", ("polars_parquet::arrow::read", "polars_parquet::parquet::read", "ZSTD_decompress", "polars_io::parquet::read", "polars_stream::nodes::io_sources")),
         ("pyarrow", ("libarrow", "arrow::")),
         ("polars other", ("polars",)),
         ("python", ("_PyEval", "PyObject", "_Py")),]
IDLE = ("futex", "wait_until_cold", "sleep", "park", "epoll", "nanosleep", "sched_yield", "do_wait", "poll", "schedule")
def comp(frames):
    j = ";".join(frames)
    for name, keys in RULES:
        if any(k in j for k in keys): return name
    return "other"
c = collections.Counter(); per = collections.Counter(); idle = 0
for line in open(sys.argv[1]):
    t, tid, stack = line.rstrip("\n").split(" ", 2); t = float(t)
    frames = stack.split(";")
    if any(k in frames[-1] for k in IDLE): idle += 1; continue
    ph = next((("process" if t < m[1] else "partition") for m in marks if m[0] <= t < m[2]), "between")
    k = comp(frames); c[(ph, k)] += 1; per[ph] += 1
tot = sum(per.values())
for ph in ("process", "partition", "between"):
    print(f"{ph}: {per[ph]} samples ({per[ph]/tot:.0%})")
    for (p, k), n in sorted(c.items(), key=lambda kv: -kv[1]):
        if p == ph: print(f"   {k:14s} {n:6d}  {n/tot:5.1%} of all")
