"""The claims merge of a group, Polars' sink against pyarrow variants, on the snapshot's
claims partition files: wall, CPU, peak RSS, output size, row groups.
Usage: claims_merge_proto.py <snapshot> <variant> <out.parquet>"""
import sys, time
from pathlib import Path
import polars as pl, pyarrow as pa, pyarrow.parquet as pq
snap, v, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
paths = sorted((snap / "results/claims/all").glob("chunk_*.parquet"), key=lambda p: int(p.stem.split("_")[1]))
def kb(f): return next(int(l.split()[1]) for l in open("/proc/self/status") if l.startswith(f)) // 1024
RG = 122_880
w, c = time.perf_counter(), time.process_time()
if v == "polars":
    pl.scan_parquet(paths).sink_parquet(out)
else:
    schema = pq.read_schema(paths[0])
    with pq.ParquetWriter(out, schema, compression="zstd") as wr:
        if v == "pa_rg":  # each source row group written as one
            for p in paths:
                f = pq.ParquetFile(p)
                for i in range(f.metadata.num_row_groups):
                    wr.write_table(f.read_row_group(i))
        elif v in ("pa_accum", "pa_accum_ra"):  # whole source row groups gathered to >= RG rows
            def tables():
                for p in paths:
                    f = pq.ParquetFile(p)
                    for i in range(f.metadata.num_row_groups):
                        yield f.read_row_group(i)
            it = tables()
            if v == "pa_accum_ra":
                from concurrent.futures import ThreadPoolExecutor
                ex = ThreadPoolExecutor(1)
                def ahead(gen):
                    fut = ex.submit(next, gen, None)
                    while (t := fut.result()) is not None:
                        fut = ex.submit(next, gen, None)
                        yield t
                it = ahead(it)
            pend, n = [], 0
            for t in it:
                pend.append(t); n += t.num_rows
                if n >= RG:
                    t = pa.concat_tables(pend); wr.write_table(t, row_group_size=t.num_rows); pend, n = [], 0
            if pend:
                t = pa.concat_tables(pend); wr.write_table(t, row_group_size=t.num_rows)
        elif v == "pa_rebatch":  # re-cut to RG rows
            pend, n = [], 0
            for p in paths:
                f = pq.ParquetFile(p)
                for i in range(f.metadata.num_row_groups):
                    t = f.read_row_group(i); pend.append(t); n += t.num_rows
                    while n >= RG:
                        t = pa.concat_tables(pend); wr.write_table(t.slice(0, RG)); rest = t.slice(RG); pend, n = [rest], rest.num_rows
            if n: wr.write_table(pa.concat_tables(pend))
m = pq.read_metadata(out)
print(f"{v:11s} {time.perf_counter()-w:6.2f}s wall {time.process_time()-c:6.2f}s cpu peak {kb('VmHWM')} MB  {out.stat().st_size/2**20:.1f} MB  {m.num_row_groups} rgs, {m.num_rows} rows", flush=True)
