"""The claims partition written by Polars' sink (as partition_parquet does) against pyarrow's
write_table on the same frame (exported with to_arrow(): Arrow's large types, the Arrow schema
Polars stores), at zstd levels: wall and CPU including the export, file size, row groups,
and whether the files read back equal (Polars and pyarrow), with the same schemas.
Usage: pa_write.py <src> <claims.parquet>"""
import os, resource, shutil, sys, time
from pathlib import Path
os.environ.setdefault("WIKIDATA_RELEASE", "20261012")
sys.path.insert(0, sys.argv[1])
import polars as pl, pyarrow.parquet as pq
from wikidata.config import UNSPLIT_COL, Table
from wikidata.partitioning import prepare_for_partition, partition_parquet
def cpu():
    r = resource.getrusage(resource.RUSAGE_SELF); return r.ru_utime + r.ru_stime
src = Path(sys.argv[2]).resolve(); W = Path("/tmp/scratch/cmp/w/paw")
shutil.rmtree(W, ignore_errors=True); W.mkdir(parents=True)
lf = prepare_for_partition(src, Table.CLAIMS)
df = lf.collect()
t, c = time.perf_counter(), cpu()
partition_parquet(by=UNSPLIT_COL, lf=df.lazy(), source_name="chunk_0.parquet", dst_dir=W / "polars", log_dir=W / "audit")
pf = next((W / "polars").rglob("*.parquet")); print(f"  polars sink                  {time.perf_counter()-t:5.2f}s wall {cpu()-c:5.2f}s cpu  {pf.stat().st_size/2**20:6.2f} MB")
results = []
for level in (1, 3, 1, 3):
    f = W / f"pa{level}.parquet"
    t, c = time.perf_counter(), cpu()
    tb = df.drop(UNSPLIT_COL).to_arrow()
    w0, c0 = time.perf_counter() - t, cpu() - c
    pq.write_table(tb, f, compression="zstd", compression_level=level, row_group_size=122_880)
    w, cc = time.perf_counter() - t, cpu() - c
    del tb
    m = pq.read_metadata(f)
    print(f"  pyarrow zstd {level}               {w:5.2f}s wall {cc:5.2f}s cpu (export {w0:.2f}s, {c0:.2f} cpu)  {f.stat().st_size/2**20:6.2f} MB  rgs {[m.row_group(i).num_rows for i in range(m.num_row_groups)]}", flush=True)
    results.append(f)
ref_pl = pl.read_parquet(pf); ref_pa = pq.read_table(pf)
for f in results[:2]:
    got_pl = pl.read_parquet(f); got_pa = pq.read_table(f)
    print(f"  {f.name}: polars read equal {got_pl.schema == ref_pl.schema and got_pl.equals(ref_pl, null_equal=True)}, pyarrow read equal {got_pa.schema.equals(ref_pa.schema, check_metadata=False) and got_pa.equals(ref_pa)}", flush=True)
