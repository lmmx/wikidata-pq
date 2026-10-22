"""The claims partition from partition_parquet with Polars' sink and with arrow=True, on
processed claims files: wall/CPU of each, file sizes, row groups, and whether the files hold
the same data (Polars and pyarrow read-backs equal, schemas equal with metadata).
Usage: arrow_equiv.py <src> <claims.parquet>..."""
import os, resource, shutil, sys, time
from pathlib import Path
os.environ.setdefault("WIKIDATA_RELEASE", "20261012")
sys.path.insert(0, sys.argv[1])
import polars as pl, pyarrow.parquet as pq
from wikidata.config import UNSPLIT_COL, Table
from wikidata.partitioning import prepare_for_partition, partition_parquet
def cpu():
    r = resource.getrusage(resource.RUSAGE_SELF); return r.ru_utime + r.ru_stime
W = Path("/tmp/scratch/cmp/w/equiv")
for src in map(Path, sys.argv[2:]):
    shutil.rmtree(W, ignore_errors=True)
    files = {}
    for arrow in (False, True):
        t, c = time.perf_counter(), cpu()
        partition_parquet(by=UNSPLIT_COL, lf=prepare_for_partition(src.resolve(), Table.CLAIMS), source_name="chunk_0.parquet", dst_dir=W / str(arrow), log_dir=W / f"audit{arrow}", arrow=arrow)
        files[arrow] = sorted((W / str(arrow)).rglob("*.parquet"))
        print(f"{src.stem:9s} arrow={arrow!s:5s} {time.perf_counter()-t:5.2f}s wall {cpu()-c:5.2f}s cpu  files {[f.relative_to(W / str(arrow)).as_posix() for f in files[arrow]]} {sum(f.stat().st_size for f in files[arrow])/2**20:.2f} MB", flush=True)
    a, b = files[False][0], files[True][0]
    pa_a, pa_b = pq.read_table(a), pq.read_table(b)
    pl_a, pl_b = pl.read_parquet(a), pl.read_parquet(b)
    ma, mb = pq.read_metadata(a), pq.read_metadata(b)
    print(f"   polars equal {pl_a.schema == pl_b.schema and pl_a.equals(pl_b, null_equal=True)} | pyarrow table equal {pa_a.equals(pa_b)} | arrow schema equal incl. metadata {pa_a.schema.equals(pa_b.schema, check_metadata=True)} | "
          f"row groups {[ma.row_group(i).num_rows for i in range(ma.num_row_groups)]} vs {[mb.row_group(i).num_rows for i in range(mb.num_row_groups)]} | sidecars equal but path {pl.read_parquet(W / 'auditFalse' / 'chunk_0.parquet').drop('path', 'file_size').equals(pl.read_parquet(W / 'auditTrue' / 'chunk_0.parquet').drop('path', 'file_size'))}", flush=True)
    if not pa_a.schema.equals(pa_b.schema, check_metadata=True):
        print("   schema metadata:", pa_a.schema.metadata and list(pa_a.schema.metadata), pa_b.schema.metadata and list(pa_b.schema.metadata))
