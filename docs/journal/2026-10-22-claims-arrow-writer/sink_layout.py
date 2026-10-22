"""The claims partition sink on the same rows in different in-memory layouts: as claims_base
leaves them, rechunked, and as read back from the written file. Wall, CPU, and whether the
written files are byte-identical. Usage: sink_layout.py <src> <claims.parquet>"""
import os, resource, shutil, sys, time, hashlib
from pathlib import Path
os.environ.setdefault("WIKIDATA_RELEASE", "20261012")
sys.path.insert(0, sys.argv[1])
import polars as pl
from wikidata.config import UNSPLIT_COL, UNSPLIT_KEY
from wikidata.partitioning.claims import claims_base
from wikidata.partitioning import core
def cpu():
    r = resource.getrusage(resource.RUSAGE_SELF); return r.ru_utime + r.ru_stime
src = Path(sys.argv[2]).resolve(); out = Path("/tmp/scratch/cmp/w/layout")
def sink(df):
    shutil.rmtree(out, ignore_errors=True)
    t, c = time.perf_counter(), cpu()
    df.lazy().sink_parquet(pl.PartitionBy(out, key=UNSPLIT_COL, file_path_provider=lambda a: core.custom_file_path(a, source=src), include_key=False), mkdir=True)
    f = next(out.rglob("*.parquet"))
    return time.perf_counter() - t, cpu() - c, hashlib.sha256(f.read_bytes()).hexdigest()[:12], f
df = claims_base(pl.scan_parquet(src).drop_nulls()).with_columns(pl.lit(UNSPLIT_KEY).alias(UNSPLIT_COL)).collect()
w, c, h, f = sink(df); shutil.copy(f, out.parent / "layout_ref.parquet")
back = pl.read_parquet(out.parent / "layout_ref.parquet").with_columns(pl.lit(UNSPLIT_KEY).alias(UNSPLIT_COL))
print("chunks per column:", {k: df[k].n_chunks() for k in df.columns if df[k].n_chunks() > 1} or "all 1")
variants = {"as claims_base leaves it": lambda: df, "rechunked": lambda: df.rechunk(), "read back from the file": lambda: back}
for r in range(2):
    for name, v in variants.items():
        x = v(); w, c, h2, _ = sink(x)
        print(f"  {name:26s} {w:5.2f}s wall {c:5.2f}s cpu  sha {h2} {'same' if h2 == h else 'DIFFERENT'}", flush=True)
