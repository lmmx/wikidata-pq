"""Split the claims partition's CPU on processed claims files: Polars read (scan,
drop_nulls, collect), claims_base (explodes, unnests), the Polars-to-Arrow export per
row group slice, pyarrow's write; and pyarrow's own read of the same file."""
import os, resource, sys, time
from pathlib import Path
os.environ["WIKIDATA_RELEASE"] = "20261012"
sys.path.insert(0, sys.argv[1])
import polars as pl, pyarrow.parquet as pq
from wikidata.partitioning.claims import claims_base
from wikidata.partitioning.core import ARROW_ROW_GROUP_ROWS

def cpu():
    r = resource.getrusage(resource.RUSAGE_SELF); return r.ru_utime + r.ru_stime
tot = {}
def t(name, f):
    w, c = time.perf_counter(), cpu(); v = f(); w, c = time.perf_counter() - w, cpu() - c
    a = tot.setdefault(name, [0, 0]); a[0] += w; a[1] += c; return v
def split(p: Path) -> None:
    df = t("polars read", lambda: pl.scan_parquet(p).drop_nulls().collect())
    out = t("claims_base", lambda: claims_base(df.lazy()).collect())
    parts = t("export to arrow", lambda: [s.rechunk().to_arrow() for s in out.iter_slices(ARROW_ROW_GROUP_ROWS)])

    def write():
        with pq.ParquetWriter("/dev/null", out.head(0).to_arrow().schema, compression="zstd") as w:
            for a in parts:
                w.write_table(a)

    t("pyarrow write", write)
    t("pyarrow read", lambda: pq.read_table(p))


for p in map(Path, sys.argv[2:]):
    split(p)
for k, (w, c) in tot.items():
    print(f"{k:16s} {w:6.2f}s wall {c:6.2f}s cpu")
