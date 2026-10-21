"""s90 claims: base morsel sizes into the sink; then the new frame written as one chunk and
re-chunked at those sizes, each compared byte for byte with base's partition file."""
import os, sys, shutil
os.environ["WIKIDATA_RELEASE"] = "20261012"
from pathlib import Path
import polars as pl
name = sys.argv[1]
src = next(Path(f"proc/{name}/out/claims").glob("*.parquet")).resolve()
base_file = Path(f"runs/{name}/base/r1/out/claims/all/{src.name}").resolve()
sys.path.insert(0, os.path.abspath("src_new/src"))
from wikidata.partitioning import partition_parquet, prepare_for_partition
from wikidata.config import Table, UNSPLIT_COL
sizes = [int(x) for x in sys.argv[2].split(",")]
df = prepare_for_partition(src, Table.CLAIMS).collect()
def write(frame, tag):
    d = Path("mtest") / tag; shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
    partition_parquet(by=UNSPLIT_COL, lf=frame.lazy(), source_name=src.name, dst_dir=d / "out", log_dir=d / "audit")
    out = d / "out" / "all" / src.name
    print(tag, "identical to base:", out.read_bytes() == base_file.read_bytes())
write(df, "one_chunk")
offs = [0]
for s in sizes: offs.append(offs[-1] + s)
write(pl.concat([df.slice(a, b - a) for a, b in zip(offs, offs[1:])], rechunk=False), "base_morsels")
