"""New claims_base on a claims file, step by step, with RSS and peak after each."""
import os, sys, time
os.environ["WIKIDATA_RELEASE"] = "20261012"
sys.path.insert(0, sys.argv[2])
import polars as pl
from wikidata.partitioning.claims import explode, STATEMENT_RENAMES
def mem():
    d = {l.split(":")[0]: int(l.split()[1]) // 1024 for l in open("/proc/self/status") if l.startswith(("VmRSS", "VmHWM"))}
    return f"rss {d['VmRSS']:,} MB, peak {d['VmHWM']:,} MB"
t = time.perf_counter()
def step(name): print(f"{time.perf_counter() - t:6.1f} s  {name:28s} {mem()}", flush=True)
df = pl.scan_parquet(sys.argv[1]).drop_nulls().collect(); step(f"read {df.height} rows, {df.estimated_size()/1e6:.0f} MB")
a = explode(df, "claims"); step(f"explode claims {a.height}")
b = a.select("id", pl.col("claims").struct.unnest()).drop("key"); step("unnest, drop key")
c = explode(b, "value"); step(f"explode value {c.height}, {c.estimated_size()/1e6:.0f} MB")
d = c.lazy().select("id", pl.col("value").name.map_fields(lambda f: STATEMENT_RENAMES.get(f, f))).unnest("value").unnest("mainsnak").collect(); step(f"rename, unnest {d.width} cols")
mode = sys.argv[3] if len(sys.argv) > 3 else "none"
import shutil; shutil.rmtree("sinktest", ignore_errors=True)
from wikidata.config import UNSPLIT_COL, UNSPLIT_KEY
if mode == "partition":
    d.lazy().with_columns(pl.lit(UNSPLIT_KEY).alias(UNSPLIT_COL)).sink_parquet(pl.PartitionBy("sinktest", key=UNSPLIT_COL, include_key=False), mkdir=True)
elif mode == "write":
    d.write_parquet("sinktest.parquet")
elif mode == "sink":
    d.lazy().sink_parquet("sinktest.parquet")
step(f"{mode}")
