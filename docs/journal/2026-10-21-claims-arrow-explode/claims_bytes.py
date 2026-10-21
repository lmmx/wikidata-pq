"""One claims file through prepare_for_partition + partition_parquet (src given), in a fresh
process; prints the sha256 of the partition file. Usage: claims_bytes.py <claims file> <src> <out dir>"""
import hashlib, os, resource, shutil, sys, time
from pathlib import Path
os.environ["WIKIDATA_RELEASE"] = "20261012"
f, src, d = Path(sys.argv[1]).resolve(), str(Path(sys.argv[2]).resolve()), Path(sys.argv[3]).resolve()
sys.path.insert(0, src)
import wikidata.partitioning.claims as c
assert c.__file__.startswith(src)
from wikidata.partitioning import partition_parquet, prepare_for_partition
from wikidata.config import Table, UNSPLIT_COL
import polars as pl
shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True); os.chdir(d)
t = time.perf_counter()
partition_parquet(by=UNSPLIT_COL, lf=prepare_for_partition(f, Table.CLAIMS), source_name=f.name, dst_dir=Path("out"), log_dir=Path("audit"))
out = Path("out/all") / f.name
print(f"threads={pl.thread_pool_size()} {Path(src).parent.name}: sha256 {hashlib.sha256(out.read_bytes()).hexdigest()[:16]}, {out.stat().st_size:,} bytes, {time.perf_counter() - t:.1f} s, peak {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024:,} MB")
