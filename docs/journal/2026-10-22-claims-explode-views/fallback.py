import sys, time, resource, importlib.util
S = "/tmp/scratch"
sys.path.insert(0, S + "/wpq-views/src")
import polars as pl
from wikidata.partitioning import claims as C
spec = importlib.util.spec_from_file_location("O", S + "/wpq-master/src/wikidata/partitioning/claims.py")
O = importlib.util.module_from_spec(spec); spec.loader.exec_module(O)
mod = C if sys.argv[1] == "views" else O
lf = pl.scan_parquet("c0_empty.parquet").drop_nulls()
t = time.perf_counter(); out = mod.claims_base(lf).collect()
print(sys.argv[1], f"{time.perf_counter()-t:.2f}s", out.shape, "maxrss", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024, flush=True)
out.write_parquet(f"fb_{sys.argv[1]}.parquet")
