"""CPU of the derived leaves: the split's write (reduce + write_table vs write_table) and
finalise's read (read + restore vs read of the full file), best of N, on the sample's
claims partition files."""
import glob, os, sys, time, json
from pathlib import Path
os.environ.setdefault("WIKIDATA_RELEASE", "20261012")
import pyarrow as pa, pyarrow.parquet as pq, pq_glue
from wikidata import claims_derived as cd, release_schema
from wikidata.partitioning.core import write_table
N = int(os.environ.get("ROUNDS", 3))
out = Path(sys.argv[2]); out.mkdir(exist_ok=True)
full = release_schema.CLAIMS.file_schema(0)
T = time.process_time
def best(f):
    b = 1e9
    for _ in range(N):
        c = T(); f(); b = min(b, T() - c)
    return b
tot = {}
for f in sorted(glob.glob(sys.argv[1])):
    t = pq.read_table(f)
    a, b = out / "full.parquet", out / "red.parquet"
    r = {}
    r["write_full"] = best(lambda: write_table(t, a))
    r["write_reduced"] = best(lambda: write_table(cd.reduce(t), b))
    r["reduce"] = best(lambda: cd.reduce(t))
    r["read_full_pyarrow"] = best(lambda: pq.read_table(a))
    r["read_reduced_pyarrow+restore"] = best(lambda: cd.restore(pq.read_table(b), full))
    meta = pq.read_schema(b).metadata
    r["read_full_pqglue"] = best(lambda: pq_glue.read_parquet(a))
    r["read_reduced_pqglue+restore"] = best(lambda: cd.restore(pq_glue.read_parquet(b).replace_schema_metadata(meta), full))
    red = pq.read_table(b)
    r["restore"] = best(lambda: cd.restore(red, full))
    r["bytes_full"], r["bytes_reduced"] = a.stat().st_size, b.stat().st_size
    r["rows"] = t.num_rows
    for k, v in r.items():
        tot[k] = tot.get(k, 0) + v
    print(Path(f).parts[-3], Path(f).name, {k: round(v, 3) for k, v in r.items()}, flush=True)
print("total", json.dumps({k: round(v, 3) for k, v in tot.items()}))
