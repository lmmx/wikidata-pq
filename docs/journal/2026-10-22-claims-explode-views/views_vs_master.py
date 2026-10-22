"""Partition files from claims_base with this tree's explode against master's, on real
claims files. Usage: views_vs_master.py <new src> <master src> <claims.parquet>..."""
import os, sys, importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
os.environ.setdefault("WIKIDATA_RELEASE", "20261012")
new, old = sys.argv[1], sys.argv[2]
sys.path.insert(0, new)
sys.path.insert(0, str(Path(new).parent / "scripts"))
import polars as pl
import test_claims_explode as T
from wikidata.partitioning import claims as C
spec = importlib.util.spec_from_file_location("old_claims", Path(old) / "wikidata/partitioning/claims.py")
O = importlib.util.module_from_spec(spec); spec.loader.exec_module(O)
for f in [Path(x).resolve() for x in sys.argv[3:]]:
    f = Path(f).resolve()
    with TemporaryDirectory() as tmp:
        os.chdir(tmp)
        lf = pl.scan_parquet(f).drop_nulls()
        a = T.partition_bytes(C.claims_base(lf), f.name)
        b = T.partition_bytes(O.claims_base(lf), f.name)
        print(f.name, "files", len(a), "identical:", a == b, flush=True)
        os.chdir("/")
