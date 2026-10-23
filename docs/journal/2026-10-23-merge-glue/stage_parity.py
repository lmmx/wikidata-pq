import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import polars as pl, pyarrow.parquet as pq
from wikidata.compact import _fingerprint
a, b = Path(sys.argv[2]), Path(sys.argv[3])
fa = sorted(p.relative_to(a) for p in a.rglob("*.parquet"))
fb = sorted(p.relative_to(b) for p in b.rglob("*.parquet"))
print("files", len(fa), len(fb), "same set", fa == fb)
bad = 0; rg = {}
for r in fa:
    x, y = a / r, b / r
    ok = _fingerprint(pl.scan_parquet(x)) == _fingerprint(pl.scan_parquet(y)) and pq.read_schema(x).equals(pq.read_schema(y), check_metadata=True)
    if not ok:
        bad += 1; print("DIFFERS", r) if bad < 5 else None
    t = r.parts[-3]
    m = pq.ParquetFile(y).metadata; n = pq.ParquetFile(x).metadata
    s = rg.setdefault(t, [0, 0, 0, 0]); s[0] += n.num_row_groups; s[1] += m.num_row_groups; s[2] += x.stat().st_size; s[3] += y.stat().st_size
print("differing", bad)
for t, (r1, r2, b1, b2) in rg.items():
    print(f"{t:13s} row groups {r1:6d} -> {r2:6d}  bytes {b1/2**20:6.1f} -> {b2/2**20:6.1f} MB")
