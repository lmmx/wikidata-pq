"""Every audit sidecar of two kept runs, compared column by column (the run directory
stripped from `path`): per table, how many sidecars differ in each column."""
import sys
from collections import Counter
from pathlib import Path
import polars as pl
a, b = Path(sys.argv[1]) / "audit", Path(sys.argv[2]) / "audit"
strip = lambda d: d.with_columns(pl.col("path").str.replace(r"^.*/results/", "results/"))
diff, n = Counter(), Counter()
for p in sorted(a.rglob("*.parquet")):
    x, y = strip(pl.read_parquet(p)), strip(pl.read_parquet(b / p.relative_to(a)))
    t = p.relative_to(a).parts[0]; n[t] += 1
    if x.columns != y.columns or x.height != y.height:
        diff[(t, "shape")] += 1; continue
    for c in x.columns:
        if not x[c].equals(y[c]):
            diff[(t, c)] += 1
print("sidecars per table:", dict(n))
print("differing (table, column): count:", dict(diff) or "none")
