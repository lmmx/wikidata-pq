"""Every output file of two pool_bench runs (KEEPDIR=1, no CLEAN), compared: byte-identical,
or for claims partition files the same data (Polars and pyarrow read-backs equal, Arrow
schemas equal with metadata), or for audit sidecars the same rows apart from the run
directory in `path` (and `file_size` of claims files). Also the claims files' total size.
Usage: parity.py <run dir A> <run dir B>"""
import sys
from pathlib import Path
import polars as pl, pyarrow.parquet as pq
A, B = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
wa, wb = A / "releases/20261012", B / "releases/20261012"
def files(w): return {p.relative_to(w).as_posix() for p in w.rglob("*") if p.is_file() and not p.is_symlink() and "state" not in p.parts and "data" not in p.parts}
fa, fb = files(wa), files(wb)
print("files", len(fa), len(fb), "same set", fa == fb)
same = logical = bad = 0; size = [0, 0]
for rel in sorted(fa & fb):
    a, b = wa / rel, wb / rel
    if rel.startswith("results/claims/"):
        size[0] += a.stat().st_size; size[1] += b.stat().st_size
    if a.read_bytes() == b.read_bytes():
        same += 1; continue
    if rel.startswith("results/claims/"):
        ta, tb = pq.read_table(a), pq.read_table(b); pa_, pb_ = pl.read_parquet(a), pl.read_parquet(b)
        ok = ta.equals(tb) and ta.schema.equals(tb.schema, check_metadata=True) and pa_.schema == pb_.schema and pa_.equals(pb_, null_equal=True)
    elif rel.startswith("audit/"):
        da, db = pl.read_parquet(a), pl.read_parquet(b)
        norm = lambda d, w: d.with_columns(pl.col("path").str.replace(str(w), "", literal=True))
        da, db = norm(da, wa), norm(db, wb)
        if rel.startswith("audit/claims/"):
            da, db = da.drop("file_size"), db.drop("file_size")
        ok = da.equals(db)
    else:
        ok = False
    if ok: logical += 1
    else: bad += 1; print("DIFFERS", rel)
print(f"byte-identical {same}, same data {logical}, differ {bad}; claims partition files {size[0]/2**20:.1f} MB -> {size[1]/2**20:.1f} MB")
