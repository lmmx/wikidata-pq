"""Two Parquet files (or sets of files, read in order) hold the same rows in the same order
and the same schema: equal Arrow schemas with metadata, equal Polars schemas, and equal
fingerprints (compact._fingerprint: row count and two sums of position-seeded row hashes,
streamed). Usage: pq_equal.py <src> <a> <b>   (a, b: a file, or a dir of *.parquet)"""
import sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import polars as pl, pyarrow.parquet as pq
from wikidata.compact import _fingerprint

def files(p):
    p = Path(p)
    return sorted(p.glob("*.parquet"), key=lambda f: [int(x) if x.isdigit() else x for x in __import__("re").split(r"(\d+)", f.name)]) if p.is_dir() else [p]

def same(a, b):
    fa, fb = files(a), files(b)
    sa, sb = pq.read_schema(fa[0]), pq.read_schema(fb[0])
    if not sa.equals(sb, check_metadata=True):
        return "arrow schemas differ"
    la, lb = pl.scan_parquet(fa), pl.scan_parquet(fb)
    if la.collect_schema() != lb.collect_schema():
        return "polars schemas differ"
    x, y = _fingerprint(la), _fingerprint(lb)
    return True if x == y else f"fingerprints differ {x} {y}"

if __name__ == "__main__":
    t = time.perf_counter()
    print(same(sys.argv[2], sys.argv[3]), f"({time.perf_counter() - t:.1f}s)")
