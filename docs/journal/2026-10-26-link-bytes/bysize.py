"""Level gains by page size: the pages of a table's files grouped by their uncompressed
size, each group's bytes at levels 1, 3, 6 and 9 (does a release's larger pages change
the gain?).

Usage: bysize.py ROOT TABLE"""
import sys
from pathlib import Path
import pyarrow as pa
sys.path.insert(0, str(Path(__file__).parent))
from pages import pages
root, table = Path(sys.argv[1]), sys.argv[2]
dec = pa.Codec("zstd"); c = {lv: pa.Codec("zstd", compression_level=lv) for lv in (1, 3, 6, 9)}
b = {}
for f in root.rglob("*.parquet"):
    if f"/{table}/" not in str(f):
        continue
    for body, lv, unc in pages(f):
        raw = dec.decompress(body[lv:], unc - lv, asbytes=True)
        k = "<16K" if unc < 16384 else "16K-128K" if unc < 131072 else ">=128K"
        a = b.setdefault(k, {"n": 0, "unc": 0, **{l: 0 for l in c}})
        a["n"] += 1; a["unc"] += unc
        for l, cc in c.items():
            a[l] += len(cc.compress(raw, asbytes=True))
for k, a in sorted(b.items()):
    print(table, k, a["n"], "pages", round(a["unc"] / 1e6, 2), "MB unc", {l: round(a[l] / 1e6, 3) for l in c}, "9/3", round(a[9] / a[3], 3), "1/3", round(a[1] / a[3], 3))
