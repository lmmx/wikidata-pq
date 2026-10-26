"""Each candidate column's encoding alone on top of master's (statement_id DBA): the
sample's claims bytes and the read CPU (pq_glue, pyarrow ParquetFile.read, 5 rounds, min)."""
import sys, os, json, resource, glob
import pyarrow.parquet as pq, pq_glue
sys.path.insert(0, os.environ.get("SRC", "src"))
from wikidata.partitioning.core import ARROW_ROW_GROUP_ROWS, COLUMN_ENCODINGS
P = os.environ["WORK"]  # split.py's work dir (parts/{set}/claims/chunk_*.parquet)
files = sorted(glob.glob(f"{P}/parts/*/claims/chunk_*.parquet"))
tables = [pq.read_table(f) for f in files]
C = [("id","DELTA_LENGTH_BYTE_ARRAY"),("id","PLAIN"),("datavalue.id","DELTA_BYTE_ARRAY"),
     ("datavalue.datavalue__string","DELTA_LENGTH_BYTE_ARRAY"),("datavalue.datavalue__string","DELTA_BYTE_ARRAY"),
     ("datavalue.text","DELTA_LENGTH_BYTE_ARRAY"),("datavalue.numeric-id","BYTE_STREAM_SPLIT"),("datavalue.numeric-id","PLAIN"),
     ("references.list.element.hash","DELTA_LENGTH_BYTE_ARRAY"),("references.list.element.hash","PLAIN")]
def cpu(): r = resource.getrusage(resource.RUSAGE_SELF); return r.ru_utime + r.ru_stime
def run(enc, tag):
    paths, size = [], 0
    for i, t in enumerate(tables):
        p = f"{P}/pc_{tag}_{i}.parquet"; paths.append(p)
        pq_glue.write_parquet(t, p, row_group_size=ARROW_ROW_GROUP_ROWS, column_encoding=enc)
        size += os.path.getsize(p)
    r = {}
    for n, fn in (("pq_glue", pq_glue.read_parquet), ("pyarrow", lambda p: pq.ParquetFile(p).read())):
        b = 1e9
        for _ in range(5):
            c0 = cpu(); [fn(p) for p in paths]; b = min(b, cpu() - c0)
        r[n] = b
    for p in paths: os.unlink(p)
    return size, r
bs, br = run(dict(COLUMN_ENCODINGS), "base")
print("base", bs, {k: round(v, 3) for k, v in br.items()}, flush=True)
rows = []
for col, e in C:
    s, r = run(dict(COLUMN_ENCODINGS) | {col: e}, "x")
    row = {"col": col, "enc": e, "bytes": s - bs, "pct": round(100*(s-bs)/bs, 3), **{f"{k}_cpu": round(r[k]-br[k], 3) for k in r}}
    rows.append(row); print(row, flush=True)
bs2, br2 = run(dict(COLUMN_ENCODINGS), "base")
print("base again", bs2, {k: round(v, 3) for k, v in br2.items()})
json.dump({"base": [bs, br, br2], "rows": rows}, open(f"{P}/percol.json", "w"))
