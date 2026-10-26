"""Write and read CPU of master's encoding against statement_id in DELTA_BYTE_ARRAY, on the
sample's claims (group files as the split writes them; published as sorted and written by
_write_file_parallel), interleaved A B A B over ROUNDS rounds so order and drift favour
neither. Reads: pyarrow whole, Polars whole, and the statement_id column alone (pyarrow).

Usage: cpu_ab.py WPQ_SRC PARTS_DIR OUT_DIR [ROUNDS]"""
import json, resource, sys, types
from pathlib import Path
from statistics import median
src, parts, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
rounds = int(sys.argv[4]) if len(sys.argv) > 4 else 7
sys.path.insert(0, str(src))
stub = types.ModuleType("wikidata_reshape"); stub.__getattr__ = lambda name: None
sys.modules["wikidata_reshape"] = stub
import polars as pl, pq_glue, pyarrow as pa, pyarrow.parquet as pq
from wikidata.sort_by_id import row_group_rows

def cpu():
    r = resource.getrusage(resource.RUSAGE_SELF); return r.ru_utime + r.ru_stime

files = sorted(parts.rglob("claims/*.parquet"))
group = [pq.read_table(f) for f in files]
sets = {}
for f, t in zip(files, group):
    sets.setdefault("scholar" in str(f), []).append(t)
published = []
for _, ts in sorted(sets.items()):
    t = pa.concat_tables(ts)
    idx = pl.from_arrow(t.select(["id"])).with_row_index("i").sort("id", maintain_order=True)["i"].to_arrow()
    published.append(t.take(idx).combine_chunks())

def gw(t, dst, enc):
    pq_glue.write_parquet(t, dst, engine="parallel", row_group_size=122_880, compression="zstd",
                          compression_level=1, column_encoding=enc)
def pw(t, dst, enc):
    pq_glue.write_parquet(t, dst, engine="parallel", row_group_size=row_group_rows(t), compression="zstd",
                          compression_level=3, content_defined_chunking=True, sorting_columns=[(0, False, False)],
                          column_encoding=enc)
out.mkdir(parents=True, exist_ok=True)
for kind, tables, write in (("group", group, gw), ("published", published, pw)):
    res = {a: {k: [] for k in ("write", "read_pa", "read_pl", "read_col")} for a in ("master", "delta")}
    for r in range(rounds):
        for arm in (("master", "delta") if r % 2 == 0 else ("delta", "master")):
            enc = None if arm == "master" else {"statement_id": "DELTA_BYTE_ARRAY"}
            w = rp = rl = rc = 0.0
            for i, t in enumerate(tables):
                dst = out / f"{kind}{i}.parquet"
                c = cpu(); write(t, dst, enc); w += cpu() - c
                c = cpu(); pq.read_table(dst); rp += cpu() - c
                c = cpu(); pl.read_parquet(dst); rl += cpu() - c
                c = cpu(); pq.read_table(dst, columns=["statement_id"]); rc += cpu() - c
            for k, v in zip(("write", "read_pa", "read_pl", "read_col"), (w, rp, rl, rc)):
                res[arm][k].append(v)
    row = {"kind": kind, "rounds": rounds}
    for k in ("write", "read_pa", "read_pl", "read_col"):
        m, d = median(res["master"][k]), median(res["delta"][k])
        diffs = sorted(b - a for a, b in zip(res["master"][k], res["delta"][k]))
        row[k] = {"master": round(m, 3), "delta": round(d, 3), "paired_diffs": [round(x, 3) for x in diffs]}
    print(json.dumps(row), flush=True)
for f in out.glob("*.parquet"):
    f.unlink()
