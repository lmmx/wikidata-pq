"""Each key's group file both ways, copied (copy_row_groups) and merged
(merge_row_groups): same rows by pyarrow, pq_glue and Polars, same Arrow schema, same
footer_size; CPU of each, and their layouts."""
import sys, time, json, shutil
from pathlib import Path
import polars as pl
import pyarrow.parquet as pq
import pq_glue
from wikidata.partitioning import fused, rowgroups as r
from wikidata.sort_by_id import footer_size

root = Path(sys.argv[1]); check = sys.argv[2] == "check" if len(sys.argv) > 2 else True
for table in ["labels", "descriptions", "aliases", "links"]:
    res = root / "results" / table
    processed = sorted(res.glob("chunk_*.parquet"), key=lambda p: int(p.stem.split("_")[1]))
    indexes = [fused.built_index(p.parent, p.name) for p in processed]
    out = {"table": table}
    for how in ("copy", "merge"):
        work = root / "cmp" / how / table
        shutil.rmtree(work, ignore_errors=True); work.mkdir(parents=True)
        t0 = time.process_time()
        F = r.Footer if how == "copy" else r.ChunkFooter
        footers = [F(p) if keys else None for p, keys in zip(processed, indexes)]
        t1 = time.process_time()
        parts = {}
        for footer, keys in zip(footers, indexes):
            for key, _, _, _, _, first, n in keys:
                parts.setdefault(key, []).append((footer, first, n))
        for k, ps in parts.items():
            dst = work / f"{k}.parquet"
            if how == "copy":
                r.copy_row_groups(dst, ps)
            else:
                r.merge_row_groups(dst, ps, 122_880)
        t2 = time.process_time()
        files = sorted(work.glob("*.parquet"))
        out[how] = {"parse": round(t1 - t0, 3), "write": round(t2 - t1, 3),
                    "bytes": sum(f.stat().st_size for f in files),
                    "rgs": sum(pq.read_metadata(f).num_row_groups for f in files), "files": len(files)}
    if check:
        same = True
        for f in sorted((root / "cmp" / "copy" / table).glob("*.parquet")):
            g = root / "cmp" / "merge" / table / f.name
            a = pq.read_table(f)
            ok = (a.equals(pq.read_table(g)) and a.equals(pq.ParquetFile(g).read(use_threads=False))
                  and pq_glue.read_parquet(f).equals(pq_glue.read_parquet(g))
                  and pl.read_parquet(f).equals(pl.read_parquet(g))
                  and pq.read_schema(f).equals(pq.read_schema(g), check_metadata=True)
                  and footer_size([f]) == footer_size([g])
                  and pq.read_metadata(f).num_rows == pq.read_metadata(g).num_rows)
            if not ok:
                print("DIFFERS", table, f.name); same = False
        out["same"] = same
    print(json.dumps(out), flush=True)
