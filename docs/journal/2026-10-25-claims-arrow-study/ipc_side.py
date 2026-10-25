"""Typed claims kept beside the chunk as Arrow IPC instead of Parquet: write, size, read;
and the claims partition write (write_table) for reference."""
import json, os, sys, tempfile
from pathlib import Path
import pyarrow as pa, pyarrow.ipc as ipc, pyarrow.parquet as pq
sys.path.insert(0, str(Path(__file__).parent))
from pipeline_side_lib import claims_call, timed, med
from wikidata.partitioning.claims import statements
from wikidata.partitioning.core import write_table

DATA = Path(sys.argv[1]); OUT = Path(sys.argv[2]); chunks = sys.argv[3].split(",")
REPS = 3
res = {}
for c in chunks:
    claims = claims_call(DATA / f"{c}.parquet")
    claims = claims.combine_chunks()
    r = {"arrow_nbytes": claims.nbytes}
    with tempfile.TemporaryDirectory(dir=OUT) as d:
        for codec in (None, "lz4", "zstd"):
            f = Path(d) / f"c_{codec}.arrow"
            def w():
                with ipc.new_file(f, claims.schema, options=ipc.IpcWriteOptions(compression=codec)) as wr:
                    wr.write_table(claims, max_chunksize=1024)
            r[f"ipc_write_{codec}"] = med([timed(w)[1] for _ in range(REPS)])
            r[f"ipc_bytes_{codec}"] = f.stat().st_size
            def rd():
                with pa.memory_map(str(f)) as m:
                    return ipc.open_file(m).read_all()
            back, _ = timed(rd)
            r[f"ipc_read_{codec}"] = med([timed(rd)[1] for _ in range(REPS)])
            r[f"ipc_identical_{codec}"] = statements(back).equals(statements(claims))
        from polars_genson import write_parquet
        f = Path(d) / "typed_genson.parquet"
        r["genson_write_typed_per_entity"] = med([timed(lambda: write_parquet(claims, f, row_group_size=122_880))[1] for _ in range(REPS)])
        r["genson_typed_bytes"] = f.stat().st_size
        r["genson_typed_read_pyarrow"] = med([timed(lambda: pq.read_table(f))[1] for _ in range(REPS)])
        r["genson_typed_identical"] = statements(pq.read_table(f)).equals(statements(claims))
        st = statements(claims)
        f = Path(d) / "part.parquet"
        def wt():
            f.unlink(missing_ok=True); write_table(st, f)
        r["write_table_claims_partition"] = med([timed(wt)[1] for _ in range(REPS)])
        r["partition_bytes"] = sum(p.stat().st_size for p in Path(d).rglob("part*"))
    res[c] = r
    print(c, json.dumps(r), flush=True)
(OUT / "ipc_side.json").write_text(json.dumps(res, indent=1))
