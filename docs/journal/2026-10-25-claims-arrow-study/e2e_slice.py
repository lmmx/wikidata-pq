"""Per slice: the costs the direct builder removes, adds or keeps, outside the reshape
(timed in Rust): the chunk write with and without the claims JSON, the typed claims'
Arrow IPC (zstd) write and read, and the claims call; plus the identity check."""
import json, os, sys, tempfile
from pathlib import Path
import polars as pl, pyarrow as pa, pyarrow.ipc as ipc
sys.path.insert(0, str(Path(__file__).parent))
sys.argv, out_json = sys.argv[:3], sys.argv[3]
exec(open(Path(__file__).parent / "identity.py").read().split('print("batches"')[0])
from pipeline_side_lib import timed, med
REPS = 5
src = pl.read_parquet(sys.argv[1])
r = {"rows": src.height, "claims_json_bytes": int(src["claims"].str.len_bytes().sum()), "arrow_bytes": mine.nbytes}
d = Path(tempfile.mkdtemp(dir=Path(sys.argv[2]).parent))
fa, fn, fi = d / "all.parquet", d / "noclaims.parquet", d / "claims.arrow"
r["chunk_write_json"] = med([timed(lambda: src.write_parquet(fa, compression="zstd", compression_level=3))[1] for _ in range(REPS)])
r["chunk_write_no_claims"] = med([timed(lambda: src.drop("claims").write_parquet(fn, compression="zstd", compression_level=3))[1] for _ in range(REPS)])
def w():
    with ipc.new_file(fi, mine.schema, options=ipc.IpcWriteOptions(compression="zstd")) as wr:
        for b in mine.to_batches():
            wr.write_batch(b)
r["ipc_write"] = med([timed(w)[1] for _ in range(REPS)])
r["bytes_chunk_json"], r["bytes_chunk_no_claims"], r["bytes_ipc"] = fa.stat().st_size, fn.stat().st_size, fi.stat().st_size
n = mine.num_rows; size = max(1, -(-n // (os.cpu_count() * 4)))
def rd():
    with pa.memory_map(str(fi)) as m:
        t = ipc.open_file(m).read_all().combine_chunks()
    # genson's batches (see identity.py), each its own arrays
    s = pa.BufferOutputStream()
    with ipc.new_stream(s, t.schema) as wr:
        for k in range(0, n, size):
            wr.write_table(t.slice(k, size))
    return ipc.open_stream(s.getvalue()).read_all()
back, _ = timed(rd)
r["ipc_read_rebatched"] = med([timed(rd)[1] for _ in range(REPS)])
r["claims_call"] = med([timed(lambda: claims_call(Path(sys.argv[1])))[1] for _ in range(REPS)])
with tempfile.TemporaryDirectory() as t:
    f1, f2 = Path(t) / "a.parquet", Path(t) / "b.parquet"
    sa = statements(genson); sb = statements(back)
    r["write_table"] = med([timed(lambda: write_table(sa, f1))[1] for _ in range(REPS)])
    write_table(sb, f2)
    r["partition_bytes_identical"] = f1.read_bytes() == f2.read_bytes()
import shutil; shutil.rmtree(d)
print(json.dumps(r))
Path(out_json).write_text(json.dumps(r, indent=1))
