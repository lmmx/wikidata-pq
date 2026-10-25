"""A pipeline worker's claims step on a whole chunk, in a fresh process: now (the claims
call on the chunk's JSON, then statements and write_table), or from typed claims in an
Arrow IPC file (zstd) beside the chunk. Prints CPU, wall and peak RSS."""
import json, os, resource, sys, tempfile, time
from pathlib import Path
import pyarrow as pa, pyarrow.ipc as ipc
sys.path.insert(0, str(Path(__file__).parent))
mode, chunk, arrow = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
from pipeline_side_lib import claims_call
from wikidata.partitioning.claims import statements
from wikidata.partitioning.core import write_table
if mode == "make":
    t = claims_call(chunk)
    with ipc.new_file(arrow, t.schema, options=ipc.IpcWriteOptions(compression="zstd")) as w:
        for b in t.to_batches(): w.write_batch(b)
    sys.exit()
r0, t0 = resource.getrusage(resource.RUSAGE_SELF), time.perf_counter()
if mode == "now":
    claims = claims_call(chunk)
else:
    with pa.memory_map(str(arrow)) as m:
        claims = ipc.open_file(m).read_all()
with tempfile.TemporaryDirectory() as d:
    write_table(statements(claims), Path(d) / "c.parquet")
r1 = resource.getrusage(resource.RUSAGE_SELF)
print(json.dumps({"mode": mode, "chunk": chunk.name, "user": round(r1.ru_utime - r0.ru_utime, 2), "sys": round(r1.ru_stime - r0.ru_stime, 2),
                  "wall": round(time.perf_counter() - t0, 2), "peak_rss_mb": r1.ru_maxrss // 1024}))
