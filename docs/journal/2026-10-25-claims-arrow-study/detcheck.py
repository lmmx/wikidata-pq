import hashlib, sys, tempfile
from pathlib import Path
import pyarrow as pa, pyarrow.ipc as ipc
sys.path.insert(0, str(Path(__file__).parent))
from pipeline_side_lib import claims_call
from wikidata.partitioning.claims import statements
from wikidata.partitioning.core import write_table
a = statements(claims_call(Path(sys.argv[1])))
def h(t):
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "x.parquet"; write_table(t, f); return hashlib.sha256(f.read_bytes()).hexdigest()[:12]
sink = pa.BufferOutputStream()
with ipc.new_stream(sink, a.schema) as w: w.write_table(a)
copy = ipc.open_stream(sink.getvalue()).read_all()
print("same table:", h(a), h(a), h(a), "| ipc copy:", h(copy), h(copy), "| combined:", h(a.combine_chunks()))
