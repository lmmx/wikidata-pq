"""The prototype builder's claims (its exported buffers, batch by batch) as Arrow arrays
in the type genson outputs, compared with the claims call's output: the arrays, the
statements, and the claims partition file's bytes (write_table)."""
import hashlib, struct, sys, tempfile
from pathlib import Path
import pyarrow as pa
sys.path.insert(0, str(Path(__file__).parent))
from pipeline_side_lib import claims_call
from wikidata.partitioning.claims import statements
from wikidata.partitioning.core import write_table

class Reader:
    def __init__(self, b): self.b, self.i = b, 0
    def u64(self):
        v = struct.unpack_from("<Q", self.b, self.i)[0]; self.i += 8; return v
    def blob(self):
        n = self.u64(); v = self.b[self.i:self.i + n]; self.i += n; return pa.py_buffer(v)

def node(r, t):
    length, nulls = r.u64(), r.u64()
    valid, off, data, ints, nums = (r.blob() for _ in range(5))
    vb = valid if nulls else None
    if pa.types.is_null(t):
        return pa.Array.from_buffers(t, length, [None])
    if pa.types.is_large_string(t):
        return pa.Array.from_buffers(t, length, [vb, off, data], null_count=nulls)
    if pa.types.is_int64(t):
        return pa.Array.from_buffers(t, length, [vb, ints], null_count=nulls)
    if pa.types.is_float64(t):
        return pa.Array.from_buffers(t, length, [vb, nums], null_count=nulls)
    if pa.types.is_struct(t):
        kids = [node(r, f.type) for f in t]
        return pa.Array.from_buffers(t, length, [vb], null_count=nulls, children=kids)
    if pa.types.is_large_list(t):
        vt = t.value_type
        if pa.types.is_struct(vt) and [f.name for f in vt] == ["key", "value"]:
            keys = node(r, vt.field(0).type); vals = node(r, vt.field(1).type)
            entries = pa.Array.from_buffers(vt, len(keys), [None], children=[keys, vals])
        else:
            entries = node(r, vt)
        return pa.Array.from_buffers(t, length, [vb, off], null_count=nulls, children=[entries])
    raise TypeError(t)

def load(path, t):
    r, arrays = Reader(Path(path).read_bytes()), []
    while r.i < len(r.b):
        n = r.u64(); a = node(r, t); assert len(a) == n; a.validate(full=True); arrays.append(a)
    return arrays

genson = claims_call(Path(sys.argv[1]))
t = genson.schema.field("claims").type
ids = Path(sys.argv[2] + ".ids").read_text().split("\n")
batches = load(sys.argv[2], t)
mine = pa.Table.from_batches([pa.RecordBatch.from_arrays([pa.array(ids[sum(map(len, batches[:k])):sum(map(len, batches[:k + 1]))], pa.large_string()), b], schema=genson.schema) for k, b in enumerate(batches)])
print("batches", len(batches), "rows", mine.num_rows, genson.num_rows)
print("ids equal:", mine.column("id").equals(genson.column("id")))
print("claims arrays equal:", mine.column("claims").combine_chunks().equals(genson.column("claims").combine_chunks()))
a, b = statements(genson), statements(mine)
print("statements equal:", a.equals(b), a.schema.equals(b.schema), a.num_rows)
with tempfile.TemporaryDirectory() as d:
    fa, fb = Path(d) / "a.parquet", Path(d) / "b.parquet"
    write_table(a, fa); write_table(b, fb)
    ha, hb = (hashlib.sha256(f.read_bytes()).hexdigest()[:16] for f in (fa, fb))
    print("claims partition file bytes identical:", ha == hb, ha, hb)
with tempfile.TemporaryDirectory() as d:
    fa, fb, fc = Path(d) / "a.parquet", Path(d) / "a2.parquet", Path(d) / "c.parquet"
    write_table(a, fa); write_table(a, fb); write_table(b.combine_chunks(), fc)
    h = [hashlib.sha256(f.read_bytes()).hexdigest()[:16] for f in (fa, fb, fc)]
    print("genson twice:", h[0] == h[1], "| mine, rechunked:", h[0] == h[2], "| batches genson/mine:", genson.column("claims").num_chunks, mine.column("claims").num_chunks)
if not a.equals(b):
    for name in a.column_names:
        x, y = a.column(name).combine_chunks(), b.column(name).combine_chunks()
        if not x.equals(y):
            print("differs:", name, x.type)
            if pa.types.is_struct(x.type):
                for i, f in enumerate(x.type):
                    if not x.field(i).equals(y.field(i)):
                        print("  field", f.name)
                        xs, ys = x.field(i), y.field(i)
                        if pa.types.is_struct(f.type):
                            for j, g in enumerate(f.type):
                                if not xs.field(j).equals(ys.field(j)):
                                    import pyarrow.compute as pc
                                    ne = pc.invert(pc.fill_null(pc.equal(xs.field(j), ys.field(j)), False))
                                    idx = pc.indices_nonzero(ne).to_pylist()[:3]
                                    print("   sub", g.name, [(xs.field(j)[k].as_py(), ys.field(j)[k].as_py()) for k in idx])
# genson's batches: rows.div_ceil(rayon threads * 4) (polars-genson-py parquet_io.rs)
import os
n = mine.num_rows; size = max(1, -(-n // (os.cpu_count() * 4)))
whole = mine.combine_chunks()
rebatched = pa.Table.from_batches([b for k in range(0, n, size) for b in whole.slice(k, size).to_batches()], schema=mine.schema)
with tempfile.TemporaryDirectory() as d:
    fa, fb = Path(d) / "a.parquet", Path(d) / "b.parquet"
    write_table(statements(genson), fa); write_table(statements(rebatched), fb)
    print("rebatched as genson batches: partition bytes identical:", fa.read_bytes() == fb.read_bytes(), [b.num_rows for b in rebatched.to_batches()][:3])
sa, sb = statements(genson), statements(rebatched)
print("statement batches genson:", [b.num_rows for b in sa.to_batches()][:6], len(sa.to_batches()), "mine:", [b.num_rows for b in sb.to_batches()][:6], len(sb.to_batches()))
print("null claims rows:", genson.column("claims").null_count)
# rebuilt per batch, offset 0 (as a builder flushing at those boundaries makes them)
own = pa.Table.from_batches([pa.RecordBatch.from_arrays([pa.concat_arrays([c]) for c in b.columns], schema=mine.schema) for b in rebatched.to_batches()], schema=mine.schema)
own = pa.Table.from_batches([b for b in pa.ipc.open_stream(pa.BufferReader((lambda s: (lambda w: (w.write_table(rebatched), w.close(), s.getvalue())[2])(pa.ipc.new_stream(s, mine.schema)))(pa.BufferOutputStream()))).read_all().to_batches()], schema=mine.schema)
with tempfile.TemporaryDirectory() as d:
    fa, fb = Path(d) / "a.parquet", Path(d) / "b.parquet"
    write_table(statements(genson), fa); write_table(statements(own), fb)
    print("batches rebuilt at offset 0: partition bytes identical:", fa.read_bytes() == fb.read_bytes())
