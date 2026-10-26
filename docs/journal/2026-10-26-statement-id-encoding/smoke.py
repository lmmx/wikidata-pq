import sys, types
from pathlib import Path
sys.path.insert(0, sys.argv[1])
stub = types.ModuleType("wikidata_reshape"); stub.__getattr__ = lambda n: None
sys.modules["wikidata_reshape"] = stub
import pyarrow as pa, pyarrow.parquet as pq
from wikidata.partitioning.core import write_table, column_encoding
from wikidata.compact import _write_file, _write_file_parallel
from wikidata.sort_by_id import _sorting
src = Path(sys.argv[2]); out = Path(sys.argv[3])
t = pq.read_table(src)
print("encoding", column_encoding(t.schema), "labels-like", column_encoding(pa.schema([("id", pa.string())])))
def enc(p):
    md = pq.read_metadata(p); names = [md.schema.column(i).path for i in range(md.num_columns)]
    c = md.row_group(0).column(names.index("statement_id")); return c.encodings, c.has_dictionary_page
for name, f in [("write_table", lambda d: write_table(t, d)),
                ("_write_file", lambda d: _write_file(d, t.schema, t.to_batches(50_000), 50_000, _sorting(t.schema, "id"))),
                ("_write_file_parallel", lambda d: _write_file_parallel(d, t.schema, t.to_batches(50_000), 50_000, _sorting(t.schema, "id"))),
                ("_write_file_parallel scratch", lambda d: _write_file_parallel(d, t.schema, t.to_batches(50_000), 50_000, scratch=True))]:
    d = out / f"{name}.parquet"; f(d)
    got = pq.read_table(d)
    print(name, d.stat().st_size, enc(d), got.equals(t), got.schema.equals(t.schema, check_metadata=True))
