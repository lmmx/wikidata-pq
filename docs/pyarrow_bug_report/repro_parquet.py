"""The same bug reached through Parquet: batches from ParquetFile.iter_batches are slices,
so RecordBatch.cast to the file's own schema fails once a batch starts past row 0.

Run: python repro_parquet.py (needs only pyarrow). Writes repro.parquet next to itself.
"""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

print("pyarrow", pa.__version__)

path = Path(__file__).with_name("repro.parquet")
t = pa.list_(pa.struct([("a", pa.int64()), ("n", pa.null())]))
pq.write_table(pa.table({"c": pa.array([None, [{"a": 1}, {"a": 2}]], type=t)}), path)

f = pq.ParquetFile(path)
for i, batch in enumerate(f.iter_batches(batch_size=1)):
    batch.validate(full=True)  # every batch is valid
    try:
        batch.cast(f.schema_arrow)
        print(f"batch {i}: RecordBatch.cast to the file's schema ok")
    except pa.ArrowInvalid as e:
        print(f"batch {i}: RecordBatch.cast to the file's schema FAILS: {e}")

# Reading the whole file in one batch, nothing is sliced and the cast succeeds
pq.read_table(path).cast(f.schema_arrow)
print("read_table(...).cast(schema) ok")
