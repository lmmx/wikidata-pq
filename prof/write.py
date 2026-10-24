"""The final write's settings on the claims key: zstd 3 or 1, content-defined chunking on
or off, each timed (wall and CPU) and its size."""
import sys
import time
from pathlib import Path

import pq_glue
import pyarrow.parquet as pq

from wikidata import sort_by_id as s

src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(exist_ok=True)
t = s._read_file(src)
schema = pq.read_schema(src).remove_metadata()
rg = s._row_group_rows([src])
for rnd in range(2):
    for level in (3, 1):
        for cdc in (True, False):
            dst = out / f"z{level}-{cdc}.parquet"
            c, w = time.process_time(), time.perf_counter()
            pq_glue.write_parquet(t, dst, engine="parallel", row_group_size=rg, compression="zstd",
                                  compression_level=level, content_defined_chunking=cdc,
                                  sorting_columns=[(schema.get_field_index("id"), False, False)])
            print(f"round {rnd} zstd {level} cdc {cdc!s:5s} wall {time.perf_counter() - w:5.2f}s "
                  f"cpu {time.process_time() - c:6.2f}s  {dst.stat().st_size / 1e6:6.1f} MB", flush=True)
