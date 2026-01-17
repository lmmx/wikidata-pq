# debug_aliases.py
from pathlib import Path

import polars as pl

from wikidata.process import normalise_map_direct

pq_path = Path(
    "data/huggingface_hub/philippesaade/wikidata/data/chunk_0-00000-of-00546.parquet"
)
output_path = Path("/tmp/debug_descs.parquet")

try:
    result = normalise_map_direct(pq_path, output_path, key="descriptions")
    print("Success!")
    print(result.head(10))
except SystemExit as e:
    print(f"Schema mismatch caught: {e}")
