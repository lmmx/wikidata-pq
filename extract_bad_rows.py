# extract_rows.py
from pathlib import Path

import polars as pl

SRC = Path(
    "data/huggingface_hub/philippesaade/wikidata/data/chunk_0-00057-of-00546.parquet"
)

df = pl.read_parquet(SRC, columns=["claims"])

# Edit these as needed
START = 400
END = 909

for i, row in enumerate(df.slice(START, END - START).iter_rows()):
    print(row[0])
