"""Every group file of a kept bench run rewritten by another writer and encoding (pyarrow,
dictionaries, snappy, row groups of 50,000 rows): the same rows, other bytes.
Usage: recode.py <staging> <out>"""
import sys
from pathlib import Path

import pyarrow.parquet as pq

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
for f in src.rglob("*.parquet"):
    o = dst / f.relative_to(src)
    o.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pq.read_table(f), o, compression="snappy", use_dictionary=True, row_group_size=50_000)
