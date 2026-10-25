#!/usr/bin/env python3
"""The peak RSS the fused build adds to a process holding a chunk's columns (as a split
worker holds them): the process's peak before the call and after, in a fresh process.

Usage: python rss.py CHUNK.parquet OUT_DIR  (prints one JSON line)
"""

import json
import os
import resource
import sys
from pathlib import Path

os.environ.setdefault("WIKIDATA_RELEASE", "20261012")

import polars as pl  # noqa: E402

from wikidata.partitioning.fused import FUSED, write_fused_partitions  # noqa: E402

src, out = Path(sys.argv[1]), Path(sys.argv[2])
frame = pl.read_parquet(src, columns=["id", *FUSED.values()])
cells = sum(frame[c].str.len_bytes().sum() for c in FUSED.values())
before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
assert write_fused_partitions(frame, "chunk_0.parquet", out / "part", out / "audit", out / "routed") is None
after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(json.dumps({"chunk": src.stem, "entities": frame.height, "cell MB": round(cells / 1e6, 2),
                  "peak before MB": round(before / 1e3, 1), "peak after MB": round(after / 1e3, 1),
                  "added MB": round((after - before) / 1e3, 1)}))
