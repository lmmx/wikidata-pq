#!/usr/bin/env python3
"""Each table's rows, partition files and bytes on chunk files: Polars' sink (the
pipeline) against the fused writer. Prints one JSON line a chunk.

Usage: python sizes.py CHUNK.parquet ...
"""

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("WIKIDATA_RELEASE", "20261012")
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

import contextlib, io  # noqa: E401,E402
import polars as pl  # noqa: E402

from test_fused_terms import COLUMNS, pipeline  # noqa: E402
from wikidata.partitioning.fused import FUSED, write_fused_partitions  # noqa: E402

for src in map(Path, sys.argv[1:]):
    with tempfile.TemporaryDirectory(dir=os.environ.get("CPU_TMP")) as tmp:
        tmp = Path(tmp)
        with contextlib.redirect_stdout(io.StringIO()):
            tables = pipeline(src, tmp / "ref", "chunk_0.parquet")
        frame = pl.read_parquet(src, columns=COLUMNS)
        assert write_fused_partitions(frame, "chunk_0.parquet", tmp / "f" / "part", tmp / "f" / "audit", tmp / "f" / "routed") is None
        out = {"chunk": src.stem}
        for t in FUSED:
            ref = list((tmp / "ref" / "part" / t).rglob("*.parquet"))
            mine = list((tmp / "f" / "part" / t).rglob("*.parquet"))
            out[t] = {"rows": tables[t].height, "files": len(ref), "sink bytes": sum(f.stat().st_size for f in ref),
                      "fused bytes": sum(f.stat().st_size for f in mine)}
        print(json.dumps(out), flush=True)
