import os, sys
from pathlib import Path
from wikidata.partitioning import fused
built, out, n = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
chunks = sys.argv[4].split(",")
for t in ["labels", "descriptions", "aliases", "links"]:
    res = out / "results" / t; res.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        c = chunks[i % len(chunks)]
        src = built / c / t / "chunk.parquet"
        if src.exists():
            os.link(src, res / f"chunk_{i}.parquet")
        os.link(fused.index_path(built / c / t, "chunk.parquet"), fused.index_path(res, f"chunk_{i}.parquet"))
