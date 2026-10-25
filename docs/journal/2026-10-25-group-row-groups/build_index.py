"""Build each bench chunk's four tables in the one-file layout (as the split does) into
OUT/{chunk}/{table}/, and print per chunk: entities, and per table keys, rows, bytes."""
import sys, json
from pathlib import Path
import polars as pl
from wikidata.partitioning import fused
from wikidata.partitioning.fused import FUSED, large_strings
out = Path(sys.argv[1])
for src in map(Path, sys.argv[2:]):
    frame = pl.read_parquet(src, columns=["id", *FUSED.values()])
    texts = {t: large_strings(frame.get_column(c)) for t, c in FUSED.items()}
    d = out / src.stem
    sizes, why = fused.write_one_file(frame.get_column("id"), texts, None, d, "chunk.parquet")
    assert sizes is not None, why
    rec = {"chunk": src.stem, "entities": frame.height}
    for t in FUSED:
        keys = fused.built_index(d / t, "chunk.parquet")
        rec[str(t)] = {"keys": len(keys), "rows": sum(k[1] for k in keys), "bytes": sizes[t], "rgs": sum(k[6] for k in keys)}
    print(json.dumps(rec), flush=True)
