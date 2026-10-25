"""Slices of the bench's chunk files: their rows (as split writes them) and the same
rows as dump lines (bench/dumpslice.entity_line), for the split-side prototype."""
import sys
from pathlib import Path
import polars as pl
sys.path.insert(0, sys.argv[1])  # bench/
from dumpslice import entity_line
DATA, OUT = Path(sys.argv[2]), Path(sys.argv[3])
for spec in sys.argv[4:]:
    name, n = spec.split(":")
    df = pl.read_parquet(DATA / f"{name}.parquet").head(int(n))
    df.write_parquet(OUT / f"{name}_{n}.parquet", compression="zstd", compression_level=3)
    with open(OUT / f"{name}_{n}.jsonl", "wb") as f:
        for row in df.iter_rows(named=True):
            f.write(entity_line(row) + b"\n")
    print(name, n, (OUT / f"{name}_{n}.jsonl").stat().st_size / 1e6, "MB lines")
