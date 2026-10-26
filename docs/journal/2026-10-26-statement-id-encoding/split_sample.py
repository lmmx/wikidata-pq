"""The quick tier's split, as the bench runs it (bench/pipeline.split_stage), its parts and
built files kept under OUT/parts: each set's claims group files as the split writes them.

Usage: split_sample.py WPQ_CHECKOUT DATA_DIR OUT_DIR"""
import os, sys
from pathlib import Path
wpq, data, out = map(lambda p: Path(p).resolve(), sys.argv[1:4])
os.environ.setdefault("WIKIDATA_RELEASE", "20261012")
sys.path.insert(0, str(wpq / "bench"))
import pipeline
files = [str(data / f"{f}.parquet") for f in ["c0", "r50", "s30", "s90"]]
out.mkdir(parents=True, exist_ok=True)
with open(out / "split.log", "w") as log:
    res = pipeline.split_stage(wpq / "src", sorted(files), 4, out, log, keep=out / "parts")
parts = res.pop("_parts")
print({k: v for k, v in res.items() if "parity" in k or "wall" in k or "cpu" in k})
for name, part, built in parts:
    print(name, part, built)
