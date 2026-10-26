import os
import sys, json
from pathlib import Path
WT = Path(os.environ.get("REPO", "."))
sys.path.insert(0, str(WT / "bench"))
import pipeline
D = Path(os.environ["DATA"])  # the bench data (e8c97e1)
W = Path(os.environ["WORK"])
files = [str(D / f) for f in ("c0.parquet", "r50.parquet", "s30.parquet", "s90.parquet")]
with open(W / "split.log", "w") as log:
    out = pipeline.split_stage(WT, files, 4, W, log, keep=W / "parts")
print(json.dumps({k: v for k, v in out.items() if k != "_parts"}, indent=0)[:2000])
print(out.get("_parts"))
