"""process() on one routed chunk, in a fresh process, with release 20261012 settings.
Usage: python process_only.py <routed chunk.parquet> <workdir> <src dir>"""
import os, sys
from pathlib import Path
SRC = str(Path(sys.argv[3]).resolve())  # before the chdir below
chunk, work = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
os.environ["WIKIDATA_RELEASE"] = "20261012"
work.mkdir(parents=True, exist_ok=True); os.chdir(work)
sys.path.insert(0, SRC)
from wikidata import main as M
data, out, state = (work / d for d in ("data", "out", "state"))
for d in (data, out, state): d.mkdir(parents=True, exist_ok=True)
(data / chunk.name).symlink_to(chunk)
M.process(data_dir=data, output_dir=out, repo_id="unused", state_dir=state, chunk_idx=int(chunk.stem.split("_")[1]))
print("processed", chunk, sorted(str(p.relative_to(out)) for p in out.rglob("*.parquet")))
