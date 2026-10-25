"""Each bench chunk file processed and partitioned alone (as bench_work does); its
partition files' bytes and rows per table.

Usage: sizes.py <dir with data/{name}.parquet, writes ana/sizes.json> <src> <names, comma-separated>"""
import json, os, sys, shutil, collections
from pathlib import Path
SP = Path(sys.argv[1]); SRC = Path(sys.argv[2]); names = sys.argv[3].split(",")
run = SP / "sizes-run"; shutil.rmtree(run, ignore_errors=True)
work = run / "releases" / "20261012"
data, state, results = work / "data", work / "state", work / "results"
for d in (data, state, results): d.mkdir(parents=True)
for i, n in enumerate(names):
    (data / f"chunk_{i}.parquet").symlink_to(SP / "data" / f"{n}.parquet")
os.chdir(run)
os.environ["WIKIDATA_RELEASE"] = "20261012"; os.environ["HF_HUB_OFFLINE"] = "1"
sys.path.insert(0, str(SRC))
import types; sys.modules.setdefault("wikidata_reshape", types.SimpleNamespace(entity_rows=None))  # (the split's, unused here)
from wikidata.main import partition_chunk, process
from wikidata.config import Table
import pyarrow.parquet as pq
out = {}
for i, n in enumerate(names):
    process(chunk_idx=i, data_dir=data.resolve(), output_dir=results.resolve(), repo_id="unused", state_dir=state.resolve())
    partition_chunk(i, state.resolve(), results.resolve())
    per = {}
    for t in Table:
        files = [p for p in (results / t).rglob("*.parquet") if p.parent != results / t]
        per[t.value] = {"bytes": sum(p.stat().st_size for p in files),
                        "rows": sum(pq.read_metadata(p).num_rows for p in files), "files": len(files)}
        for p in files: p.unlink()
    out[n] = per
    print(n, json.dumps(per), flush=True)
json.dump(out, open(SP / "ana" / "sizes.json", "w"))
shutil.rmtree(run)
