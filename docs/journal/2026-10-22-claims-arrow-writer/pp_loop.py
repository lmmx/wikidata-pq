"""process_and_partition on each given chunk in turn in this one process (a warm worker),
printing a marker line with time.monotonic() at each chunk's process/partition boundary.
Usage: pp_loop.py <src> <workdir> <chunk>..."""
import os, shutil, sys, time
from pathlib import Path
os.environ["WIKIDATA_RELEASE"] = "20261012"
src, work = str(Path(sys.argv[1]).resolve()), Path(sys.argv[2]).resolve()
sys.path.insert(0, src)
import wikidata.main as M
for i, chunk in enumerate(sys.argv[3:]):
    d = work / f"c{i}"; shutil.rmtree(d, ignore_errors=True)
    data, out, state = d / "data", d / "out", d / "state"
    for x in (data, out, state): x.mkdir(parents=True)
    os.chdir(d); (data / f"chunk_{i}.parquet").symlink_to(Path(chunk).resolve())
    saved = os.dup(1); os.dup2(os.open(os.devnull, os.O_WRONLY), 1)
    t0 = time.monotonic(); M.process(data_dir=data, output_dir=out, repo_id="unused", state_dir=state, chunk_idx=i)
    t1 = time.monotonic(); M.partition_chunk(i, state, out); t2 = time.monotonic()
    os.dup2(saved, 1); print(f"MARK {i} {t0:.3f} {t1:.3f} {t2:.3f}", flush=True)
    os.chdir(work); shutil.rmtree(d)
