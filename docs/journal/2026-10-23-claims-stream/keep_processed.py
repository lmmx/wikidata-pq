"""process() each chunk file and keep its processed tables in <out>/<table>/<stem>.parquet."""
import os, shutil, sys
from pathlib import Path
os.environ["WIKIDATA_RELEASE"] = "20261012"
src, out = sys.argv[1], Path(sys.argv[2]).resolve()
run = out.with_suffix(".run"); shutil.rmtree(run, ignore_errors=True)
work = run / "releases" / "20261012"
data, state, results = work / "data", work / "state", work / "results"
for d in (data, state, results): d.mkdir(parents=True)
os.chdir(run); sys.path.insert(0, str(Path(src).resolve()))
import wikidata.main as main
from wikidata.config import Table
for i, (label, f) in enumerate(a.split("=", 1) for a in sys.argv[3:]):
    f = Path(f)
    (data / f"chunk_{i}.parquet").symlink_to(f.resolve())
    main.process(chunk_idx=i, data_dir=data.resolve(), output_dir=results.resolve(), state_dir=state.resolve(), repo_id="x/none")
    for t in Table:
        p = results / t / f"chunk_{i}.parquet"
        if p.exists():
            (out / t).mkdir(parents=True, exist_ok=True); p.replace(out / t / f"{label}.parquet")
shutil.rmtree(run, ignore_errors=True)
