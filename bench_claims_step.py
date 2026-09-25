"""Time each stage of the claims step (`normalise_claims_direct`) on one chunk.

Usage: python bench_claims_step.py [--keep DIR] chunk_prefix [chunk_prefix...]
Each chunk runs in its own subprocess so peak RSS is per chunk.
--keep DIR writes the decoded claims frame to DIR/<chunk>.parquet, for checking
a later change's output with `DataFrame.equals`.
Needs `src` importable (run from the repo root, or with the package installed).
"""

import subprocess
import sys
from pathlib import Path

DATA = Path(__file__).parent / "data/huggingface_hub/philippesaade/wikidata/data"

WORKER = """
import resource, sys, time
from pathlib import Path
from tempfile import TemporaryDirectory
import polars as pl
from importlib.metadata import version
from polars_genson import avro_to_polars_schema, normalise_from_parquet, read_parquet_metadata
from wikidata.process import CLAIMS_INFERENCE_OPTIONS

src, keep = Path(sys.argv[1]), sys.argv[2]
key = "claims"
times = {}
def lap(name, t0):
    times[name] = time.perf_counter() - t0
    return time.perf_counter()

with TemporaryDirectory() as tmpdir:
    tmp_path = Path(tmpdir) / src.name
    t = time.perf_counter()
    normalise_from_parquet(
        input_path=src, column=key, output_path=tmp_path, output_column=key,
        wrap_root=key, **CLAIMS_INFERENCE_OPTIONS, profile=True, max_builders=1000,
    )
    t = lap("normalise_from_parquet", t)
    tmp_mb = tmp_path.stat().st_size / 1e6
    result = pl.read_parquet(tmp_path)
    t = lap("read_parquet(tmp)", t)
    avro = read_parquet_metadata(tmp_path)["genson_avro_schema"]
    schema = pl.Struct(avro_to_polars_schema(avro))
    t = lap("schema", t)
    result = result.select(pl.col(key).str.json_decode(dtype=schema)).unnest(key)
    t = lap("json_decode", t)
    out = Path(tmpdir) / "out.parquet"
    result.lazy().sink_parquet(out)
    t = lap("sink_parquet", t)
    if keep:
        Path(keep).mkdir(parents=True, exist_ok=True)
        result.write_parquet(Path(keep) / src.name)

rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
total = sum(times.values())
print(f"polars-genson {version('polars-genson')}: rows={result.height} "
      f"tmp={tmp_mb:.0f}MB total={total:.1f}s peak_rss={rss:.2f}GB")
for k, v in times.items():
    print(f"  {k:24s} {v:6.2f}s")
"""

args = sys.argv[1:]
keep = ""
if "--keep" in args:
    i = args.index("--keep")
    keep = args[i + 1]
    del args[i : i + 2]
for prefix in args:
    (path,) = DATA.glob(f"{prefix}-*.parquet")
    print(f"{path.name} ({path.stat().st_size / 1e9:.3f} GB)", flush=True)
    r = subprocess.run(
        [sys.executable, "-c", WORKER, str(path), keep],
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "PYTHONPATH": str(Path(__file__).parent / "src")},
    )
    profile = [l for l in r.stderr.splitlines() if "[profile]" in l or "profile" in l.lower()]
    print(r.stdout.rstrip() or f"FAILED rc={r.returncode} {r.stderr.strip()[-300:]}")
    for line in profile:
        print(f"    {line}")
    print(flush=True)
