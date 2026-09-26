"""Time polars-genson infer_json_schema on one parquet chunk across versions.

Usage: python bench_genson_versions.py [chunk_glob] [versions...]
Defaults to chunk_0-00004 and 0.7.4 0.7.5 0.7.6. Needs uv on PATH.
Each version runs in its own `uv run` subprocess, one at a time.
"""

import subprocess
import sys
from pathlib import Path

DATA = Path(__file__).parent / "data/huggingface_hub/philippesaade/wikidata/data"
PREFIX = sys.argv[1] if len(sys.argv) > 1 else "chunk_0-00004"
VERSIONS = sys.argv[2:] or ["0.7.4", "0.7.5", "0.7.6"]

WORKER = """
import resource, sys, time
import polars as pl, polars_genson
from importlib.metadata import version

df = pl.read_parquet(sys.argv[1], columns=["claims"])
mb = df["claims"].str.len_bytes().sum() / 1e6
t = time.perf_counter()
df.genson.infer_json_schema("claims", ndjson=True, max_builders=100)
dt = time.perf_counter() - t
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
print(f"{version('polars-genson')}: rows={df.height} claims={mb:.1f}MB "
      f"time={dt:.1f}s peak_rss={rss:.1f}GB")
"""

(path,) = DATA.glob(f"{PREFIX}-*.parquet")
print(f"{path.name} ({path.stat().st_size / 1e9:.3f} GB)", flush=True)
for v in VERSIONS:
    r = subprocess.run(
        [
            "uv",
            "run",
            "-q",
            "--no-project",
            "--python",
            "3.12",
            "--exclude-newer=P0D",
            "--with",
            "polars",
            "--with",
            f"polars-genson=={v}",
            "python",
            "-c",
            WORKER,
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    out = r.stdout.strip() or f"{v}: FAILED rc={r.returncode} {r.stderr.strip()[-200:]}"
    print(out, flush=True)
