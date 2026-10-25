"""An error in a split worker ends the split with the error (exit 1 and its traceback),
not a crash: with the claims built and as JSON, on a dump slice made of the chunk files
given (bench/dumpslice.py), the worker failing at its routing (after its reshape) or at
its start. Before the progress bar was closed on every path, the error's traceback was
followed by a segfault (exit 139) as the interpreter shut down.

    .venv/bin/python scripts/test_split_errors.py chunk.parquet ...
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))

from dumpslice import make  # noqa: E402

RELEASE = "20261012"
CODE = """
import sys
sys.path.insert(0, {src!r})
from wikidata import dump
dump.CHUNK_ENTITIES = 1000
def fail(*args, **kwargs):
    raise ValueError("a worker failed")
setattr(dump, {at!r}, fail)
dump.split()
"""


def main(chunks: list[str]) -> None:
    with tempfile.TemporaryDirectory() as work:
        work = Path(work)
        dump_slice = work / "slice.json.bz2"
        make(dump_slice, chunks)
        for mode in ("built", "json"):
            for at in ("route_parts", "write_chunk"):
                d = work / f"{mode}-{at}"
                (d / "releases" / RELEASE / "dump").mkdir(parents=True)
                os.link(dump_slice, d / "releases" / RELEASE / "dump" / f"wikidata-{RELEASE}-all.json.bz2")
                env = {**os.environ, "WIKIDATA_RELEASE": RELEASE, "WIKIDATA_SPLIT_CLAIMS": mode, "HF_HUB_OFFLINE": "1"}
                proc = subprocess.run([sys.executable, "-c", CODE.format(src=str(ROOT / "src"), at=at)],
                                      cwd=d, env=env, capture_output=True, text=True)
                assert proc.returncode == 1 and "ValueError: a worker failed" in proc.stderr, (
                    f"{mode}, failing at {at}: exit {proc.returncode}\n{proc.stderr[-2000:]}")
                print(f"{mode}, a worker failing at {at}: exit 1, with its traceback")


if __name__ == "__main__":
    main(sys.argv[1:])
