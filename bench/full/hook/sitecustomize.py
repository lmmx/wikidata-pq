"""Started with every Python process of a compute-benchmark run (bench/full/run.py puts
this directory on PYTHONPATH): with BENCH_FULL_HUB set, the stand-in Hub
(bench/full/standin.py, loaded by path, so nothing else of the harness is importable)
answers the process's Hub calls, and wikidata.main's steps are timed (BENCH_FULL_SPANS)."""
import importlib.util
import os
import sys

if os.environ.get("BENCH_FULL_HUB"):
    _path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "standin.py")
    _spec = importlib.util.spec_from_file_location("bench_full_standin", _path)
    _standin = importlib.util.module_from_spec(_spec)
    sys.modules["bench_full_standin"] = _standin
    _spec.loader.exec_module(_standin)
    _standin.hook(os.environ["BENCH_FULL_HUB"])
