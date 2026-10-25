"""Started with every worker process of a benchmark run (on PYTHONPATH): time the
pipeline's functions there too (phases.py, loaded by path so the harness's other
modules stay off the workers' import path), and note the process's start and exit
(phases.track_process)."""
import importlib.util
import os
import sys

_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "phases.py")
_spec = importlib.util.spec_from_file_location("phases", _path)
phases = importlib.util.module_from_spec(_spec)
sys.modules["phases"] = phases
_spec.loader.exec_module(phases)
phases.install()
phases.track_process()
