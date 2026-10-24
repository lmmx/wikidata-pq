"""Where the time goes: each listed function of the pipeline's and finalise's modules
timed in its own process, its CPU time (every thread's: a worker runs one job at a time,
so its CPU time in a function is the function's, the native threads it waits on
included) and wall time, less the listed functions it calls. One JSON line a call, to
the file WIKIDATA_PHASES names; nothing without it.

The functions are wrapped as their modules are imported (an import hook, installed in
pipeline.py's process and, through phases_site/sitecustomize.py on PYTHONPATH, in every
worker process it starts), so the pipeline's code is unchanged and either commit of a
comparison is timed the same way. A name a module lacks is skipped.

summarise() gives each function's own CPU time between two markers (pipeline.py marks
the pipeline's run and finalise's two)."""

import collections
import functools
import importlib.abc
import importlib.machinery
import json
import os
import sys
import threading
import time

TARGETS = {
    "wikidata.dump": ["write_chunk", "reshape", "entity_row", "route_chunk"],
    "wikidata.process": ["normalise_in_schema", "normalise_map_direct", "normalise_sitelinks",
                         "atomic_sink_parquet", "check_ids", "normalise_claims_direct", "lookup_to_long",
                         "quarantine_rows", "inferred_schema"],
    "wikidata.partitioning.claims": ["statements"],
    "wikidata.partitioning.core": ["file_stats", "write_arrow", "write_table", "place_unsplit", "write_sidecar",
                                   "partition_parquet"],
    "wikidata.partitioning.transforms": ["prepare_for_partition"],
    "wikidata.push.core": ["merge_group"],
    "wikidata.compact": ["_write_file_parallel", "_write_run", "_check_glued", "_deduplicated", "_key_schema",
                         "rewrite_table", "compact_table"],
    "wikidata.sort_by_id": ["check_compacted_copy", "_read_key", "_sorted", "_check_sort", "_check_written",
                            "_boundaries", "_bucket_source", "_read_fragments", "_ipc_stream", "_read_streams",
                            "_check_schemas", "_pack_file", "_check_whole", "_check_sorted", "write_metadata",
                            "replace_local_copy", "sort_in_memory", "_sort_bucket", "bucket_key", "write_table"],
}
# A stage's CPU is its process's (its own and its native threads') while it runs, but the
# split runs its chunks in threads side by side, so its stages take their own thread's
# (the compiled reshape runs on it; what polars hands to its own threads is in neither arm)
CLOCKS = {"wikidata.dump": time.thread_time}
_local = threading.local()
_out = None


def _record(entry: dict) -> None:
    global _out
    if _out is None:
        _out = open(os.environ["WIKIDATA_PHASES"], "a", buffering=1)
    _out.write(json.dumps(entry) + "\n")


def _wrap(module: str, name: str, f):
    label = f"{module.removeprefix('wikidata.')}.{name}"
    clock = CLOCKS.get(module, time.process_time)

    @functools.wraps(f)
    def timed(*args, **kwargs):
        stack = _local.__dict__.setdefault("stack", [])
        stack.append(0.0)
        c, w = clock(), time.perf_counter()
        try:
            return f(*args, **kwargs)
        finally:
            cpu, wall = clock() - c, time.perf_counter() - w
            inner = stack.pop()
            if stack:
                stack[-1] += cpu
            _record({"fn": label, "self": cpu - inner, "wall": wall, "t": time.time()})
    timed.__wrapped_phase__ = True
    return timed


def _instrument(module) -> None:
    for name in TARGETS[module.__name__]:
        f = module.__dict__.get(name)
        if callable(f) and not getattr(f, "__wrapped_phase__", False):
            module.__dict__[name] = _wrap(module.__name__, name, f)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name not in TARGETS:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None:
            return spec
        exec_module = spec.loader.exec_module

        def exec_and_time(module):
            exec_module(module)
            _instrument(module)
        spec.loader.exec_module = exec_and_time
        return spec


def install() -> None:
    """Time the listed functions of modules imported from now on (if WIKIDATA_PHASES)."""
    if os.environ.get("WIKIDATA_PHASES") and not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())


def marker(label: str) -> None:
    if os.environ.get("WIKIDATA_PHASES"):
        _record({"marker": label, "t": time.time()})


def summarise(path: str, spans: dict[str, tuple[str, str]]) -> dict[str, dict[str, float]]:
    """Each span's (start and end markers) functions' own CPU time, in seconds."""
    rows = [json.loads(line) for line in open(path)]
    marks = {r["marker"]: r["t"] for r in rows if "marker" in r}
    out = {}
    for span, (a, b) in spans.items():
        if a in marks and b in marks:
            c = collections.Counter()
            for r in rows:
                if "fn" in r and marks[a] <= r["t"] <= marks[b]:
                    c[r["fn"]] += r["self"]
            out[span] = {k: round(v, 3) for k, v in c.items()}
    return out
