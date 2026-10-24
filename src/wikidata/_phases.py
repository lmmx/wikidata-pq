"""Profiling only (the profile/ branch): with WIKIDATA_PHASES set to a file, each listed
function of a module is timed, its process CPU time (every thread's) and wall time, in
and out of the listed functions it calls, a JSON line a call. A worker process runs one
job at a time, so its CPU time in a phase is that phase's, its native threads included."""

import functools
import json
import os
import threading
import time

PATH = os.environ.get("WIKIDATA_PHASES")
_local = threading.local()


def _wrap(name, f):
    @functools.wraps(f)
    def timed(*args, **kwargs):
        stack = _local.__dict__.setdefault("stack", [])
        stack.append(0.0)
        c, w = time.process_time(), time.perf_counter()
        try:
            return f(*args, **kwargs)
        finally:
            cpu, wall = time.process_time() - c, time.perf_counter() - w
            inner = stack.pop()
            if stack:
                stack[-1] += cpu
            with open(PATH, "a") as out:
                out.write(json.dumps({"pid": os.getpid(), "fn": name, "cpu": cpu, "self": cpu - inner,
                                      "wall": wall, "t": time.time(), "depth": len(stack)}) + "\n")
    return timed


def instrument(namespace: dict, names: list[str]) -> None:
    if not PATH:
        return
    for name in names:
        if callable(namespace.get(name)):
            namespace[name] = _wrap(name, namespace[name])


def marker(label: str) -> None:
    if PATH:
        with open(PATH, "a") as out:
            out.write(json.dumps({"marker": label, "t": time.time()}) + "\n")
