"""Run independent jobs of finalise (compaction, sort, claims_labels) in worker processes.

Each job is mostly single-threaded, so FINALISE_WORKERS run at once, each in a spawned
process (spawned, not forked, as the parent may have threads). A job's function must be
picklable (module-level) and is imported from disk by the worker, so the code must not
change while a run is going. The parent records each result as it finishes, so ledgers
are only ever written by one process.

Light jobs (`tasks_per_child=None`) share one pool per worker count for the whole run, so
a worker imports Polars, pyarrow and this package once, not once per table and stage: a
spawned worker's imports were 31% of the sort's samples on the benchmark's quick tier,
whose ~2,300 keys are run in a dozen calls (docs/journal/2026-10-24-finalise-bench.md).
"""

from __future__ import annotations

import atexit
import multiprocessing
from collections.abc import Callable, Iterator
from concurrent.futures import ALL_COMPLETED, ProcessPoolExecutor, as_completed, wait
from typing import Any

from tqdm import tqdm

from .config import FINALISE_WORKERS

# The light jobs' pools, by worker count, kept for the run
_shared: dict[int, ProcessPoolExecutor] = {}

# In a worker started ahead (in_parallel's `ahead`): the semaphore its pool's jobs take
# before they run, so at most `workers` run at once
_gate = None


def _init_gate(gate) -> None:
    global _gate
    _gate = gate


def _gated(fn: Callable[..., Any], *args: Any) -> Any:
    """Run `fn(*args)` once one of the pool's `workers` slots is free. `fn` is unpickled,
    its module imported, before this runs: a worker started ahead has imported it by the
    time a slot frees."""
    assert _gate is not None
    with _gate:
        return fn(*args)


@atexit.register
def shutdown_shared() -> None:
    """Shut the shared pools down, their workers exiting (and their CPU time counted
    as the caller's children's)."""
    for pool in _shared.values():
        pool.shutdown(cancel_futures=True)
    _shared.clear()


def in_parallel(
    fn: Callable[..., Any],
    jobs: dict[Any, tuple],
    desc: str,
    unit: str,
    total: int | None = None,
    workers: int | None = None,
    tasks_per_child: int | None = None,
    ahead: int = 0,
) -> Iterator[tuple[Any, Any]]:
    """Run `fn(*jobs[i])` for each i, `workers` (default FINALISE_WORKERS) at a time,
    yielding `(i, result)` as each finishes, under a progress bar of `total` (default the
    number of jobs; more when some were done before). A failure cancels the jobs not yet
    started and raises once those running finish.

    `tasks_per_child`: None runs the jobs in the run's shared pool of `workers`, its
    workers kept for every job of every call (light jobs); 1 replaces each worker after
    its job, in a pool of its own, so a heavy job's memory goes back to the OS. Not more than 1:
    with 8, the pool hung once its workers were due to be replaced, twice, after exactly
    6 workers x 8 jobs (labels compaction, Python 3.14.2, 2026-10-05).

    `ahead` (with `tasks_per_child=1`): this many more workers started, each taking its
    next job and importing its module but waiting to run it until one of the `workers`
    slots frees (a semaphore). Still at most `workers` jobs run, and hold their work, at
    once; the next job's worker start and imports (about half a second of a core each,
    from a spawned process) happen while the running jobs run, not after them. For many
    short heavy jobs on few slots, such as the sort's larger keys, a worker's start was
    most of each job's time, with cores idle."""
    assert tasks_per_child in (None, 1), "the pool hangs replacing workers after more jobs"
    total = len(jobs) if total is None else total
    if not jobs:
        return
    ctx = multiprocessing.get_context("spawn")
    if tasks_per_child is not None:  # heavy jobs: a pool of their own, a worker per job
        running = min(workers or FINALISE_WORKERS, len(jobs))
        started = min(running + ahead, len(jobs))
        gated = started > running
        with (
            ProcessPoolExecutor(
                started,
                mp_context=ctx,
                max_tasks_per_child=tasks_per_child,
                **({"initializer": _init_gate, "initargs": (ctx.BoundedSemaphore(running),)} if gated else {}),
            ) as pool,
            tqdm(total=total, initial=total - len(jobs), desc=desc, unit=unit) as bar,
        ):
            submit = (lambda args: pool.submit(_gated, fn, *args)) if gated else (lambda args: pool.submit(fn, *args))
            futures = {submit(args): i for i, args in jobs.items()}
            try:
                for fut in as_completed(futures):
                    yield futures[fut], fut.result()
                    bar.update()
            except BaseException:
                pool.shutdown(cancel_futures=True)
                raise
        return
    n = workers or FINALISE_WORKERS
    pool = _shared.get(n)
    if pool is None:
        pool = _shared[n] = ProcessPoolExecutor(n, mp_context=ctx)
    futures = {pool.submit(fn, *args): i for i, args in jobs.items()}
    finished = False
    with tqdm(total=total, initial=total - len(jobs), desc=desc, unit=unit) as bar:
        try:
            for fut in as_completed(futures):
                yield futures[fut], fut.result()
                bar.update()
            finished = True
        finally:
            if not finished:  # failed or abandoned: this call's jobs not yet started are
                # cancelled, those running finish, and the pool is not reused
                for fut in futures:
                    fut.cancel()
                wait(futures, return_when=ALL_COMPLETED)
                _shared.pop(n, None)
                pool.shutdown(cancel_futures=True)
