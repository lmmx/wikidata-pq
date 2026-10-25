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
from concurrent.futures import ALL_COMPLETED, FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
from typing import Any

from tqdm import tqdm

from .config import FINALISE_MEMORY_BYTES, FINALISE_WORKERS

# The light jobs' pools, by worker count, kept for the run
_shared: dict[int, ProcessPoolExecutor] = {}


@atexit.register
def shutdown_shared() -> None:
    """Shut the shared pools down, their workers exiting (and their CPU time counted
    as the caller's children's)."""
    for pool in _shared.values():
        pool.shutdown(cancel_futures=True)
    _shared.clear()


def memory_budget() -> int | None:
    """The bytes finalise's heavy jobs may take at once: FINALISE_MEMORY_BYTES if set,
    else the memory available now (Linux's MemAvailable, which counts reclaimable page
    cache as available), else None (no budget: the worker count alone limits them)."""
    if FINALISE_MEMORY_BYTES:
        return FINALISE_MEMORY_BYTES
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def in_parallel(
    fn: Callable[..., Any],
    jobs: dict[Any, tuple],
    desc: str,
    unit: str,
    total: int | None = None,
    workers: int | None = None,
    tasks_per_child: int | None = None,
    memory: dict[Any, int] | None = None,
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

    `memory` (heavy jobs only): each job's estimated peak bytes. Jobs are then started
    in order while those running and the next fit the budget (memory_budget, read once
    at the call), at least one at a time, so a machine short of memory runs fewer at
    once rather than swapping or being killed. Where every worker's job fits, as many
    run as without a budget. Which jobs run, their results and the ledgers are the
    same either way."""
    assert tasks_per_child in (None, 1), "the pool hangs replacing workers after more jobs"
    total = len(jobs) if total is None else total
    if not jobs:
        return
    ctx = multiprocessing.get_context("spawn")
    if tasks_per_child is not None:  # heavy jobs: a pool of their own, a worker per job
        n = min(workers or FINALISE_WORKERS, len(jobs))
        with (
            ProcessPoolExecutor(
                n,
                mp_context=ctx,
                max_tasks_per_child=tasks_per_child,
            ) as pool,
            tqdm(total=total, initial=total - len(jobs), desc=desc, unit=unit) as bar,
        ):
            if memory is None:
                futures = {pool.submit(fn, *args): i for i, args in jobs.items()}
                try:
                    for fut in as_completed(futures):
                        yield futures[fut], fut.result()
                        bar.update()
                except BaseException:
                    pool.shutdown(cancel_futures=True)
                    raise
                return
            yield from _budgeted(pool, fn, jobs, memory, n, desc, bar)
        return
    assert memory is None, "a memory budget is for heavy jobs, a worker each"
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


def _budgeted(pool, fn, jobs, memory, workers, desc, bar) -> Iterator[tuple[Any, Any]]:
    """in_parallel's heavy jobs started in order while those running and the next fit
    the memory budget (at least one running), each yielded as it finishes."""
    budget = memory_budget()
    pending = list(jobs)
    running: dict[Any, Any] = {}  # future -> job
    used = 0
    fewest = workers
    try:
        while pending or running:
            while pending and len(running) < workers and (
                not running or budget is None or used + memory[pending[0]] <= budget
            ):
                i = pending.pop(0)
                running[pool.submit(fn, *jobs[i])] = i
                used += memory[i]
            if pending and len(running) < fewest:
                fewest = len(running)
                print(
                    f"{desc}: {len(running)} at once, not {workers}: the next needs about"
                    f" {memory[pending[0]] / 1e9:.1f} GB, with {used / 1e9:.1f} GB of"
                    f" {budget / 1e9:.1f} GB taken",
                    flush=True,
                )
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for fut in done:
                i = running.pop(fut)
                used -= memory[i]
                yield i, fut.result()
                bar.update()
    except BaseException:
        pool.shutdown(cancel_futures=True)
        raise
