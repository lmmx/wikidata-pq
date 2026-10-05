"""Run independent jobs of finalise (compaction, sort, claims_labels) in worker processes.

Each job is mostly single-threaded, so FINALISE_WORKERS run at once, each in a spawned
process (spawned, not forked, as the parent may have threads). A job's function must be
picklable (module-level) and is imported from disk by the worker, so the code must not
change while a run is going. The parent records each result as it finishes, so ledgers
are only ever written by one process.
"""

from __future__ import annotations

import multiprocessing
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

from tqdm import tqdm

from .config import FINALISE_WORKERS


def in_parallel(
    fn: Callable[..., Any],
    jobs: dict[Any, tuple],
    desc: str,
    unit: str,
    total: int | None = None,
    workers: int | None = None,
    tasks_per_child: int = 8,
) -> Iterator[tuple[Any, Any]]:
    """Run `fn(*jobs[i])` for each i, `workers` (default FINALISE_WORKERS) at a time, each
    process replaced after `tasks_per_child` jobs (its memory then freed), yielding
    `(i, result)` as each finishes, under a progress bar of `total` (default the number of
    jobs; more when some were done before). A failure cancels the jobs not yet started and
    raises once those running finish."""
    total = len(jobs) if total is None else total
    if not jobs:
        return
    ctx = multiprocessing.get_context("spawn")
    with (
        ProcessPoolExecutor(
            min(workers or FINALISE_WORKERS, len(jobs)),
            mp_context=ctx,
            max_tasks_per_child=tasks_per_child,
        ) as pool,
        tqdm(total=total, initial=total - len(jobs), desc=desc, unit=unit) as bar,
    ):
        futures = {pool.submit(fn, *args): i for i, args in jobs.items()}
        try:
            for fut in as_completed(futures):
                yield futures[fut], fut.result()
                bar.update()
        except BaseException:
            pool.shutdown(cancel_futures=True)
            raise
