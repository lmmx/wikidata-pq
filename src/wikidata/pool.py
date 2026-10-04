"""Process several chunks at once and upload groups in the background (see main.run).

Each chunk is processed and partitioned in its own spawned process, so the memory it used
goes back to the OS when it exits (freed native memory otherwise stays in the allocator
and RSS climbs chunk after chunk); spawned, not forked, as the parent has threads. One
chunk's work is mostly single-threaded, so `workers` of them run at once.

Chunks finish out of order but a group is a contiguous range of chunks, so a group closes
only over partitioned chunks below every chunk still running or queued. One group uploads
at a time, in a thread, while the chunks after it are processed.
"""

from __future__ import annotations

import multiprocessing
import threading
from collections import deque
from collections.abc import Callable
from multiprocessing.connection import wait


def closable(ungrouped: set[int], unfinished: set[int]) -> list[int]:
    """The partitioned chunks not yet in a group that can make one: those below every
    unfinished (running or queued) chunk, in order."""
    bound = min(unfinished, default=None)
    return sorted(c for c in ungrouped if bound is None or c < bound)


class _Upload(threading.Thread):
    """A group's close (merge, push, verify), in a daemon thread so an interrupted run
    exits at once (the group ledger resumes it)."""

    def __init__(self, close: Callable[[list[int]], None], chunks: list[int]):
        super().__init__(name=f"upload_{chunks[0]}", daemon=True)
        self.close, self.chunks = close, chunks
        self.error: BaseException | None = None

    def run(self) -> None:
        try:
            self.close(self.chunks)
        except BaseException as e:
            self.error = e

    def result(self) -> None:
        self.join()
        if self.error is not None:
            raise RuntimeError(f"[push] Group from chunk {self.chunks[0]} failed") from self.error


def process_chunks(
    chunks: list[int],
    ungrouped: list[int],
    *,
    workers: int,
    start: Callable[[int], None],
    work: Callable[[int], None],
    done: Callable[[int], None],
    ready: Callable[[list[int]], bool],
    close: Callable[[list[int]], None],
) -> None:
    """Process `chunks` (in order) `workers` at a time and close groups as they fill.

    `start(c)` runs in the parent before chunk c's process (the pull), `work(c)` in the
    process (picklable: a module-level function or a partial of one), `done(c)` in the
    parent once it succeeds. `ungrouped` are chunks partitioned before this run and in no
    group. A group closes (`close`, in the background) once `ready` says the closable
    chunks are enough, and the rest close once every chunk is done. A failed chunk stops
    new ones starting; those running finish, then the run raises.
    """
    ctx = multiprocessing.get_context("spawn")
    queue = deque(sorted(chunks))
    running: dict[int, multiprocessing.process.BaseProcess] = {}
    pending_groups = set(ungrouped)
    upload: _Upload | None = None
    failed: list[str] = []
    try:
        while queue or running:
            while queue and len(running) < workers and not failed:
                c = queue.popleft()
                start(c)
                p = ctx.Process(target=work, args=(c,), name=f"chunk_{c}")
                p.start()
                running[c] = p
            if not running:
                break
            finished = wait([p.sentinel for p in running.values()])
            for c, p in list(running.items()):
                if p.sentinel not in finished:
                    continue
                p.join()
                del running[c]
                if p.exitcode != 0:
                    cause = (
                        f"killed by signal {-p.exitcode}"
                        if p.exitcode < 0
                        else f"exit code {p.exitcode}"
                    )
                    failed.append(f"Chunk {c} failed in its subprocess ({cause})")
                    continue
                done(c)
                pending_groups.add(c)
            if upload is not None and not upload.is_alive():
                upload.result()
                upload = None
            if upload is None and not failed:
                group = closable(pending_groups, set(running) | set(queue))
                if group and ready(group):
                    pending_groups -= set(group)
                    upload = _Upload(close, group)
                    upload.start()
        if failed:
            raise RuntimeError("[run] " + "; ".join(failed))
        if upload is not None:
            upload.result()
            upload = None
        # The remainder, once every chunk is done
        if pending_groups:
            close(sorted(pending_groups))
    except KeyboardInterrupt:
        upload = None  # its thread dies with the run; the group ledger resumes it
        raise
    finally:
        for p in running.values():
            p.join()
        if upload is not None and upload.is_alive():
            print(f"[push] Waiting for the upload from chunk {upload.chunks[0]}", flush=True)
            upload.join()
