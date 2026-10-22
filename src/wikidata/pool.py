"""Process several chunks at once and upload groups in the background (see main.run).

Chunks are processed and partitioned in `workers` spawned worker processes (spawned, not
forked, as the parent has threads), each taking one chunk at a time. One chunk's work is
mostly single-threaded, so `workers` of them run at once. A worker takes up to
`chunks_per_worker` chunks, so that each pays the imports once rather than per chunk,
then exits and is replaced, so that whatever a worker keeps cannot build up without
bound; a worker whose chunk failed is not given another.

Chunks finish out of order but a group is a contiguous range of chunks, so a group closes
only over partitioned chunks below every chunk still running or queued. One group uploads
at a time, in a thread, while the chunks after it are processed.
"""

from __future__ import annotations

import multiprocessing
import sys
import threading
import traceback
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing.connection import Connection, wait
from multiprocessing.process import BaseProcess


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


def _serve(conn: Connection, work: Callable[[int], None]) -> None:
    """A worker: `work(c)` for each chunk index `c` received on `conn` until it receives
    None, replying None once a chunk succeeds, or (printing it, and exiting) the traceback
    of its failure."""
    while (chunk := conn.recv()) is not None:
        try:
            work(chunk)
        except BaseException:
            print(error := traceback.format_exc(), file=sys.stderr, flush=True)
            conn.send(error)
            return
        conn.send(None)


@dataclass
class _Worker:
    process: BaseProcess
    conn: Connection
    chunk: int | None = None
    taken: int = 0


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
    chunks_per_worker: int = 1,
) -> None:
    """Process `chunks` (in order) `workers` at a time and close groups as they fill.

    `start(c)` runs in the parent before chunk c's process (the pull), `work(c)` in the
    process (picklable: a module-level function or a partial of one), `done(c)` in the
    parent once it succeeds; a worker process runs up to `chunks_per_worker` chunks (1: a
    fresh process for each). `ungrouped` are chunks partitioned before this run and in no
    group. A group closes (`close`, in the background) once `ready` says the closable
    chunks are enough, and the rest close once every chunk is done. A failed chunk stops
    new ones starting; those running finish, then the run raises.
    """
    ctx = multiprocessing.get_context("spawn")
    queue = deque(sorted(chunks))
    idle: list[_Worker] = []
    running: dict[int, _Worker] = {}
    pending_groups = set(ungrouped)
    upload: _Upload | None = None
    failed: list[str] = []

    def spawn() -> _Worker:
        conn, child = ctx.Pipe()
        p = ctx.Process(target=_serve, args=(child, work), name="chunk_worker")
        p.start()
        child.close()
        return _Worker(p, conn)

    def retire(w: _Worker) -> None:
        try:
            w.conn.send(None)
        except OSError:
            pass
        w.process.join()
        w.conn.close()

    try:
        while queue or running:
            while queue and len(running) < workers and not failed:
                c = queue.popleft()
                start(c)
                w = idle.pop() if idle else spawn()
                w.chunk, w.taken = c, w.taken + 1
                w.conn.send(c)
                running[c] = w
            if not running:
                break
            ready_now = wait([x for w in running.values() for x in (w.conn, w.process.sentinel)])
            for c, w in list(running.items()):
                if w.conn not in ready_now and w.process.sentinel not in ready_now:
                    continue
                try:
                    error = w.conn.recv()
                except EOFError:  # the worker died mid-chunk
                    w.process.join()
                    code = w.process.exitcode or 0
                    cause = f"killed by signal {-code}" if code < 0 else f"exit code {code}"
                    error = f"its worker died, {cause}"
                del running[c]
                w.chunk = None
                if error is not None:
                    cause = error.strip().splitlines()[-1]
                    failed.append(f"Chunk {c} failed in its subprocess ({cause})")
                    retire(w)
                    continue
                done(c)
                pending_groups.add(c)
                if w.taken < chunks_per_worker and queue and not failed:
                    idle.append(w)
                else:
                    retire(w)
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
        for w in [*running.values(), *idle]:
            retire(w)
        if upload is not None and upload.is_alive():
            print(f"[push] Waiting for the upload from chunk {upload.chunks[0]}", flush=True)
            upload.join()
