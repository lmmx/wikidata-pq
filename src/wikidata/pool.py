"""Process several chunks at once and upload groups in the background (see main.run).

Chunks are processed and partitioned in `workers` spawned worker processes (spawned, not
forked, as the parent has threads), each taking one chunk at a time. One chunk's work is
mostly single-threaded, so `workers` of them run at once. A worker takes up to
`chunks_per_worker` chunks, so that each pays the imports once rather than per chunk,
then exits and is replaced, so that whatever a worker keeps cannot build up without
bound; a worker whose chunk failed is not given another.

Chunks finish out of order but a group is a contiguous range of chunks, so a group closes
only over partitioned chunks below every chunk still running or queued: the fewest of
them, from the first, that `ready` takes for a group (`first_group`), so the groups do not
depend on how many happen to be closable when one is formed. One group uploads at a time,
in a thread, while the chunks after it are processed.

Given `more`, the chunks are not all known at the start: the run follows the split as it
numbers each set's parts (dump._Numbering), taking the chunks `more` gives as they come,
until it says no more will.
"""

from __future__ import annotations

import multiprocessing
import sys
import threading
import traceback
from collections import deque
import time
from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing.connection import Connection, wait
from multiprocessing.process import BaseProcess


# How often a run following the split looks for more chunks while it has none to wait on
FOLLOW_POLL = 2.0


def first_group(chunks: list[int], ready: Callable[[list[int]], bool]) -> list[int]:
    """The fewest of `chunks` (in order, from the first) that `ready` takes for a group, or
    none: `ready` holds of a run of chunks once it holds of a shorter one."""
    for k in range(1, len(chunks) + 1):
        if ready(chunks[:k]):
            return chunks[:k]
    return []


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


def _next_upload(upload: _Upload | None, pending: set[int], unfinished: set[int],
                 ready: Callable[[list[int]], bool], close: Callable[[list[int]], None]) -> _Upload | None:
    """The upload running: the one given while it runs (its error raised once it ends), or
    else the next group's, if the closable chunks make one (taken out of `pending`)."""
    if upload is not None:
        if upload.is_alive():
            return upload
        upload.result()
    group = first_group(closable(pending, unfinished), ready)
    if not group:
        return None
    pending -= set(group)
    upload = _Upload(close, group)
    upload.start()
    return upload


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
    more: Callable[[], tuple[list[int], bool]] | None = None,
    hold: Callable[[bool], None] | None = None,
) -> None:
    """Process `chunks` (in order) `workers` at a time and close groups as they fill.

    `start(c)` runs in the parent before chunk c's process (the pull), `work(c)` in the
    process (picklable: a module-level function or a partial of one), `done(c)` in the
    parent once it succeeds; a worker process runs up to `chunks_per_worker` chunks (1: a
    fresh process for each). `ungrouped` are chunks partitioned before this run and in no
    group. A group closes (`close`, in the background) once `ready` takes the first of
    the closable chunks for one (`first_group`); once every chunk is done, the rest close
    in the same way, the last group the chunks left. A failed chunk stops new ones
    starting; those running finish, then the run raises. `more()`, if given, returns the
    chunks that have come since it was last called (after every chunk known, in order)
    and whether they are the last. `hold(True)`, if given, is called in the parent once a
    group's close starts with no more chunks to come (`more` has said they are the last,
    or there is none), and `hold(False)` once no close is running and none can start (and
    at the end, and while chunks are still to come): so between the two, groups close back
    to back (push/turn.py).
    """
    ctx = multiprocessing.get_context("spawn")
    queue = deque(sorted(chunks))
    idle: list[_Worker] = []
    running: dict[int, _Worker] = {}
    pending_groups = set(ungrouped)
    upload: _Upload | None = None
    failed: list[str] = []
    following = more is not None
    hold = hold or (lambda on: None)

    def follow() -> None:
        nonlocal following
        if following:
            new, last = more()
            queue.extend(new)
            following = not last

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
        while queue or running or (following and not failed):
            follow()
            while queue and len(running) < workers and not failed:
                c = queue.popleft()
                start(c)
                w = idle.pop() if idle else spawn()
                w.chunk, w.taken = c, w.taken + 1
                w.conn.send(c)
                running[c] = w
            if not running:
                if following and not failed:  # waiting for the split's next chunks
                    upload = _next_upload(upload, pending_groups, set(queue), ready, close)
                    hold(False)
                    time.sleep(FOLLOW_POLL)
                    continue
                break
            # (an upload's end noticed within FOLLOW_POLL, so a held turn is let go)
            ready_now = wait([x for w in running.values() for x in (w.conn, w.process.sentinel)],
                             timeout=FOLLOW_POLL if following or upload is not None else None)
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
                # (a run following the split keeps its workers for the chunks to come)
                if w.taken < chunks_per_worker and (queue or following) and not failed:
                    idle.append(w)
                else:
                    retire(w)
            if not failed:
                upload = _next_upload(upload, pending_groups, set(running) | set(queue), ready, close)
                hold(upload is not None and not following)
        if failed:
            raise RuntimeError("[run] " + "; ".join(failed))
        if upload is not None:
            upload.result()
            upload = None
        # The rest, once every chunk is done: groups as before, the last the chunks left
        while pending_groups:
            hold(True)
            group = first_group(sorted(pending_groups), ready) or sorted(pending_groups)
            pending_groups -= set(group)
            close(group)
        hold(False)
    except KeyboardInterrupt:
        upload = None  # its thread dies with the run; the group ledger resumes it
        raise
    finally:
        for w in [*running.values(), *idle]:
            retire(w)
        if upload is not None and upload.is_alive():
            print(f"[push] Waiting for the upload from chunk {upload.chunks[0]}", flush=True)
            upload.join()
        hold(False)
