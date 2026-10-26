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
depend on how many happen to be closable when one is formed. A group's close (merge, push,
verify) runs in a thread while the chunks after it are processed, and the next group's
close starts once it has merged (MERGE_AHEAD): the next group is merged while one is
pushed, so the link has a group ready when one is done, and groups are still pushed one
at a time, in order (see push.close_group).

Given `more`, the chunks are not all known at the start: the run follows the split as it
numbers each set's parts (dump._Numbering), taking the chunks `more` gives as they come,
until it says no more will.
"""

from __future__ import annotations

import multiprocessing
import os
import sys
import threading
import traceback
from collections import deque
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


# How many groups may merge ahead of the one uploading: 1, so that as a group is pushed
# the next is merged, ready for the link, and the staging dir holds at most two groups.
# A group merges once the one before it has merged (one merge at a time) and is pushed
# once the one before it is done (one push at a time, in order); see push.close_group
MERGE_AHEAD = 1

# The upload a pool's upload thread runs (`_Upload.run`), for the hooks a close calls
_thread = threading.local()


class _Upload(threading.Thread):
    """A group's close (merge, push, verify), in a daemon thread so an interrupted run
    exits at once (the group ledger and the merge's own files resume it). `before` is the
    upload of the group before it, still running when this one started: this one merged
    ahead of it, and is pushed once it is done (`push_turn`). `wake` is called once its
    merge is done (`merged`) and once it ends."""

    def __init__(self, close: Callable[[list[int]], None], chunks: list[int],
                 before: _Upload | None = None, wake: Callable[[], None] = lambda: None):
        super().__init__(name=f"upload_{chunks[0]}", daemon=True)
        self.close, self.chunks, self.before, self.wake = close, chunks, before, wake
        self.error: BaseException | None = None
        self.merged = threading.Event()
        self.ended = threading.Event()  # (set before its wake: is_alive() may not be false yet)

    def run(self) -> None:
        _thread.upload = self
        try:
            self.close(self.chunks)
        except BaseException as e:
            self.error = e
        finally:
            self.before = None  # (so that the uploads done are not kept, each by the next)
            self.merged.set()
            self.ended.set()
            self.wake()

    def result(self) -> None:
        self.join()
        if self.error is not None:
            raise RuntimeError(f"[push] Group from chunk {self.chunks[0]} failed") from self.error


# The hooks a group's close calls (push.core.close_group), in a pool's upload thread; in
# any other thread (a resumed group's close, before the pool starts), they do nothing


def _current() -> _Upload | None:
    return getattr(_thread, "upload", None)


def merging_ahead() -> bool:
    """Whether the group this thread closes is merged ahead: the group before it is still
    being closed (pushed or verified), so this one is pushed once that is done."""
    u = _current()
    return u is not None and (b := u.before) is not None and not b.ended.is_set()


def merged() -> None:
    """The group this thread closes is merged: the pool may start the next group's merge."""
    if (u := _current()) is not None and not u.merged.is_set():
        u.merged.set()
        u.wake()


def push_turn() -> None:
    """Wait until the group before this thread's is done, if it was still being closed
    when this one started, so that groups are pushed one at a time, in order; raise if it
    failed (this group is then not pushed, and is resumed by the next run)."""
    if (u := _current()) is None or (b := u.before) is None:
        return
    b.join()
    u.before = None
    if b.error is not None:
        raise RuntimeError(f"[push] The group before, from chunk {b.chunks[0]}, failed")


class _Uploads:
    """The groups being closed, in order, each in an _Upload: the first being merged,
    pushed or verified, and up to `ahead` more after it, each started once the one before
    it has merged. A pipe wakes the run (`fd`, for `wait`) when a merge or a close ends."""

    def __init__(self, close: Callable[[list[int]], None], ready: Callable[[list[int]], bool],
                 ahead: int = MERGE_AHEAD):
        self.close, self.ready, self.ahead = close, ready, ahead
        self.closing: deque[_Upload] = deque()
        self.fd, self._w = os.pipe()
        os.set_blocking(self.fd, False)
        os.set_blocking(self._w, False)

    def _wake(self) -> None:
        try:
            os.write(self._w, b"x")
        except (BlockingIOError, OSError):  # (full: the run has wakes to read already; closed)
            pass

    def drain(self) -> None:
        try:
            while os.read(self.fd, 4096):
                pass
        except BlockingIOError:
            pass

    def update(self, pending: set[int], unfinished: set[int], rest: bool = False) -> None:
        """Raise a failed close's error, once it has ended (the first in order); drop the
        closes done; then start the next groups' closes while fewer than 1 + `ahead` are
        running and the last has merged: each the first group of the closable chunks
        (taken out of `pending`), or with `rest` (every chunk done), the chunks left."""
        for u in self.closing:
            if u.ended.is_set() and u.error is not None:
                u.result()
        while self.closing and self.closing[0].ended.is_set():
            self.closing.popleft().result()
        while len(self.closing) <= self.ahead and (not self.closing or self.closing[-1].merged.is_set()):
            chunks = sorted(pending) if rest else closable(pending, unfinished)
            group = first_group(chunks, self.ready) or (chunks if rest else [])
            if not group:
                return
            pending -= set(group)
            before = self.closing[-1] if self.closing else None
            self.closing.append(u := _Upload(self.close, group, before, self._wake))
            u.start()

    def finish(self, interrupted: bool) -> None:
        """Wait for the closes still running and close the pipe, unless the run was
        interrupted: their threads die with it (and may still wake it, so the pipe stays
        open, its fds not reused)."""
        if interrupted:
            return
        for u in self.closing:
            if u.is_alive():
                print(f"[push] Waiting for the upload from chunk {u.chunks[0]}", flush=True)
                u.join()
        os.close(self._w)
        os.close(self.fd)


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
    merge_ahead: int = MERGE_AHEAD,
) -> None:
    """Process `chunks` (in order) `workers` at a time and close groups as they fill.

    `start(c)` runs in the parent before chunk c's process (the pull), `work(c)` in the
    process (picklable: a module-level function or a partial of one), `done(c)` in the
    parent once it succeeds; a worker process runs up to `chunks_per_worker` chunks (1: a
    fresh process for each). `ungrouped` are chunks partitioned before this run and in no
    group. A group closes (`close`, in the background) once `ready` takes the first of
    the closable chunks for one (`first_group`); once every chunk is done, the rest close
    in the same way, the last group the chunks left. Up to `merge_ahead` groups close
    at once after the one uploading, each started once the one before it has merged (a
    close that calls the hooks above: `merged`, `push_turn`), so the next group is merged
    as one is pushed. A failed chunk stops new ones
    starting; those running finish, then the run raises. `more()`, if given, returns the
    chunks that have come since it was last called (after every chunk known, in order)
    and whether they are the last.
    """
    ctx = multiprocessing.get_context("spawn")
    queue = deque(sorted(chunks))
    idle: list[_Worker] = []
    running: dict[int, _Worker] = {}
    pending_groups = set(ungrouped)
    uploads = _Uploads(close, ready, merge_ahead)
    failed: list[str] = []
    following = more is not None
    interrupted = False

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
                    uploads.update(pending_groups, set(queue))
                    wait([uploads.fd], timeout=FOLLOW_POLL)
                    uploads.drain()
                    continue
                break
            # (woken too as a group merges or its close ends, to start the next's)
            ready_now = wait([uploads.fd, *(x for w in running.values() for x in (w.conn, w.process.sentinel))],
                             timeout=FOLLOW_POLL if following else None)
            uploads.drain()
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
                uploads.update(pending_groups, set(running) | set(queue))
        if failed:
            raise RuntimeError("[run] " + "; ".join(failed))
        # The rest, once every chunk is done: groups as before, the last the chunks left
        while True:
            uploads.update(pending_groups, set(), rest=True)
            if not uploads.closing:
                break
            wait([uploads.fd])
            uploads.drain()
    except KeyboardInterrupt:
        interrupted = True  # the uploads' threads die with the run; the group ledger resumes them
        raise
    finally:
        for w in [*running.values(), *idle]:
            retire(w)
        uploads.finish(interrupted)
