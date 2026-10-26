#!/usr/bin/env python3
"""Test pool.process_chunks with stand-in chunk work and uploads (no data, no Hub).

Usage: python scripts/test_pool.py
"""

import json
import os
import signal
import time
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

from wikidata import pool
from wikidata.pool import closable, first_group, process_chunks

# Seconds of work per chunk, out of order: later chunks often finish first
DELAYS = [0.6, 0.1, 0.3, 0.05, 0.4, 0.2, 0.05, 0.5, 0.1, 0.3, 0.2, 0.05]


def work(chunk: int, log_dir: Path, fail: int | None = None, kill: int | None = None) -> None:
    start = time.time()
    (log_dir / f"{chunk}.pid").write_text(str(os.getpid()))
    time.sleep(DELAYS[chunk % len(DELAYS)])
    if chunk == fail:
        raise ValueError(f"chunk {chunk}")
    if chunk == kill:
        os.kill(os.getpid(), signal.SIGKILL)
    (log_dir / f"{chunk}.json").write_text(json.dumps([start, time.time()]))


def spans(log_dir: Path) -> dict[int, tuple[float, float]]:
    return {int(p.stem): tuple(json.loads(p.read_text())) for p in log_dir.glob("*.json")}


def pids(log_dir: Path) -> dict[int, int]:
    """The worker process each chunk ran in."""
    return {int(p.stem): int(p.read_text()) for p in log_dir.glob("*.pid")}


def max_concurrent(intervals: list[tuple[float, float]]) -> int:
    events = sorted([(s, 1) for s, _ in intervals] + [(e, -1) for _, e in intervals])
    n = best = 0
    for _, d in events:
        n += d
        best = max(best, n)
    return best


def run(
    chunks: list[int],
    ungrouped: list[int] = [],
    workers: int = 3,
    group_size: int = 4,
    fail: int | None = None,
    kill: int | None = None,
    close_fail: bool = False,
    close_secs: float = 0.0,
    chunks_per_worker: int = 1,
    more=None,
):
    tmp = TemporaryDirectory()
    log_dir = Path(tmp.name)
    started, done, groups = [], [], []

    def close(group: list[int]) -> None:
        assert group == list(range(group[0], group[-1] + 1)), f"not contiguous: {group}"
        t = time.time()
        time.sleep(close_secs)
        if close_fail:
            raise OSError("upload failed")
        groups.append((group, t, time.time()))

    error = None
    try:
        process_chunks(
            chunks,
            ungrouped,
            workers=workers,
            start=started.append,
            work=partial(work, log_dir=log_dir, fail=fail, kill=kill),
            done=done.append,
            ready=lambda g: len(g) >= group_size,
            close=close,
            chunks_per_worker=chunks_per_worker,
            more=more,
        )
    except RuntimeError as e:
        error = e
    return started, done, groups, spans(log_dir), error, tmp


def test_closable() -> None:
    assert closable({3, 4, 6}, {5, 7}) == [3, 4]
    assert closable({3, 4, 6}, {2}) == []
    assert closable({6, 3, 4}, set()) == [3, 4, 6]


def test_first_group() -> None:
    sizes = {0: 3, 1: 1, 2: 4, 3: 2, 4: 5}
    def ready(g):
        return sum(sizes[c] for c in g) >= 5

    assert first_group([0, 1, 2, 3, 4], ready) == [0, 1, 2], "the fewest from the first"
    assert first_group([0, 1], ready) == []
    assert first_group([4], ready) == [4]


def test_groups_fixed_however_chunks_finish() -> None:
    """The groups are the fewest closable chunks from the first, the last the rest: the
    same groups whatever the finishing order or the uploads' pace."""
    for close_secs in (0.0, 0.7):
        _, _, groups, _, error, _ = run(list(range(14)), close_secs=close_secs)
        assert error is None, error
        assert [g for g, *_ in groups] == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13]], groups


def test_follows_more_chunks() -> None:
    """Given `more`, the run takes the chunks it gives as they come, until it says they are
    the last, and forms the same groups as a run given them all at the start."""
    t0 = time.time()
    batches = [list(range(3, 7)), list(range(7, 9)), list(range(9, 14))]

    def more():
        due = int((time.time() - t0) / 0.8)  # a batch every 0.8 s, as the split numbers them
        out = []
        while batches and 3 - len(batches) < due:
            out += batches.pop(0)
        return out, not batches

    started, done, groups, _, error, _ = run(list(range(3)), more=more, chunks_per_worker=4)
    assert error is None, error
    assert started == list(range(14)), started
    assert [g for g, *_ in groups] == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13]], groups


def test_groups_contiguous_and_complete(chunks_per_worker: int = 1) -> None:
    chunks = list(range(2, 26))
    started, done, groups, sp, error, _ = run(
        chunks, ungrouped=[0, 1], chunks_per_worker=chunks_per_worker
    )
    assert error is None, error
    assert started == chunks, "chunks start in order"
    assert sorted(done) == chunks and done != chunks, "chunks finish out of order"
    covered = [c for g, *_ in groups for c in g]
    assert covered == list(range(26)), f"groups cover every chunk once, in order: {covered}"
    assert groups[0][0][:2] == [0, 1], "chunks partitioned before the run join the first group"
    assert all(len(g) >= 4 for g, *_ in groups[:-1])
    assert max_concurrent(list(sp.values())) == 3


def test_upload_overlaps_processing() -> None:
    _, _, groups, sp, error, _ = run(list(range(24)), close_secs=0.5)
    assert error is None, error
    g, t0, t1 = groups[0]
    after = [c for c in sp if c > g[-1] and sp[c][0] < t1 and sp[c][1] > t0]
    assert after, "chunks after a group are processed while it uploads"


def test_failed_chunk_stops_the_run(chunks_per_worker: int = 1) -> None:
    started, done, groups, _, error, _ = run(
        list(range(20)), fail=5, workers=2, chunks_per_worker=chunks_per_worker
    )
    assert error is not None and "Chunk 5 failed" in str(error), error
    assert 5 not in done
    assert max(started) <= 6, f"no chunk starts after the failure: {started}"
    assert all(c < 5 for g, *_ in groups for c in g), groups


def test_failed_upload_raises() -> None:
    _, _, groups, _, error, _ = run(list(range(12)), close_fail=True)
    assert error is not None and "Group from chunk 0 failed" in str(error), error
    assert not groups


def test_workers_take_several_chunks() -> None:
    """With `chunks_per_worker`, each worker process runs up to that many chunks, then is
    replaced; the groups and the concurrency are as with a process per chunk."""
    test_groups_contiguous_and_complete(chunks_per_worker=4)
    _, done, _, _, error, tmp = run(list(range(24)), chunks_per_worker=4)
    assert error is None, error
    by_pid: dict[int, list[int]] = {}
    for chunk, pid in pids(Path(tmp.name)).items():
        by_pid.setdefault(pid, []).append(chunk)
    assert sorted(c for cs in by_pid.values() for c in cs) == list(range(24))
    assert max(len(cs) for cs in by_pid.values()) == 4, by_pid
    assert len(by_pid) < 24, f"workers are reused: {len(by_pid)} processes for 24 chunks"


def test_failed_chunk_retires_its_worker() -> None:
    """A chunk that raises stops the run as before, and its worker takes no other chunk."""
    test_failed_chunk_stops_the_run(chunks_per_worker=8)
    _, _, _, _, error, tmp = run(list(range(20)), fail=5, workers=2, chunks_per_worker=8)
    assert error is not None and "Chunk 5 failed" in str(error) and "ValueError" in str(error)
    ran = pids(Path(tmp.name))
    assert not [c for c, pid in ran.items() if pid == ran[5] and c > 5], ran


def test_killed_worker_fails_its_chunk() -> None:
    """A worker that dies mid-chunk (here SIGKILL, as the OOM killer sends) fails that chunk
    with the signal named; the others running finish."""
    for per in (1, 8):
        started, done, _, _, error, _ = run(list(range(12)), kill=3, workers=2, chunks_per_worker=per)
        assert error is not None and "Chunk 3 failed" in str(error), error
        assert "killed by signal 9" in str(error), error
        assert 3 not in done and max(started) <= 4, (started, done)


def staged_close(log: list, merge_secs: float, push_secs: float, fail: int | None = None, hooks: bool = True):
    """A close as push.close_group runs one in a pool: a merge, then (its turn come) a push,
    each logged (stage, group's first chunk, start, end); the group from chunk `fail`
    fails in its push. Without `hooks`, a close that does not call the pool's hooks (as
    bench/pipeline.py's)."""
    def close(group: list[int]) -> None:
        t = time.time()
        time.sleep(merge_secs)
        log.append(("merge", group[0], t, time.time()))
        if hooks:
            pool.merged()
            pool.push_turn()
        t = time.time()
        time.sleep(push_secs)
        if group[0] == fail:
            raise OSError("upload failed")
        log.append(("push", group[0], t, time.time()))
    return close


def run_staged(close, merge_ahead: int = pool.MERGE_AHEAD, n: int = 16):
    error, tmp = None, TemporaryDirectory()
    try:
        process_chunks(list(range(n)), [], workers=4, start=lambda c: None,
                       work=partial(work, log_dir=Path(tmp.name)), done=lambda c: None,
                       ready=lambda g: len(g) >= 2, close=close, merge_ahead=merge_ahead)
    except RuntimeError as e:
        error = e
    tmp.cleanup()
    return error


def test_next_group_merges_as_one_is_pushed() -> None:
    """With merge_ahead, the next group merges while one is pushed: one merge at a time
    and one push at a time, in order, each push once the group before is pushed, and a
    group's merge only once the one before it has merged (at most one group merged and
    not pushed while one is pushed); the groups as without it."""
    log: list = []
    error = run_staged(staged_close(log, 0.3, 0.5))
    assert error is None, error
    merges = sorted((f, s, e) for k, f, s, e in log if k == "merge")
    pushes = sorted((f, s, e) for k, f, s, e in log if k == "push")
    assert [f for f, *_ in merges] == [f for f, *_ in pushes] == list(range(0, 16, 2)), log
    assert max_concurrent([(s, e) for _, s, e in merges]) == 1
    assert max_concurrent([(s, e) for _, s, e in pushes]) == 1
    assert all(b[1] >= a[2] for a, b in zip(pushes, pushes[1:])), "pushed in order, one at a time"
    assert all(b[1] >= a[2] for a, b in zip(merges, merges[1:])), "a merge once the one before has merged"
    # A group's merge overlaps the push before it (the link has the next group ready)
    overlapped = [b for a, b in zip(pushes, merges[1:]) if b[1] < a[2]]
    assert len(overlapped) >= len(pushes) // 2, (merges, pushes)
    # Merged ahead, not pushed: at most one group while one is pushed
    for f, s, e in pushes:
        ahead = [m for m in merges if m[0] > f and m[1] < e]
        assert len(ahead) <= 1, (f, ahead)


def test_no_merge_ahead_without_hooks_or_bound() -> None:
    """A close that does not call the hooks, or merge_ahead=0, closes one group at a time,
    as before."""
    for hooks, ahead in ((False, 1), (True, 0)):
        log: list = []
        error = run_staged(staged_close(log, 0.2, 0.2, hooks=hooks), merge_ahead=ahead, n=10)
        assert error is None, error
        spans = sorted((s, e) for _, _, s, e in log)
        assert all(b[0] >= a[1] for a, b in zip(spans, spans[1:])), (hooks, ahead, log)


def test_failed_push_stops_the_group_merged_ahead() -> None:
    """A group whose push fails raises; the group merged ahead of it is not pushed (its
    push_turn raises), and none after it is started."""
    log: list = []
    error = run_staged(staged_close(log, 0.2, 0.4, fail=2))
    assert error is not None and "Group from chunk 2 failed" in str(error), error
    assert [f for k, f, *_ in log if k == "push"] == [0], log
    assert max(f for k, f, *_ in log if k == "merge") <= 4, log


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            t = time.time()
            fn()
            print(f"{name}: ok ({time.time() - t:.1f} s)")
