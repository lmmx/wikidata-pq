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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            t = time.time()
            fn()
            print(f"{name}: ok ({time.time() - t:.1f} s)")
