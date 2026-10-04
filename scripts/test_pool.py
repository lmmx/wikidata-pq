#!/usr/bin/env python3
"""Test pool.process_chunks with stand-in chunk work and uploads (no data, no Hub).

Usage: python scripts/test_pool.py
"""

import json
import time
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

from wikidata.pool import closable, process_chunks

# Seconds of work per chunk, out of order: later chunks often finish first
DELAYS = [0.6, 0.1, 0.3, 0.05, 0.4, 0.2, 0.05, 0.5, 0.1, 0.3, 0.2, 0.05]


def work(chunk: int, log_dir: Path, fail: int | None = None) -> None:
    start = time.time()
    time.sleep(DELAYS[chunk % len(DELAYS)])
    if chunk == fail:
        raise ValueError(f"chunk {chunk}")
    (log_dir / f"{chunk}.json").write_text(json.dumps([start, time.time()]))


def spans(log_dir: Path) -> dict[int, tuple[float, float]]:
    return {int(p.stem): tuple(json.loads(p.read_text())) for p in log_dir.glob("*.json")}


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
    close_fail: bool = False,
    close_secs: float = 0.0,
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
            work=partial(work, log_dir=log_dir, fail=fail),
            done=done.append,
            ready=lambda g: len(g) >= group_size,
            close=close,
        )
    except RuntimeError as e:
        error = e
    return started, done, groups, spans(log_dir), error, tmp


def test_closable() -> None:
    assert closable({3, 4, 6}, {5, 7}) == [3, 4]
    assert closable({3, 4, 6}, {2}) == []
    assert closable({6, 3, 4}, set()) == [3, 4, 6]


def test_groups_contiguous_and_complete() -> None:
    chunks = list(range(2, 26))
    started, done, groups, sp, error, _ = run(chunks, ungrouped=[0, 1])
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


def test_failed_chunk_stops_the_run() -> None:
    started, done, groups, _, error, _ = run(list(range(20)), fail=5, workers=2)
    assert error is not None and "Chunk 5 failed" in str(error), error
    assert 5 not in done
    assert max(started) <= 6, f"no chunk starts after the failure: {started}"
    assert all(c < 5 for g, *_ in groups for c in g), groups


def test_failed_upload_raises() -> None:
    _, _, groups, _, error, _ = run(list(range(12)), close_fail=True)
    assert error is not None and "Group from chunk 0 failed" in str(error), error
    assert not groups


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            t = time.time()
            fn()
            print(f"{name}: ok ({time.time() - t:.1f} s)")
