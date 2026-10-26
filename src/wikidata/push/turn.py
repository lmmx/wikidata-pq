"""The main set's turn on the link before the scholarly set's, while both sets' runs push
at once (`just overlap-release`), so the main set's uploads end first and its finalise
can start under the scholarly set's last uploads
(docs/journal/2026-10-26-finalise-before-uploads-end.md).

- **The main run holds its turn** (`hold_turn(True)`) once the split has numbered every
  chunk, from the start of a group's close for as long as its pool closes groups back to
  back, and lets it go (`hold_turn(False)`) once it has no group to close, and at its end.
  Not while the split runs: there a group's merge, on the cores the split leaves, could
  take minutes, and the link would wait through each (the journal's model: −24 min at 7
  MB/s held from the split's end, whatever the merges take; held from the start, −38 min
  at a 1-minute merge but +67 min at a 5-minute one).
- **The scholarly run waits for it** (`wait_turn`) before each group's push, and pushes
  once the main run holds no turn. A push already going on is not stopped: a main group
  closing during a scholarly push shares the link with it.

The turn is an flock on the main set's state dir's `push.turn`: no state, nothing
recorded. The kernel lets it go when the main run's process ends, however it ends, so a
main run killed mid-push never holds up the scholarly run. Only the order the existing
pushes start in changes: the same calls, the same files, the same ledgers.
WIKIDATA_FINALISE_EARLY=0 (config.FINALISE_EARLY): no turn is held or waited for.
"""

from __future__ import annotations

import fcntl
import os
import time
from pathlib import Path

from ..config import FINALISE_EARLY, OTHER_WORK_DIR, RELEASE, SCHOLAR, STATE_DIR

_held: int | None = None  # (the main run's open turn file, while it holds the turn)


def turn_path() -> Path | None:
    """The main set's turn file: a release's, with FINALISE_EARLY; else None."""
    if not (RELEASE and FINALISE_EARLY):
        return None
    return (OTHER_WORK_DIR / "state" if SCHOLAR else STATE_DIR) / "push.turn"


def hold_turn(on: bool) -> None:
    """The main run takes its turn (on) or lets it go; nothing for the scholarly set's or
    without a turn file. Idempotent."""
    global _held
    path = turn_path()
    if path is None or SCHOLAR:
        return
    if on and _held is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o644)
        fcntl.flock(fd, fcntl.LOCK_EX)  # (a scholarly wait takes it for an instant at most)
        _held = fd
    elif not on and _held is not None:
        fd, _held = _held, None
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def wait_turn(name: str) -> float:
    """The scholarly run, before group `name`'s push: wait while the main run holds its
    turn. The seconds waited; nothing for the main set's, or with no main run."""
    path = turn_path()
    if path is None or not SCHOLAR or not path.exists():
        return 0.0
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
            return 0.0
        except BlockingIOError:
            pass
        print(f"[push] {name}: waiting for the main set's pushes", flush=True)
        t = time.monotonic()
        fcntl.flock(fd, fcntl.LOCK_SH)
        waited = time.monotonic() - t
        print(f"[push] {name}: pushing, after {waited:.0f} s", flush=True)
        return waited
    finally:
        os.close(fd)  # (and with it the shared lock, at once)
