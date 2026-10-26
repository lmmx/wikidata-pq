"""A release's finalise runs overlapped across their processes (WIKIDATA_FINALISE_CROSS_RUN=1,
off by default; docs/journal/2026-10-26-cross-run-overlap.md).

A release is finalised in three runs, each its own process: the main set, the scholarly
set, the main set again (claims_labels needs both sets' labels sorted). Each run's
committer (sort_by_id.Committer) commits its tables while it writes the next, but the run
ends only once its last commit is in, so the next run started from cold: the CPU idle
under the last commits, then the link idle under the next run's downloads and first sort.
With this, the next run starts once the one before has written its last table, while its
commits are still going on:

- **The orchestrator** (`run_release`, the `finalise-release-runs` command, run by
  `just overlap-release` and `just release`) starts the runs in order, each once the run
  before has written its last table (its `finalise.written`) or ended. Only one run
  writes at a time, so the CPU and memory are as before; the runs' commits share the link
  in turn. The scholarly run starts only once the scholarly set's own run has ended
  (`after`: `overlap-release` passes its exit code), as finalise's guards require.
- **Each run, under the flag** (main.finalise):
  - holds its set's `finalise.lock` (`hold`) for as long as it runs, so a set's second
    run waits for its first to end (they share a ledger), and the other set's run can
    tell it is still going (`alive`);
  - writes its `finalise.written` (`mark_written`) once it has handed its last table to
    its committer;
  - publishes the bytes it has written and not yet committed (`finalise.pending`,
    `Committer`), and starts no table to write while the other set's run publishes more
    than WIKIDATA_FINALISE_CROSS_RUN_GB (the disk the next run's sort shares with the
    last run's sorted files waiting for the link);
  - before claims_labels' build, waits for the other set's labels to be done while the
    other set's run is still going (`wait_labels`), instead of stopping at once; with no
    such run, it stops as before.

Nothing else changes: each run's tables, commits, verification and ledgers are as they
were, and claims_labels still reads both sets' sorted labels. A killed run leaves its
ledgers as before, and running the recipe again resumes each run where it stopped.
"""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

from .config import FINALISE_CROSS_RUN_BYTES, RELEASES_DIR, RELEASE, Table, set_name
from .sort_by_id import STAGES, last_stage
from .sort_by_id import Committer as _Committer

LOCK, WRITTEN, PENDING = "finalise.lock", "finalise.written", "finalise.pending"
POLL = 1.0  # (seconds between looks at the other set's run)
# `just release`'s finalise runs, in order: (set, scholar)
RUNS = (("main", False), ("scholar", True), ("main", False))

_held: dict[Path, int] = {}  # (this process's set lock, while it runs)


def state_dir(scholar: bool, root: Path = Path(".")) -> Path:
    return root / RELEASES_DIR / set_name(scholar) / "state"


def hold(state: Path) -> None:
    """Take the set's finalise lock for as long as this process runs (the kernel lets it
    go with the process, however it ends), waiting for a run of the same set to end."""
    if state in _held:
        return
    state.mkdir(parents=True, exist_ok=True)
    fd = os.open(state / LOCK, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"[finalise] waiting for this set's other finalise run to end ({state / LOCK})", flush=True)
        fcntl.flock(fd, fcntl.LOCK_EX)
    _held[state] = fd


def alive(state: Path) -> bool:
    """Whether a finalise run of that set holds its lock (is running)."""
    try:
        fd = os.open(state / LOCK, os.O_RDONLY | os.O_CLOEXEC)
    except FileNotFoundError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        return False
    except BlockingIOError:
        return True
    finally:
        os.close(fd)  # (and with it the shared lock, at once)


def mark_written(state: Path) -> None:
    """This run has handed its last table to its committer: the next run may start."""
    (state / WRITTEN).write_text(f"{os.getpid()}\n")


def _write_pending(state: Path, nbytes: int) -> None:
    tmp = state / (PENDING + ".tmp")
    tmp.write_text(json.dumps({"bytes": nbytes, "pid": os.getpid()}))
    tmp.replace(state / PENDING)


def other_pending(state: Path) -> int:
    """The bytes the set's running finalise has written and not yet committed (0 with
    none running)."""
    if not alive(state):
        return 0
    try:
        return int(json.loads((state / PENDING).read_text())["bytes"])
    except (FileNotFoundError, ValueError, KeyError):
        return 0


def wait_labels(other: Path) -> str | None:
    """Wait while the other set's labels are not done and its finalise run is going: the
    labels' last sort stage then."""
    said = False
    while (stage := last_stage(other, Table.LABEL)) != "done" and alive(other):
        if not said:
            print(f"[finalise] claims_labels waits for the other set's labels ({stage or 'not started'}), "
                  "its finalise still running", flush=True)
            said = True
        time.sleep(POLL)
    return stage


class Committer(_Committer):
    """sort_by_id.Committer, its bytes written and not yet committed published to the
    set's `finalise.pending` as they change, and a table to write not started while the
    other set's finalise has more than `other_limit` of its own (`wait_for_other`)."""

    def __init__(self, state: Path, other: Path, other_limit: int = FINALISE_CROSS_RUN_BYTES, **kw):
        super().__init__(state, **kw)
        self.other, self.other_limit = other, other_limit
        _write_pending(state, 0)
        self._publisher = threading.Thread(target=self._publish, name="finalise-pending", daemon=True)
        self._publisher.start()

    def _publish(self) -> None:
        last = 0
        while True:
            with self._cond:
                if sum(self._pending.values()) == last and not self._stop:
                    self._cond.wait(timeout=POLL)
                now, stop = sum(self._pending.values()), self._stop
            if now != last:
                _write_pending(self.state_dir, now)
                last = now
            if stop:
                return

    def wait_for_other(self, table: Table) -> None:
        """Before the table is compacted and sorted: wait while the other set's finalise
        has more than `other_limit` written and not yet committed, unless the table is
        written already (nothing to write)."""
        stage = last_stage(self.state_dir, table)
        if stage is not None and STAGES.index(stage) >= STAGES.index("written"):
            return
        said = False
        while (n := other_pending(self.other)) > self.other_limit:
            if not said:
                print(f"[finalise] the other set's finalise has {n / 1e9:.1f} GB written and not yet committed "
                      f"(over {self.other_limit / 1e9:g} GB): {table} waits", flush=True)
                said = True
            self.reap()
            time.sleep(POLL)


def run_release(start: Callable[[int, str, bool], subprocess.Popen], root: Path = Path("."),
                after: Path | None = None, poll: float = POLL) -> int:
    """Run `RUNS` in order, each started (`start(index, set, scholar)`, a process) once the
    run before has written its last table or ended. The scholarly run waits for `after`
    (the scholarly set's run's exit code, written there), when given. A run that fails
    before the next is started stops the rest from starting; every run started is waited
    for. 0 when every run ended with 0."""
    procs: list[tuple[str, subprocess.Popen]] = []
    failed = False
    for i, (set_, scholar) in enumerate(RUNS):
        state = state_dir(scholar, root)
        if scholar and after is not None:
            while not after.exists() and not any(p.poll() for _, p in procs):
                time.sleep(poll)
            if not after.exists() or after.read_text().strip() != "0":
                print("[finalise] the scholarly set's run did not end with 0: its finalise not started",
                      flush=True)
                failed = True
                break
        state.mkdir(parents=True, exist_ok=True)
        (state / WRITTEN).unlink(missing_ok=True)
        print(f"[finalise] run {i + 1} of {len(RUNS)}: the {set_} set", flush=True)
        p = start(i, set_, scholar)
        procs.append((set_, p))
        if i == len(RUNS) - 1:
            break
        while p.poll() is None and not (state / WRITTEN).exists():
            time.sleep(poll)
        if p.poll():
            print(f"[finalise] run {i + 1} failed ({p.returncode}): the next not started", flush=True)
            failed = True
            break
    for i, (_, p) in enumerate(procs):
        if p.wait():
            print(f"[finalise] run {i + 1} failed ({p.returncode})", flush=True)
            failed = True
    return 1 if failed else 0


def run() -> None:
    """`finalise-release-runs [--after FILE]`: a release's three finalise runs, each
    `finalise-wikidata` with WIKIDATA_FINALISE_CROSS_RUN=1, overlapped (run_release)."""
    if not RELEASE:
        raise SystemExit("finalise-release-runs needs WIKIDATA_RELEASE")
    args = sys.argv[1:]
    after = Path(args[args.index("--after") + 1]) if "--after" in args else None

    def start(i: int, set_: str, scholar: bool) -> subprocess.Popen:
        env = {**os.environ, "WIKIDATA_SCHOLAR": "1" if scholar else "", "WIKIDATA_FINALISE_CROSS_RUN": "1"}
        return subprocess.Popen(["finalise-wikidata"], env=env)

    sys.exit(run_release(start, after=after))
