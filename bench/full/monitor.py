"""The machine sampled through a run (run.py): a line a sample to timeline.jsonl.

Each sample (every `interval` seconds) holds:

- `cpu`: /proc/stat's aggregate jiffies (user, nice, system, idle, iowait, irq, softirq,
  steal), so any window's split between them can be taken (report.py: per stage, which
  tells a stage bound by the CPU from one waiting on the disk);
- `load`: the 1-minute load average (other work on the machine shows here);
- `mem_avail`: MemAvailable, the memory left to the machine;
- `rss`: the summed RSS of every live process of each stage's tree (each registered
  root pid and its descendants at any depth), `rss_total` of the run's whole tree, and
  `rss_max_proc` its largest process's (RSS counts shared pages in each process);
- `free`: the work file system's free bytes (statvfs), so the disk the run takes is the
  fall from its start (on a shared machine, other writers too);
- `disk`: the work file system's device's sectors read and written (/proc/diskstats);
- `net`: bytes received and sent on every interface but lo (/proc/net/dev);
- `watch`: the size of each watched file (a streaming download's .part, then its dump);
- `lines`: the lines of each counted file so far (the split's manifest, a line a chunk
  written; each set's, a line a part numbered for its run), read as they are appended;
- every `du_every` seconds or so (longer if a walk is slow), `du`: the bytes the work
  tree holds, each inode once (a file linked into the stand-in Hub counted once).
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path


def _children(pid: int) -> list[int]:
    out = []
    try:
        for task in os.scandir(f"/proc/{pid}/task"):
            try:
                with open(f"{task.path}/children") as f:
                    out += [int(c) for c in f.read().split()]
            except OSError:
                pass
    except OSError:
        pass
    return out


def _rss_kb(pid: int) -> int:
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return 0


def tree(pid: int) -> list[int]:
    out, todo = [], [pid]
    while todo:
        p = todo.pop()
        out.append(p)
        todo += _children(p)
    return out


def _cpu() -> list[int]:
    with open("/proc/stat") as f:
        return [int(x) for x in f.readline().split()[1:9]]


def _meminfo() -> dict[str, int]:
    out = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":", 1)
            out[k] = int(v.split()[0]) * 1024
    return out


def device_of(path: Path) -> str | None:
    """The /proc/diskstats name of the device holding `path` (None if not found)."""
    st = os.stat(path)
    major, minor = os.major(st.st_dev), os.minor(st.st_dev)
    try:
        with open("/proc/diskstats") as f:
            for line in f:
                p = line.split()
                if int(p[0]) == major and int(p[1]) == minor:
                    return p[2]
    except OSError:
        pass
    return None


def _diskstats(dev: str | None) -> list[int] | None:
    if not dev:
        return None
    with open("/proc/diskstats") as f:
        for line in f:
            p = line.split()
            if p[2] == dev:  # (sectors read, sectors written: 512 bytes each; ms doing I/O)
                return [int(p[5]) * 512, int(p[9]) * 512, int(p[12])]
    return None


def _net() -> list[int]:
    rx = tx = 0
    with open("/proc/net/dev") as f:
        for line in f.readlines()[2:]:
            name, rest = line.split(":", 1)
            if name.strip() != "lo":
                v = rest.split()
                rx, tx = rx + int(v[0]), tx + int(v[8])
    return [rx, tx]


def du(root: Path) -> int:
    """The bytes allocated under `root`, each inode once."""
    seen, total, todo = set(), 0, [str(root)]
    while todo:
        d = todo.pop()
        try:
            it = os.scandir(d)
        except OSError:
            continue
        with it:
            for e in it:
                try:
                    if e.is_dir(follow_symlinks=False):
                        todo.append(e.path)
                    elif e.is_file(follow_symlinks=False):
                        st = e.stat(follow_symlinks=False)
                        if st.st_ino not in seen:
                            seen.add(st.st_ino)
                            total += st.st_blocks * 512
                except OSError:
                    pass
    return total


class Monitor(threading.Thread):
    def __init__(self, work: Path, out: Path, interval: float = 1.0, du_every: float = 15.0):
        super().__init__(daemon=True)
        self.work, self.out, self.interval, self.du_every = work, out, interval, du_every
        self.roots: dict[str, int] = {}
        self.watch: dict[str, Path] = {}
        self.lines: dict[str, Path] = {}  # (files whose lines are counted as they grow)
        self._counted: dict[str, list[int]] = {}  # (name: [bytes read, lines])
        self.dev = device_of(work)
        self._stop = threading.Event()
        self._lock = threading.Lock()

    def register(self, name: str, pid: int) -> None:
        with self._lock:
            self.roots[name] = pid

    def unregister(self, name: str) -> None:
        with self._lock:
            self.roots.pop(name, None)

    def _count(self, name: str, path: Path) -> int:
        """`path`'s lines, read from where the last count stopped (a manifest only grows,
        or is replaced: then counted again)."""
        done = self._counted.setdefault(name, [0, 0])
        try:
            with open(path, "rb") as f:
                if f.seek(0, os.SEEK_END) < done[0]:
                    done[:] = [0, 0]
                f.seek(done[0])
                data = f.read()
        except OSError:
            return done[1]
        end = data.rfind(b"\n") + 1  # (whole lines only)
        done[0] += end
        done[1] += data.count(b"\n", 0, end)
        return done[1]

    def sample(self, du_bytes: int | None = None) -> dict:
        with self._lock:
            roots = dict(self.roots)
        rss, every, one = {}, set(), {}
        for name, pid in roots.items():
            pids = tree(pid)
            every.update(pids)
            for p in pids:
                one[p] = _rss_kb(p)
            rss[name] = sum(one[p] for p in pids) * 1024
        mem = _meminfo()
        st = os.statvfs(self.work)
        s = {"t": time.time(), "cpu": _cpu(), "load": os.getloadavg()[0], "mem_avail": mem.get("MemAvailable"), "rss": rss,
             "rss_total": sum(one[p] for p in every) * 1024, "rss_max_proc": max(one.values(), default=0) * 1024,
             "free": st.f_bavail * st.f_frsize,
             "disk": _diskstats(self.dev), "net": _net(),
             "watch": {k: (p.stat().st_size if p.exists() else None) for k, p in self.watch.items()},
             "lines": {k: self._count(k, p) for k, p in self.lines.items()}}
        if du_bytes is not None:
            s["du"] = du_bytes
        return s

    def run(self) -> None:
        next_du = 0.0
        with open(self.out, "a", buffering=1) as f:
            while not self._stop.is_set():
                t = time.time()
                du_bytes = None
                if t >= next_du:
                    du_bytes = du(self.work)
                    took = time.time() - t
                    next_du = time.time() + max(self.du_every, 20 * took)
                f.write(json.dumps(self.sample(du_bytes)) + "\n")
                self._stop.wait(max(0.05, self.interval - (time.time() - t)))
            f.write(json.dumps(self.sample(du(self.work))) + "\n")

    def stop(self) -> None:
        self._stop.set()
        self.join()
