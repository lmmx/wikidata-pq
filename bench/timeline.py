"""Where finalise's wall time goes, beside its CPU time: which cores are busy over time,
and what each stage and each worker runs while they are (or are not).

Measurement only. `sample` runs in a process of its own (pipeline.py starts it before
finalise's first run and stops it after the second, then reaps it, so its CPU time is in
none of the run's figures), every INTERVAL seconds writing a JSON line of cumulative
counters: the machine's CPU time from /proc/stat (busy, iowait and idle, in seconds), the
CPU time of the pipeline's process tree (each live process's own and its reaped
children's), each live process's own CPU time by pid, and the run's disk device's sectors
read and written and its milliseconds busy (/proc/diskstats). Reading a sample costs about
a millisecond.

`summarise` lays the samples beside phases.py's calls (each with its process, start, end
and depth): finalise's stages as the pipeline's process runs them (each table's compaction,
the copy's check, the sort's write by table, and for a key sorted through buckets its
bucketing, bucket sorts and packing, the in-memory keys after them), each with its wall
time, the cores busy (the machine's CPU time over the stage's wall time), the cores
waiting on I/O, the device's busy share, the core-seconds left idle (cores × wall less the
CPU time), and its jobs: how many, on how many worker processes, the mean and peak running
at once, the longest, and the time the last job ran alone (a straggler's tail)."""

import json
import os
import signal
import sys
import time
from pathlib import Path

INTERVAL = 0.5
TICK = os.sysconf("SC_CLK_TCK")

# A worker's job: the outermost listed call of a process other than the pipeline's
JOBS = {"sort_by_id._bucket_source", "sort_by_id._sort_bucket", "sort_by_id._pack_file",
        "sort_by_id.sort_in_memory", "compact._write_run"}
# The bucket path's steps, within a write of the sort (in the pipeline's process)
STEPS = {"sort_by_id.bucket_key": "bucketing", "sort_by_id.sort_buckets": "bucket sorts",
         "sort_by_id.pack_key": "packing"}


def _stat() -> dict:
    f = list(map(int, open("/proc/stat").readline().split()[1:]))
    user, nice, system, idle, iowait, irq, softirq, steal = (f + [0] * 8)[:8]
    return {"busy": (user + nice + system + irq + softirq + steal) / TICK, "iowait": iowait / TICK,
            "idle": idle / TICK}


def _proc(pid: int) -> tuple[float, float]:
    """A process's own CPU time and its reaped children's, in seconds."""
    f = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
    return (int(f[11]) + int(f[12])) / TICK, (int(f[13]) + int(f[14])) / TICK


def _tree(root: int) -> list[int]:
    out, todo = [], [root]
    while todo:
        pid = todo.pop()
        out.append(pid)
        try:
            for task in os.listdir(f"/proc/{pid}/task"):
                todo += map(int, open(f"/proc/{pid}/task/{task}/children").read().split())
        except OSError:
            pass
    return out


def _disk(dev: tuple[int, int]) -> list[int] | None:
    for line in open("/proc/diskstats"):
        f = line.split()
        if (int(f[0]), int(f[1])) == dev:
            return [int(f[5]), int(f[9]), int(f[12])]  # sectors read, written; ms busy
    return None


def sample(root: int, out: Path, where: Path) -> None:
    """Sample until SIGTERM or the root process ends."""
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    st = os.stat(where).st_dev
    dev = (os.major(st), os.minor(st))
    with open(out, "w", buffering=1) as f:
        while True:  # (a last sample once stopped)
            t = time.time()
            procs, tree = {}, 0.0
            for pid in _tree(root):
                try:
                    own, children = _proc(pid)
                except (OSError, IndexError, ValueError):
                    continue
                procs[pid] = own
                tree += own + children
            if root not in procs:
                break
            f.write(json.dumps({"t": t, **_stat(), "tree": round(tree, 2),
                                "procs": {p: round(c, 2) for p, c in procs.items()}, "disk": _disk(dev)}) + "\n")
            if stop:
                break
            time.sleep(max(0.0, INTERVAL - (time.time() - t)))


def _at(samples: list[dict], key, t: float) -> float:
    """A cumulative counter at time t, interpolated between the samples around it."""
    if t <= samples[0]["t"]:
        return key(samples[0])
    for a, b in zip(samples, samples[1:]):
        if t <= b["t"]:
            w = (t - a["t"]) / ((b["t"] - a["t"]) or 1)
            return key(a) + w * (key(b) - key(a))
    return key(samples[-1])


def _stages(calls: list[dict], main: int, t0: float, t1: float) -> list[dict]:
    """finalise's stages, consecutive, as the pipeline's process ran them between t0 and t1."""
    top = sorted((c for c in calls if c.get("pid") == main and c["depth"] == 0 and t0 <= c["t0"] and c["t"] <= t1),
                 key=lambda c: c["t0"])
    out = []
    for c in top:
        name = f"{c['fn'].split('.', 1)[1]} {c.get('of', [''])[0]}".strip()
        if c["fn"] != "sort_by_id.write_table":
            out.append({"stage": name, "start": c["t0"], "end": c["t"]})
            continue
        steps = sorted((s for s in calls if s.get("pid") == main and s["fn"] in STEPS and c["t0"] <= s["t0"]
                        and s["t"] <= c["t"]), key=lambda s: s["t0"])
        at = c["t0"]
        for s in steps:
            if s["t0"] > at + 0.05:
                out.append({"stage": f"{name}: between steps", "start": at, "end": s["t0"]})
            out.append({"stage": f"{name}: {'/'.join(s.get('of', []))} {STEPS[s['fn']]}", "start": s["t0"], "end": s["t"]})
            at = s["t"]
        out.append({"stage": f"{name}: in-memory keys" if steps else name, "start": at, "end": c["t"]})
    return out


def _jobs(jobs: list[dict], a: float, b: float) -> dict:
    """The worker jobs run in [a, b]: how many, on how many processes, running at once
    (mean and peak), the longest, and how long the last ran alone."""
    inside = [j for j in jobs if a <= j["t0"] and j["t"] <= b + 0.05]
    if not inside:
        return {}
    events = sorted([(j["t0"], 1) for j in inside] + [(j["t"], -1) for j in inside])
    running, peak, area, prev = 0, 0, 0.0, events[0][0]
    for t, d in events:
        area += running * (t - prev)
        running, prev = running + d, t
        peak = max(peak, running)
    ends = sorted(j["t"] for j in inside)
    span = ends[-1] - min(j["t0"] for j in inside)
    return {"jobs": len(inside), "workers": len({j["pid"] for j in inside}), "peak_running": peak,
            "mean_running": round(area / span, 2) if span > 0 else None,
            "longest": round(max(j["wall"] for j in inside), 2),
            "alone_at_end": round(ends[-1] - ends[-2], 2) if len(ends) > 1 else round(span, 2)}


def summarise(phases_log: Path, samples_log: Path, main: int, spans: dict[str, tuple[str, str]],
              cores: int) -> dict:
    """Each span's (start and end markers) stages, each with its wall time, the cores busy,
    idle and waiting on I/O over it, the device's busy share and its worker jobs."""
    rows = [json.loads(line) for line in open(phases_log)]
    marks = {r["marker"]: r["t"] for r in rows if "marker" in r}
    calls = [r for r in rows if "fn" in r and "t0" in r]
    samples = [json.loads(line) for line in open(samples_log)] if samples_log.exists() else []
    if len(samples) < 2:
        return {}
    jobs = [c for c in calls if c["pid"] != main and c["depth"] == 0 and c["fn"] in JOBS]
    disk = (lambda s: (s["disk"] or [0, 0, 0])[2] / 1000)

    def over(a: float, b: float) -> dict:
        wall = max(b - a, 1e-9)
        busy = _at(samples, lambda s: s["busy"], b) - _at(samples, lambda s: s["busy"], a)
        iowait = _at(samples, lambda s: s["iowait"], b) - _at(samples, lambda s: s["iowait"], a)
        tree = _at(samples, lambda s: s["tree"], b) - _at(samples, lambda s: s["tree"], a)
        return {"wall": round(b - a, 2), "cpu": round(busy, 1), "cores": round(busy / wall, 2),
                "tree_cores": round(tree / wall, 2), "iowait_cores": round(iowait / wall, 2),
                "idle_core_s": round(max(cores * (b - a) - busy, 0.0), 1),
                "disk_busy": round((_at(samples, disk, b) - _at(samples, disk, a)) / wall, 2)}

    out = {}
    for span, (m0, m1) in spans.items():
        if m0 not in marks or m1 not in marks or samples[0]["t"] > marks[m0] or samples[-1]["t"] < marks[m1] - 1:
            continue
        a, b = marks[m0], marks[m1]
        stages = []
        for s in _stages(calls, main, a, b):
            stages.append({"stage": s["stage"], "start": round(s["start"] - a, 2), "end": round(s["end"] - a, 2),
                           **over(s["start"], s["end"]), **_jobs(jobs, s["start"], s["end"])})
        covered = sum(s["wall"] for s in stages)
        out[span] = {**over(a, b), "between_stages": round(b - a - covered, 2), "stages": stages}
    return out


if __name__ == "__main__":
    sample(int(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
