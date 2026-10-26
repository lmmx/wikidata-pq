"""A/B of the benchmark workload (see README.md): each round runs every arm once, the
commit before first (--order forward) or the commit after first (reverse); the workflow
runs each round as a forward and a reverse job, a pair. primer.py runs before each run. Round 1 keeps each arm's outputs (KEEP=1), and
parity.py checks every arm's against the first's.

Per run, besides pipeline.py's own figures: the peak rise in the used space of the disk
the run writes to (sampled every 0.2 s), the bytes that disk's device wrote
(/proc/diskstats), and the temporary directories (/tmp/tmp*) made during the run and left
after it. Every 30 s a run's stage and the lowest memory available go to the job's log
(Heartbeat), for a runner lost mid-run. Writes <out>/results.json (every run's figures, the machine, the packages) and
<out>/summary.md, and exits 1 if parity fails, or the split from the gz dump's own parity
(pipeline.split_gz), or the misfit chunk's (misfit.py), or the new field's (evolve.py).

Each arm runs in its checkout's own environment (SRC/.venv) if it has one, else in the
Python running this script. With --cache, the commit before is not run again when the
cache holds a measurement of the same pipeline (cache_key) made on the same CPU model
with the same Polars: its figures and its outputs' fingerprint are taken from there.
With --write-cache, the commits measured here are added to the cache (never replacing an
entry).

Usage: ab.py --data DIR --out DIR --arm before=SRC --arm after=SRC [--rounds 1]
             [--cache DIR [--write-cache]]
"""

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from combine import render, rounds_of
from overlap import release_wall
from version import VERSION
from parity import compare

HERE = Path(__file__).resolve().parent
# The workloads (tiers). quick: 4 chunk files, one chunk each, 4 workers (the runner's
# cores), one group: a fast regression signal for every pull request. full: the
# configuration of the campaign's 66-chunk A/Bs, the 11 files cycled 6 times, 6 workers
# taking 8 chunks each, groups of 22 merged while chunks are still processing.
ALL = ["c0", "r10", "r30", "r50", "r70", "r90", "s10", "s30", "s50", "s70", "s90"]
TIERS = {
    "quick": {"files": ["c0", "r50", "s30", "s90"], "chunks": 4, "workers": 4, "per": 8, "group": 4},
    "full": {"files": ALL, "chunks": 66, "workers": 6, "per": 8, "group": 22},
}
TIER = TIERS["quick"]  # set from --tier in main
PACKAGES = ["polars", "polars-genson", "pq-glue", "pyarrow", "orjson"]


def device_sectors_written(path: Path) -> int | None:
    """Sectors written by the block device holding `path`, None if it has none."""
    dev = os.stat(path).st_dev
    for line in open("/proc/diskstats"):
        f = line.split()
        if (int(f[0]), int(f[1])) == (os.major(dev), os.minor(dev)):
            return int(f[9])
    return None


def python(checkout: Path) -> str:
    """The arm's interpreter: its checkout's environment, else this script's."""
    venv = checkout / ".venv" / "bin" / "python"
    return str(venv) if venv.exists() else sys.executable


def packages(py: str) -> dict:
    code = ("import json, importlib.metadata as m\n"
            f"def v(p):\n try: return m.version(p)\n except m.PackageNotFoundError: return None\n"
            f"print(json.dumps({{p: v(p) for p in {PACKAGES!r}}}))")
    return json.loads(subprocess.run([py, "-c", code], capture_output=True, text=True, check=True).stdout)


def memory_available_mb() -> int:
    return next(int(l.split()[1]) for l in open("/proc/meminfo") if l.startswith("MemAvailable:")) >> 10


# How often a run says where it is in the job's log, and the lowest memory available
HEARTBEAT = 30


class Heartbeat:
    """A run's progress in the job's log (this process's stdout), every HEARTBEAT seconds:
    its stage (the last marker pipeline.py wrote to its phases log) and the machine's
    lowest memory available since the line before. A runner that is lost mid-run (the
    workflow's lost.py reads its log) leaves these behind, where its results are lost."""

    def __init__(self, name: str, phases_log: Path):
        self.name, self.log, self.at, self.stage = name, phases_log, 0, "starting"
        self.t0 = self.last = time.monotonic()
        self.low = memory_available_mb()

    def tick(self) -> None:
        self.low = min(self.low, memory_available_mb())
        if time.monotonic() - self.last >= HEARTBEAT:
            self.say()

    def say(self) -> None:
        try:
            with open(self.log, "rb") as f:
                f.seek(self.at)
                new = f.read()
            self.at += new.rfind(b"\n") + 1  # (whole lines only)
            for line in new[:new.rfind(b"\n") + 1].splitlines():
                if line.startswith(b'{"marker"'):
                    self.stage = json.loads(line)["marker"]
        except (OSError, ValueError):
            pass
        print(f"[bench] {self.name} +{time.monotonic() - self.t0:.0f}s: {self.stage}; memory available at its "
              f"lowest {self.low / 1000:.1f} GB", flush=True)
        self.last, self.low = time.monotonic(), memory_available_mb()


def run_once(name: str, checkout: Path, files: list[str], out: Path, keep: bool) -> dict:
    # Warm: the input files and the arm's packages read from disk once, untimed, so no run
    # pays for a cold page cache (with a cached commit before, the commit after runs first)
    for f in files:
        Path(f).read_bytes()
    subprocess.run([python(checkout), "-c", "import polars, pyarrow.parquet, polars_genson"], check=True)
    subprocess.run([sys.executable, HERE / "primer.py", "3"], check=True)
    stop, peak = threading.Event(), [0]
    used0 = shutil.disk_usage(out).used
    beat = Heartbeat(name, out / f"{name}.phases.jsonl")

    def sample() -> None:
        while not stop.wait(0.2):
            peak[0] = max(peak[0], shutil.disk_usage(out).used - used0)
            beat.tick()

    tmp = Path(tempfile.gettempdir())
    tmp_before = {p.name for p in tmp.glob("tmp*")}
    os.sync()  # the previous run's buffered writes are counted before this run starts
    w0 = device_sectors_written(out)
    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    result = out / f"{name}.json"
    env = {**os.environ, **({"KEEP": "1"} if keep else {})}
    cmd = [python(checkout), HERE / "pipeline.py", checkout / "src", result,
           TIER["per"], TIER["workers"], TIER["group"], TIER["chunks"], *files]
    with open(out / f"{name}.log", "w") as log:
        subprocess.run(list(map(str, cmd)), check=True, env=env, stdout=log, stderr=subprocess.STDOUT)
    os.sync()
    w1 = device_sectors_written(out)
    stop.set()
    sampler.join()
    run = json.loads(result.read_text())
    run.update(
        disk_peak_mb=peak[0] >> 20,
        device_written_mb=None if w0 is None else (w1 - w0) * 512 >> 20,
        tmp_left=sorted({p.name for p in tmp.glob("tmp*")} - tmp_before),
    )
    # The split from the gz dump, after the timed run and outside its figures; its parity
    # also needs the manifests it wrote to be the bz2 split's
    gz = out / f"{name}.split-gz.json"
    subprocess.run(list(map(str, [python(checkout), HERE / "pipeline.py", "--split-gz", checkout / "src", gz,
                                  TIER["workers"], *files])), check=True, env=env)
    if split_gz := json.loads(gz.read_text()):
        if "gz_split_parity_ok" in split_gz:
            split_gz["gz_split_parity_ok"] &= split_gz.pop("gz_split_manifests_sha256") == run.get("split_manifests_sha256")
        run.update(split_gz)
    # The split and both sets' pipelines, overlapped and not, after the timed run and
    # outside its figures (bench 17.0.0, overlap.py)
    ov = out / f"{name}.overlap.json"
    subprocess.run(list(map(str, [python(checkout), HERE / "pipeline.py", "--overlap", checkout / "src", ov,
                                  TIER["workers"], *files])), check=True, env=env)
    run.update(json.loads(ov.read_text()))
    # A chunk whose claims misfit claims.avsc, the sample's first given one: the split's
    # work on it and the pipeline's process step on its part, after the timed run and
    # outside its figures (bench 17.3.0, misfit.py)
    mf = out / f"{name}.misfit.json"
    subprocess.run(list(map(str, [python(checkout), HERE / "pipeline.py", "--misfit", checkout / "src", mf, files[0]])),
                   check=True, env=env)
    run.update(json.loads(mf.read_text()))
    # A new claims field found mid-run, and the release schema widened for it: discovery,
    # the split following, a merge's widening and finalise's, after the timed run and
    # outside its figures (bench 17.5.0, evolve.py)
    ev = out / f"{name}.evolve.json"
    subprocess.run(list(map(str, [python(checkout), HERE / "pipeline.py", "--evolve", checkout / "src", ev, files[0]])),
                   check=True, env=env)
    run.update(json.loads(ev.read_text()))
    # Bench 10.0.0's ETA on the same run: its split on its own slice, in orjson's form
    if "orjson_split_eta" in run and "eta_10" in run:
        run["split_eta_10"] = run["orjson_split_eta"]
        run["eta_10"] += run["split_eta_10"] - run["split_eta"]
    # A release's wall time as its recipes overlap its stages, with the uploads (modelled
    # from the figures above: overlap.release_wall, bench 17.1.0)
    run.update(release_wall(run))
    result.write_text(json.dumps(run))
    return run


def machine() -> dict:
    cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), None)
    mem = next(int(l.split()[1]) for l in open("/proc/meminfo") if l.startswith("MemTotal"))
    return {
        "cpu": cpu, "cores": os.cpu_count(), "memory_gb": round(mem / 2**20, 1),
        "kernel": platform.release(), "python": platform.python_version(),
        "runner": {k: os.environ.get(k) for k in ("RUNNER_NAME", "ImageOS", "ImageVersion")},
    }


def git_sha(src: Path) -> str | None:
    r = subprocess.run(["git", "-C", src, "rev-parse", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() or None


def git(checkout: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", checkout, *args], capture_output=True, text=True, check=True).stdout.strip()


def cache_key(checkout: Path, data: Path) -> str:
    """What a commit's measurement depends on: its pipeline (src/, pyproject.toml,
    uv.lock), the harness, the data and the workload. A docs-only commit has its
    parent's key; a harness change makes every key new."""
    h = hashlib.sha256()
    for path in ("src", "pyproject.toml", "uv.lock"):
        h.update(git(checkout, "rev-parse", f"HEAD:{path}").encode())
    for f in sorted(HERE.glob("*.py")):
        if f.name != "record.py":
            h.update(f.read_bytes())
    h.update((data / "SHA256SUMS").read_bytes())
    h.update(json.dumps(TIER, sort_keys=True).encode())
    h.update(VERSION.encode())
    return h.hexdigest()


def fingerprint(py: str, keep: Path, out: Path) -> dict:
    subprocess.run([py, HERE / "parity.py", "fingerprint", keep, out], check=True)
    return json.loads(out.read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arm", action="append", required=True, help="NAME=SRC: the commit before, then the commit after")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--tier", choices=sorted(TIERS), default="quick")
    ap.add_argument("--seed", type=int, default=0, help="this job's round number (one per parallel pair of jobs)")
    ap.add_argument("--order", choices=["forward", "reverse"], default="forward",
                    help="forward: the commit before runs first; reverse: the commit after does")
    ap.add_argument("--on-round", help="a shell command run after each round but the last, the results so far in --out")
    ap.add_argument("--cache", type=Path, help="measurements of earlier commits, by cache key")
    ap.add_argument("--write-cache", action="store_true", help="add this run's measurements to --cache")
    args = ap.parse_args()
    arms = {n: Path(s).resolve() for n, s in (a.split("=", 1) for a in args.arm)}
    (before, a), (after, b) = arms.items()
    global TIER
    TIER = TIERS[args.tier]
    files = [str((args.data / f"{f}.parquet").resolve()) for f in TIER["files"]]
    # The sample's size, for throughput and release-scale figures: pipeline.py cycles the
    # files to make TIER["chunks"] chunks
    code = "import sys, pyarrow.parquet as pq; print(*(pq.ParquetFile(f).metadata.num_rows for f in sys.argv[1:]))"
    rows = list(map(int, subprocess.run([python(arms[after]), "-c", code, *files], capture_output=True, text=True,
                                        check=True).stdout.split()))
    cycle = [i % len(files) for i in range(TIER["chunks"])]
    entities = sum(rows[i] for i in cycle)
    source_mb = round(sum(os.path.getsize(files[i]) for i in cycle) / 1e6, 1)
    args.out.mkdir(parents=True, exist_ok=True)
    cpu = machine()["cpu"]
    pk = {n: packages(python(c)) for n, c in arms.items()}
    keys = {n: cache_key(c, args.data) for n, c in arms.items()}

    # The commit before, measured earlier on the same CPU model with the same Polars (so
    # its fingerprint compares), is not run again
    cached = None
    if args.cache and (entry := args.cache / f"{keys[before]}.json").exists():
        e = json.loads(entry.read_text())
        if e["cpu"] == cpu and e["fingerprint"]["polars"] == pk[after]["polars"]:
            cached = e
    print(f"{before}: " + (f"cached measurement of {cached['commit'][:9]}" if cached else "measured in this run"), flush=True)

    runs = [{**r, "arm": before} for r in cached["runs"]] if cached else []
    report, fp = None, {}

    def results(done: int) -> dict:
        return {
            "workload": {"tier": args.tier, "chunks": TIER["chunks"], "files": len(files), "workers": TIER["workers"],
                         "chunks_per_worker": TIER["per"], "group": TIER["group"],
                         "entities": entities, "source_mb": source_mb},
            "bench": VERSION, "round": args.seed, "order": args.order, "rounds": args.rounds, "rounds_done": done,
            "arms": {n: {"src": str(c), "sha": git_sha(c), "packages": pk[n], "cache_key": keys[n]} for n, c in arms.items()},
            "before_cached": bool(cached),
            "machine": machine(),
            "data_sha256sums": hashlib.sha256((args.data / "SHA256SUMS").read_bytes()).hexdigest(),
            "runs": runs, "parity": {after: report},
            # (and each run's split made exactly the sample's chunks: pipeline.split_stage)
            "parity_ok": report["ok"] and all(r.get("split_parity_ok", True) for r in runs),
            # (the split from gz's own: a commit without it passes; and the split on bench
            # 10.0.0's slice, its lines in orjson's form)
            "gz_split_parity_ok": all(r.get("gz_split_parity_ok", True) and r.get("orjson_split_parity_ok", True)
                                      for r in runs),
            # (and the misfit chunk's claims as genson's inference gives them: misfit.py)
            "misfit_parity_ok": all(r.get("misfit_parity_ok", True) for r in runs),
            # (and the new field's: evolve.py; a commit without a release schema halts at it)
            "evolve_parity_ok": all(r.get("evolve_parity_ok", True) for r in runs),
        }

    for rnd in range(1, args.rounds + 1):
        # forward: the commit before first; reverse: the commit after first. A round is a
        # forward job and a reverse job on two runners, so what running first or second
        # does falls on each commit once and cancels in the pair (combine.paired)
        order = [after] if cached else [before, after] if args.order == "forward" else [after, before]
        for name in order:
            print(f"round {rnd} ({args.order}): {name}", flush=True)
            r = run_once(f"{name}_r{rnd}", arms[name], files, args.out, keep=rnd == 1)
            r["close_cpu"] = round(sum(c["cpu_process"] for c in r["closes"]), 1)
            print(f"  wall {r['wall']:.1f} s, CPU {r['cpu_total']:.0f} s, RSS peak {r['peak_sum_rss_mb']} MB, "
                  f"device wrote {r['device_written_mb']} MB", flush=True)
            runs.append({"arm": name, "round": rnd, **{k: v for k, v in r.items() if k not in ("chunks", "samples")}})

        if rnd == 1:  # parity, from round 1's kept outputs
            fp = {n: fingerprint(python(c), args.out / f"{n}_r1.keep", args.out / f"{n}.fingerprint.json")
                  for n, c in arms.items() if not (cached and n == before)}
            fp_before = cached["fingerprint"] if cached else fp[before]
            fp_after = fp[after]
            if fp_after["polars"] != fp_before["polars"]:  # row hashes compare within one Polars version
                fp_after = fingerprint(python(a), args.out / f"{after}_r1.keep",
                                       args.out / f"{after}.fingerprint-before-polars.json")
            report = compare(fp_before, fp_after)
            print(json.dumps(report), flush=True)
            for n in arms:
                shutil.rmtree(args.out / f"{n}_r1.keep", ignore_errors=True)

        # The results so far, after every round: --on-round posts them as they come in
        res = results(rnd)
        (args.out / "results.json").write_text(json.dumps(res, indent=1))
        (args.out / "summary.md").write_text(render(rounds_of(res), args.rounds))
        if not report["ok"]:
            break  # output that differs makes the timing moot
        if args.on_round and rnd < args.rounds:
            subprocess.run(args.on_round, shell=True)

    if args.write_cache and report["ok"]:
        args.cache.mkdir(parents=True, exist_ok=True)
        for n, c in arms.items():
            if n in fp and not (args.cache / f"{keys[n]}.json").exists():  # entries are never replaced
                (args.cache / f"{keys[n]}.json").write_text(json.dumps({
                    "commit": git_sha(c), "key": keys[n], "cpu": cpu, "packages": pk[n],
                    "runs": [{k: v for k, v in r.items() if k != "arm"} for r in runs if r["arm"] == n],
                    "fingerprint": fp[n],
                }))
    print(render(rounds_of(res), args.rounds))
    ok = res["parity_ok"] and res["gz_split_parity_ok"] and res["misfit_parity_ok"] and res["evolve_parity_ok"]
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
