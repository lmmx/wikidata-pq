"""Interleaved A/B of the benchmark workload (see README.md): each round runs every arm
once, in an order shuffled afresh from a fixed seed, primer.py before each run. Round 1
keeps each arm's outputs (KEEP=1), and parity.py checks every arm's against the first's.

Per run, besides pipeline.py's own figures: the peak rise in the used space of the disk
the run writes to (sampled every 0.2 s), the bytes that disk's device wrote
(/proc/diskstats), and the temporary directories (/tmp/tmp*) made during the run and left
after it. Writes <out>/results.json (every run's figures, the machine, the packages) and
<out>/summary.md, and exits 1 if parity fails.

Usage: ab.py --data DIR --out DIR --arm NAME=SRC [--arm NAME=SRC ...] [--rounds 3]
"""

import argparse
import hashlib
import json
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from importlib import metadata
from pathlib import Path

HERE = Path(__file__).resolve().parent
# The workload: 66 chunks (the 11 files cycled 6 times), 6 workers taking 8 chunks each,
# groups of 22 chunks merged in the background
PER, WORKERS, GROUP, CHUNKS = 8, 6, 22, 66
PACKAGES = ["polars", "polars-genson", "pq-glue", "pyarrow", "orjson"]


def device_sectors_written(path: Path) -> int | None:
    """Sectors written by the block device holding `path`, None if it has none."""
    dev = os.stat(path).st_dev
    for line in open("/proc/diskstats"):
        f = line.split()
        if (int(f[0]), int(f[1])) == (os.major(dev), os.minor(dev)):
            return int(f[9])
    return None


def run_once(name: str, src: Path, files: list[str], out: Path, keep: bool) -> dict:
    subprocess.run([sys.executable, HERE / "primer.py", "3"], check=True)
    stop, peak = threading.Event(), [0]
    used0 = shutil.disk_usage(out).used

    def sample() -> None:
        while not stop.wait(0.2):
            peak[0] = max(peak[0], shutil.disk_usage(out).used - used0)

    tmp = Path(tempfile.gettempdir())
    tmp_before = {p.name for p in tmp.glob("tmp*")}
    w0 = device_sectors_written(out)
    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    result = out / f"{name}.json"
    env = {**os.environ, **({"KEEP": "1"} if keep else {})}
    cmd = [sys.executable, HERE / "pipeline.py", src, result, PER, WORKERS, GROUP, CHUNKS, *files]
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
    result.write_text(json.dumps(run))
    return run


def machine() -> dict:
    cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), None)
    mem = next(int(l.split()[1]) for l in open("/proc/meminfo") if l.startswith("MemTotal"))
    versions = {}
    for p in PACKAGES:
        try:
            versions[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            versions[p] = None
    return {
        "cpu": cpu, "cores": os.cpu_count(), "memory_gb": round(mem / 2**20, 1),
        "kernel": platform.release(), "python": platform.python_version(),
        "runner": {k: os.environ.get(k) for k in ("RUNNER_NAME", "ImageOS", "ImageVersion")},
        "packages": versions,
    }


def git_sha(src: Path) -> str | None:
    r = subprocess.run(["git", "-C", src, "rev-parse", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() or None


def summary(res: dict) -> str:
    arms = list(res["arms"])
    med = lambda a, k: statistics.median(r[k] for r in res["runs"] if r["arm"] == a and r[k] is not None)
    rng = lambda a, k: (lambda v: f"{min(v):g}–{max(v):g}")(
        [r[k] for r in res["runs"] if r["arm"] == a and r[k] is not None])
    ref, *cands = arms
    rows = [
        ("wall (s)", "wall"), ("CPU, all processes (s)", "cpu_total"),
        ("CPU, process (s)", "process_cpu"), ("CPU, partition (s)", "partition_cpu"),
        ("CPU, group merges (s)", "close_cpu"), ("peak RSS, all processes (MB)", "peak_sum_rss_mb"),
        ("peak RSS, one worker (MB)", "worker_peak_mb"), ("disk used, peak rise (MB)", "disk_peak_mb"),
        ("device bytes written (MB)", "device_written_mb"),
    ]
    out = ["## Benchmark: wikidata-pq pipeline, 66 chunks", ""]
    for c in cands:
        out.append(f"**Score ({c}):** {med(ref, 'wall') / med(c, 'wall'):.2f}× the reference's speed "
                   f"(median wall time, {ref} {med(ref, 'wall'):.1f} s, {c} {med(c, 'wall'):.1f} s)")
    out += ["", f"**Parity:** {'passed' if res['parity_ok'] else 'FAILED'}", ""]
    out.append("| | " + " | ".join(f"{a} (`{(res['arms'][a]['sha'] or '?')[:9]}`)" for a in arms) + " |")
    out.append("|---|" + "---|" * len(arms))
    for label, k in rows:
        out.append(f"| {label} | " + " | ".join(rng(a, k) for a in arms) + " |")
    left = sum(len(r["tmp_left"]) for r in res["runs"])
    m = res["machine"]
    pk = ", ".join(f"{k} {v}" for k, v in m["packages"].items() if v)
    out += [
        "", f"Ranges are the lowest and highest of {res['rounds']} rounds. Temporary directories left: {left}.",
        "", f"**Machine:** {m['cpu']}, {m['cores']} cores, {m['memory_gb']} GB, Python {m['python']}, "
        f"{m['runner'].get('ImageOS') or 'local'} {m['runner'].get('ImageVersion') or ''}".rstrip(),
        f"**Packages (both arms):** {pk}",
        f"**Data:** the 11 chunk files, SHA256SUMS hashing to `{res['data_sha256sums'][:16]}`", "",
    ]
    for a, p in res["parity"].items():
        out.append(f"<details><summary>Parity, {a} against {ref}</summary>\n\n```json\n"
                   f"{json.dumps(p, indent=1)}\n```\n</details>")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arm", action="append", required=True, help="NAME=SRC, the reference first")
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()
    arms = {n: Path(s).resolve() for n, s in (a.split("=", 1) for a in args.arm)}
    files = sorted(str(p.resolve()) for p in args.data.glob("*.parquet"))
    assert len(files) == 11, files
    args.out.mkdir(parents=True, exist_ok=True)
    runs = []
    for rnd in range(1, args.rounds + 1):
        order = list(arms)
        random.Random(f"bench-{rnd}").shuffle(order)
        for name in order:
            print(f"round {rnd}: {name}", flush=True)
            r = run_once(f"{name}_r{rnd}", arms[name] / "src", files, args.out, keep=rnd == 1)
            r["close_cpu"] = round(sum(c["cpu_process"] for c in r["closes"]), 1)
            print(f"  wall {r['wall']:.1f} s, CPU {r['cpu_total']:.0f} s, RSS peak {r['peak_sum_rss_mb']} MB, "
                  f"device wrote {r['device_written_mb']} MB", flush=True)
            runs.append({"arm": name, "round": rnd, **{k: v for k, v in r.items() if k not in ("chunks", "samples")}})
    ref, *others = arms
    parity = {}
    for a in others:
        p = subprocess.run([sys.executable, HERE / "parity.py", arms[ref] / "src",
                            args.out / f"{ref}_r1.keep", args.out / f"{a}_r1.keep"],
                           capture_output=True, text=True)
        print(p.stdout + p.stderr, flush=True)
        parity[a] = json.loads(p.stdout.strip().splitlines()[-1]) if p.stdout.strip() else {"ok": False, "error": p.stderr[-2000:]}
    for a in arms:
        shutil.rmtree(args.out / f"{a}_r1.keep", ignore_errors=True)
    res = {
        "workload": {"chunks": CHUNKS, "files": len(files), "workers": WORKERS, "chunks_per_worker": PER, "group": GROUP},
        "rounds": args.rounds,
        "arms": {n: {"src": str(s), "sha": git_sha(s)} for n, s in arms.items()},
        "machine": machine(),
        "data_sha256sums": hashlib.sha256((args.data / "SHA256SUMS").read_bytes()).hexdigest(),
        "runs": runs, "parity": parity, "parity_ok": all(p.get("ok") for p in parity.values()),
    }
    (args.out / "results.json").write_text(json.dumps(res, indent=1))
    (args.out / "summary.md").write_text(summary(res))
    print(summary(res))
    sys.exit(0 if res["parity_ok"] else 1)


if __name__ == "__main__":
    main()
