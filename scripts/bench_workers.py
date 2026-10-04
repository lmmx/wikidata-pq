#!/usr/bin/env python3
"""Measure chunk throughput by number of workers (WIKIDATA_WORKERS), on a release's own
unprocessed chunks, to choose how many to run at once.

Each trial runs the real pipeline (`main.run`: process, partition, merge groups) on the
same sample of chunks, hard-linked into a scratch release under releases/_bench_workers/
(the originals are untouched; nothing is uploaded: a group is merged and then dropped).
It reports wall time, chunks and source MB per hour, the whole machine's CPU use, and the
peak memory of the trial's processes. Run it while no release is running.

The speedup column is over the first worker count in --workers.

Usage: python scripts/bench_workers.py 20260928 [--set scholar|main] [--chunks 30]
           [--workers 1,2,3,4,6,8]
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "releases" / "_bench_workers"


def sample_chunks(set_dir: Path, n: int) -> list[dict]:
    """n manifest entries spread evenly over the set's chunks still on disk."""
    data = set_dir / "data"
    entries = [json.loads(l) for l in (data / "manifest.jsonl").read_text().splitlines() if l]
    left = sorted(
        (e for e in entries if (data / e["file"]).exists()), key=lambda e: e["chunk"]
    )
    if len(left) < n:
        raise SystemExit(f"Only {len(left)} chunks left in {data}")
    return [left[i * len(left) // n] for i in range(n)]


def setup_trial(trial_dir: Path, set_dir: Path, sample: list[dict]) -> None:
    shutil.rmtree(trial_dir, ignore_errors=True)
    data = trial_dir / "releases" / "bench" / "data"
    data.mkdir(parents=True)
    with (data / "manifest.jsonl").open("w") as f:
        for i, e in enumerate(sample):
            src = set_dir / "data" / e["file"]
            dst = data / f"chunk_{i}.parquet"
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
            f.write(json.dumps({**e, "chunk": i, "file": dst.name}) + "\n")
    (data / "split.done").touch()
    (data / "route.done").touch()


def cpu_busy() -> tuple[int, int]:
    """(busy, total) jiffies over all CPUs, from /proc/stat."""
    fields = list(map(int, Path("/proc/stat").read_text().split("\n", 1)[0].split()[1:]))
    idle = fields[3] + fields[4]
    return sum(fields) - idle, sum(fields)


def trial_rss(trial_dir: Path) -> int:
    """Bytes resident in every process whose working directory is the trial's."""
    total = 0
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            if Path(os.readlink(p / "cwd")) != trial_dir:
                continue
            for line in (p / "status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1]) * 1024
        except OSError:
            continue
    return total


def run_trial(workers: int, set_dir: Path, sample: list[dict]) -> dict:
    trial_dir = BENCH / f"k{workers}"
    setup_trial(trial_dir, set_dir, sample)
    env = {
        **os.environ,
        "WIKIDATA_RELEASE": "bench",
        "WIKIDATA_SCHOLAR": "",
        "WIKIDATA_WORKERS": str(workers),
    }
    log = (BENCH / f"k{workers}.log").open("w")
    b0, t0 = cpu_busy()
    start = time.time()
    proc = subprocess.Popen(
        [sys.executable, __file__, "--trial", str(len(sample))],
        cwd=trial_dir,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    peak = 0
    while proc.poll() is None:
        peak = max(peak, trial_rss(trial_dir))
        time.sleep(0.5)
    wall = time.time() - start
    b1, t1 = cpu_busy()
    log.close()
    if proc.returncode != 0:
        raise SystemExit(f"Trial with {workers} workers failed: see {BENCH}/k{workers}.log")
    shutil.rmtree(trial_dir)
    mb = sum(e["bytes"] for e in sample) / 1e6
    return {
        "workers": workers,
        "wall_s": wall,
        "chunks_per_h": len(sample) / wall * 3600,
        "mb_per_h": mb / wall * 3600,
        "cpu_pct": 100 * (b1 - b0) / (t1 - t0),
        "peak_rss_gb": peak / 1e9,
    }


def trial(n_chunks: int) -> None:
    """In the trial's directory: the pipeline with groups merged locally, not uploaded."""
    from wikidata import main as m
    from wikidata.config import AUDIT_DIR, STAGING_DIR
    from wikidata.push.core import _set_step, merge_group
    from wikidata.push.groups import Group, record_stage
    from wikidata.state import Step

    def close(chunks, target_repos, state_dir):
        group = Group(chunks[0], chunks[-1])
        record_stage(state_dir, group, "closed")
        merge_group(group, AUDIT_DIR, STAGING_DIR)
        shutil.rmtree(STAGING_DIR, ignore_errors=True)
        _set_step(group, Step.COMPLETE, state_dir)
        record_stage(state_dir, group, "done")

    # About three groups over the sample, as a run closes one every few hundred chunks
    m._close_open_group = close
    m.open_group_bytes = lambda state_dir, chunks: len(chunks)
    m.group_threshold_bytes = lambda state_dir, total: max(1, n_chunks // 3)
    m.run()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("release", nargs="?")
    ap.add_argument("--set", default="scholar", choices=["scholar", "main"])
    ap.add_argument("--chunks", type=int, default=30)
    ap.add_argument("--workers", default="1,2,3,4,6,8")
    ap.add_argument("--trial", type=int, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.trial:
        return trial(args.trial)

    name = f"{args.release}-scholar" if args.set == "scholar" else args.release
    set_dir = ROOT / "releases" / name
    sample = sample_chunks(set_dir, args.chunks)
    mb = sum(e["bytes"] for e in sample) / 1e6
    print(
        f"{len(sample)} chunks of {set_dir.name}, spread over those left (e.g. "
        f"{', '.join(str(e['chunk']) for e in sample[:3])}, …, {sample[-1]['chunk']}): "
        f"{sum(e['rows'] for e in sample):,} rows, {mb:.0f} MB"
    )
    print(
        f"{'workers':>7} {'wall':>7} {'chunks/h':>9} {'MB/h':>7} {'speedup':>8} "
        f"{'CPU':>5} {'peak RSS':>9}"
    )
    base = None
    for k in map(int, args.workers.split(",")):
        r = run_trial(k, set_dir, sample)
        base = base or r["wall_s"]  # the speedup is over the first count tried
        speedup = f"{base / r['wall_s']:.2f}x"
        print(
            f"{k:>7} {r['wall_s']:>6.0f}s {r['chunks_per_h']:>9.0f} {r['mb_per_h']:>7.0f} "
            f"{speedup:>8} {r['cpu_pct']:>4.0f}% {r['peak_rss_gb']:>7.1f} GB",
            flush=True,
        )
    shutil.rmtree(BENCH, ignore_errors=True)


if __name__ == "__main__":
    main()
