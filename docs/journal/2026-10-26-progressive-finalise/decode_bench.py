"""Decoding the same bz2 blocks as one pipeline (today's split: one lbzip2 on all cores)
against 8 sweeps (one decoder per shard window, as a synchronised split would run them).

Input: a directory with w0..w7/s.bz2 (each a window of one shard, its whole blocks
rewrapped as single-block streams by bzip2recover) and all.bz2 (their concatenation).
Each mode's wall and children's CPU (getrusage), `rounds` times, modes alternating.

    python3 -I decode_bench.py DIR --rounds 3
"""

import argparse
import os
import resource
import subprocess
import time


def run(cmds: list[list[str]]) -> tuple[float, float]:
    r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    t0 = time.perf_counter()
    procs = [subprocess.Popen(c, stdout=subprocess.DEVNULL) for c in cmds]
    for p in procs:
        if p.wait() != 0:
            raise SystemExit(f"{p.args} failed")
    wall = time.perf_counter() - t0
    r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    return wall, (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--rounds", type=int, default=3)
    a = ap.parse_args()
    d = a.dir
    windows = [os.path.join(d, f"w{k}", "s.bz2") for k in range(8)]
    modes = {
        "one pipeline, lbzip2 -n4": [["lbzip2", "-dc", "-n4", os.path.join(d, "all.bz2")]],
        "8 sweeps, lbzip2 -n1 each": [["lbzip2", "-dc", "-n1", w] for w in windows],
        "4 sweeps at a time x2, lbzip2 -n1": None,  # (below)
    }
    size = os.path.getsize(os.path.join(d, "all.bz2"))
    print(f"{size / 1e6:.1f} MB of bz2 in 8 windows; load {os.getloadavg()}")
    for r in range(a.rounds):
        for name, cmds in modes.items():
            if cmds is None:
                w1, c1 = run([["lbzip2", "-dc", "-n1", w] for w in windows[:4]])
                w2, c2 = run([["lbzip2", "-dc", "-n1", w] for w in windows[4:]])
                wall, cpu = w1 + w2, c1 + c2
            else:
                wall, cpu = run(cmds)
            print(f"round {r}: {name}: wall {wall:.2f} s, CPU {cpu:.2f} s, "
                  f"{cpu / size * 1e9:.0f} CPU-s a GB of bz2; load {os.getloadavg()[0]:.2f}", flush=True)


if __name__ == "__main__":
    main()
