"""push.merge_group on a snapshot of partitioned chunks (pipeline_bench.py SNAPSHOT), one
table at a time, the snapshot hard-linked afresh each round: wall, CPU and peak RSS of
each table's merge, files in and out, bytes out. KEEP=dir keeps the staged files.

Usage: merge_bench.py <src> <snapshot> <first> <last> [rounds] [tables...]"""
import os, shutil, subprocess, sys, time
from pathlib import Path

os.environ["WIKIDATA_RELEASE"] = "20261012"
src, snap, first, last = sys.argv[1], Path(sys.argv[2]).resolve(), int(sys.argv[3]), int(sys.argv[4])
rounds = int(sys.argv[5]) if len(sys.argv) > 5 else 1
only = sys.argv[6:]
run = Path(os.environ.get("RUN", str(snap.parent / "snap.run")))  # where the snapshot was taken: its sidecars hold absolute paths
sys.path.insert(0, str(Path(src).resolve()))


def kb(f):
    return next(int(l.split()[1]) for l in open("/proc/self/status") if l.startswith(f)) // 1024


for r in range(rounds):
    shutil.rmtree(run, ignore_errors=True)
    run.mkdir()
    subprocess.run(["cp", "-al", str(snap), str(run / "releases")], check=True)
    (run / "releases").rename(run / "x"); (run / "releases").mkdir(); (run / "x").rename(run / "releases" / "20261012")
    os.chdir(run)
    import wikidata.push.core as core
    from wikidata.config import AUDIT_DIR, STAGING_DIR, Table
    from wikidata.push.groups import Group

    g = Group(first, last, last)
    tot_w = tot_c = 0.0
    for t in Table:
        if only and t.value not in only:
            continue
        core.Table = [t]
        n_in = len(list((Path("releases/20261012/results") / t).rglob("*.parquet")))
        open("/proc/self/clear_refs", "w").write("5")
        w, c = time.perf_counter(), time.process_time()
        core.merge_group(g, AUDIT_DIR, STAGING_DIR)
        w, c = time.perf_counter() - w, time.process_time() - c
        tot_w += w; tot_c += c
        outs = list((STAGING_DIR / t).rglob("*.parquet"))
        print(f"r{r} {t:14s} {w:6.2f}s wall {c:6.2f}s cpu  peak {kb('VmHWM'):5d} MB  {n_in:4d} files -> {len(outs):3d}, {sum(p.stat().st_size for p in outs)/2**20:7.1f} MB", flush=True)
    print(f"r{r} total          {tot_w:6.2f}s wall {tot_c:6.2f}s cpu", flush=True)
    if os.environ.get("KEEP") and r == rounds - 1:
        k = Path(os.environ["KEEP"]); shutil.rmtree(k, ignore_errors=True)
        shutil.copytree(STAGING_DIR, k)
    os.chdir("/")
    shutil.rmtree(run)
