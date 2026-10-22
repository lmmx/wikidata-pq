"""wikidata-pq's chunk pool as `run` drives it (pool.process_chunks, the real
process_and_partition per chunk, CHUNK_WORKERS at once, chunks in index order), over a
release's worth of stand-in chunks: chunk i is a link to the i-th of the given chunk files,
cycled. No upload (groups never close). Per chunk: its worker's pid, its wall and CPU, and
the worker's RSS and peak RSS after it; every 0.2 s, the summed RSS of the workers.

Usage: pool_bench.py <src> <out.json> <chunks_per_worker> <workers> <n chunks> <chunk files...>
"""
import json, os, resource, shutil, sys, threading, time
from functools import partial
from pathlib import Path

os.environ["WIKIDATA_RELEASE"] = "20261012"


def kb(field: str) -> int:
    return next(int(l.split()[1]) for l in open("/proc/self/status") if l.startswith(field))


def bench_work(chunk: int, log: str, **kw) -> None:
    from wikidata.main import process_and_partition

    t, r = time.perf_counter(), resource.getrusage(resource.RUSAGE_SELF)
    process_and_partition(chunk, **kw)
    r2 = resource.getrusage(resource.RUSAGE_SELF)
    if os.environ.get("CLEAN"):  # the chunk's partition files, once timed (disk)
        for f in Path(kw["output_dir"]).glob(f"*/*/chunk_{chunk}.parquet"):
            f.unlink()
        for f in Path(kw["output_dir"]).glob(f"*/*/chunk_{chunk}_*.parquet"):
            f.unlink()
    with open(log, "a") as f:
        f.write(json.dumps({"chunk": chunk, "pid": os.getpid(), "wall": time.perf_counter() - t,
                            "user": r2.ru_utime - r.ru_utime, "sys": r2.ru_stime - r.ru_stime,
                            "rss_mb": kb("VmRSS") // 1024, "hwm_mb": kb("VmHWM") // 1024}) + "\n")


def children_rss_mb(pid: int) -> int:
    total = 0
    for task in Path(f"/proc/{pid}/task").iterdir():
        for child in (task / "children").read_text().split():
            try:
                total += next(int(l.split()[1]) for l in open(f"/proc/{child}/status") if l.startswith("VmRSS"))
            except (OSError, StopIteration):
                pass
    return total // 1024


if __name__ == "__main__":
    src, out, per, workers, n = sys.argv[1], Path(sys.argv[2]).resolve(), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    files = [str(Path(f).resolve()) for f in sys.argv[6:]]
    run_dir = out.with_suffix(".run")
    shutil.rmtree(run_dir, ignore_errors=True)
    work_dir = run_dir / "releases" / "20261012"
    data, state, results = work_dir / "data", work_dir / "state", work_dir / "results"
    for d in (data, state, results):
        d.mkdir(parents=True)
    for i in range(n):
        (data / f"chunk_{i}.parquet").symlink_to(files[i % len(files)])
    os.chdir(run_dir)
    sys.path.insert(0, str(Path(src).resolve()))
    from wikidata.pool import process_chunks

    log = str(out.with_suffix(".chunks.jsonl"))
    Path(log).unlink(missing_ok=True)
    samples, stop = [], threading.Event()

    def sample() -> None:
        t0 = time.perf_counter()
        while not stop.wait(0.2):
            samples.append((round(time.perf_counter() - t0, 2), children_rss_mb(os.getpid())))

    threading.Thread(target=sample, daemon=True).start()
    t = time.perf_counter()
    import inspect
    extra = {"chunks_per_worker": per} if "chunks_per_worker" in inspect.signature(process_chunks).parameters else {}
    assert extra or per == 1, "this pool.py has no chunks_per_worker"
    process_chunks(
        list(range(n)), [], workers=workers, **extra, start=lambda c: None,
        work=partial(bench_work, log=log, data_dir=data.resolve(), output_dir=results.resolve(),
                     repo_id="unused", state_dir=state.resolve()),
        done=lambda c: None, ready=lambda g: False, close=lambda g: None,
    )
    wall = time.perf_counter() - t
    stop.set()
    chunks = [json.loads(l) for l in open(log)]
    res = {"chunks_per_worker": per, "workers": workers, "n": n, "wall": round(wall, 2),
           "workers_spawned": len({c["pid"] for c in chunks}),
           "peak_sum_rss_mb": max(s for _, s in samples), "mean_sum_rss_mb": round(sum(s for _, s in samples) / len(samples)),
           "chunks": chunks, "samples": samples}
    out.write_text(json.dumps(res))
    if os.environ.get("KEEP"):  # every output file's sha256, for comparing runs
        import hashlib
        sums = {str(p.relative_to(work_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(work_dir.rglob("*")) if p.is_file() and not p.is_symlink() and "state" not in p.parts}
        out.with_suffix(".sha256.json").write_text(json.dumps(sums, indent=0))
    shutil.rmtree(run_dir)
    print(f"per={per} workers={workers} n={n}: wall {wall:.1f}s, {res['workers_spawned']} processes, "
          f"summed RSS peak {res['peak_sum_rss_mb']} MB mean {res['mean_sum_rss_mb']} MB, "
          f"chunk user {sum(c['user'] for c in chunks):.1f}s sys {sum(c['sys'] for c in chunks):.1f}s")
