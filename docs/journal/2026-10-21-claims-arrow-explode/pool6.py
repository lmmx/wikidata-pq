"""A set's chunks through pool.process_chunks and main.process_and_partition, as main.run
drives them (no pull and no upload), WIKIDATA_WORKERS at a time, in a fresh run dir.
Samples every worker's RSS every 0.1 s from /proc. Prints JSON: wall, the run's CPU (the
workers'), the peak of the workers' summed RSS, each worker's peak RSS, wall time and CPU.
Usage: python pool6.py <run dir> <src dir> <scholar 0|1> <name>=<routed chunk file> ..."""
import os, sys
from pathlib import Path
# Spawned workers re-run this module from inside the run dir: they take the paths the
# parent resolved
RUN = Path(os.environ.setdefault("POOL6_RUN", str(Path(sys.argv[1]).resolve())))
SRC = os.environ.setdefault("POOL6_SRC", str(Path(sys.argv[2]).resolve()))
os.environ["WIKIDATA_RELEASE"] = "20261012"
os.environ["WIKIDATA_SCHOLAR"] = sys.argv[3]
os.environ.setdefault("WIKIDATA_WORKERS", "6")
if __name__ == "__main__":
    import shutil
    shutil.rmtree(RUN, ignore_errors=True)
    RUN.mkdir(parents=True)
os.chdir(RUN)
sys.path.insert(0, SRC)

import json, resource, threading, time
from functools import partial
import wikidata.partitioning.claims as _claims
assert _claims.__file__.startswith(SRC), _claims.__file__
from wikidata import main as M
from wikidata.config import CHUNK_WORKERS, OUTPUT_DIR, ROOT_DATA_DIR, STATE_DIR
from wikidata.pool import process_chunks


def timed(chunk_idx: int, **kw) -> None:
    """process_and_partition, its wall time and CPU written to timings/<chunk>.json."""
    t = time.perf_counter()
    M.process_and_partition(chunk_idx, **kw)
    r = resource.getrusage(resource.RUSAGE_SELF)
    Path("timings").mkdir(exist_ok=True)
    Path(f"timings/{chunk_idx}.json").write_text(json.dumps(
        {"wall": round(time.perf_counter() - t, 2), "cpu": round(r.ru_utime + r.ru_stime, 1), "maxrss_mb": r.ru_maxrss // 1024}))


def children() -> dict[int, int]:
    """RSS in MB of this process's live children, by pid."""
    me, out = os.getpid(), {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            stat = open(f"/proc/{d}/stat").read()
            if int(stat.rsplit(")", 1)[1].split()[1]) == me:
                out[int(d)] = int(open(f"/proc/{d}/statm").read().split()[1]) * 4096 // 2**20
        except (OSError, ValueError, IndexError):
            pass
    return out


if __name__ == "__main__":
    chunks = {}
    for arg in sys.argv[4:]:
        name, f = arg.split("=")
        idx = int(Path(name).stem.split("_")[1])
        ROOT_DATA_DIR.mkdir(parents=True, exist_ok=True)
        (ROOT_DATA_DIR / f"chunk_{idx}.parquet").symlink_to(Path(f).resolve())
        chunks[idx] = name
    for d in (OUTPUT_DIR, STATE_DIR):
        d.mkdir(parents=True, exist_ok=True)
    peak, stop = {"sum": 0, "n": 0}, threading.Event()

    def sample() -> None:
        while not stop.is_set():
            rss = children()
            peak["sum"], peak["n"] = max(peak["sum"], sum(rss.values())), max(peak["n"], len(rss))
            time.sleep(0.1)

    threading.Thread(target=sample, daemon=True).start()
    t = time.perf_counter()
    process_chunks(
        sorted(chunks), [], workers=CHUNK_WORKERS, start=lambda c: None,
        work=partial(timed, data_dir=ROOT_DATA_DIR, output_dir=OUTPUT_DIR, repo_id="unused", state_dir=STATE_DIR),
        done=lambda c: None, ready=lambda g: False, close=lambda g: None,
    )
    wall = time.perf_counter() - t
    stop.set()
    r = resource.getrusage(resource.RUSAGE_CHILDREN)
    per = {chunks[c]: json.loads(Path(f"timings/{c}.json").read_text()) for c in sorted(chunks)}
    print(json.dumps({"run": str(RUN), "code": _claims.__file__, "workers": CHUNK_WORKERS, "chunks": len(chunks),
                      "wall": round(wall, 1), "cpu": round(r.ru_utime + r.ru_stime, 1),
                      "peak_sum_rss_mb": peak["sum"], "max_live_workers": peak["n"], "per_chunk": per}))
