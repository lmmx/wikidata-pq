"""partition_chunk's per-table loop (prepare_for_partition then partition_parquet, with
config's PARTITION_COLS) on one chunk's processed tables, in a fresh process, from
relative paths so audit sidecars are comparable between runs. Prints JSON: wall and CPU
per table, RSS and peak RSS after each, and the process's peak RSS.
Usage: python partition_one.py <processed out dir> <run dir> <src dir>"""
import json, os, resource, shutil, sys, time
from pathlib import Path
SRC = str(Path(sys.argv[3]).resolve())  # before the chdir below
processed, run = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
os.environ["WIKIDATA_RELEASE"] = "20261012"
if run.exists(): shutil.rmtree(run)
run.mkdir(parents=True); os.chdir(run)
sys.path.insert(0, SRC)
from wikidata.config import PARTITION_COLS, Table
from wikidata.partitioning import partition_parquet, prepare_for_partition
import wikidata.partitioning.claims as _claims
assert _claims.__file__.startswith(SRC), (_claims.__file__, SRC)
def cpu():
    r = resource.getrusage(resource.RUSAGE_SELF); return r.ru_utime + r.ru_stime
def mem():
    d = {l.split(":")[0]: int(l.split()[1]) // 1024 for l in open("/proc/self/status") if l.startswith(("VmRSS", "VmHWM"))}
    return [d["VmRSS"], d["VmHWM"]]
times, mems = {}, {}
for tbl in Table:
    src = processed / tbl
    (Path("out") / tbl).mkdir(parents=True)
    for f in sorted(src.glob("*.parquet")):
        shutil.copy(f, Path("out") / tbl / f.name)
        t, c = time.perf_counter(), cpu()
        lf = prepare_for_partition(Path("out") / tbl / f.name, tbl)
        partition_parquet(by=PARTITION_COLS[tbl], lf=lf, source_name=f.name, dst_dir=Path("out") / tbl, log_dir=Path("audit") / tbl)
        times[str(tbl)] = [round(time.perf_counter() - t, 3), round(cpu() - c, 2)]
        mems[str(tbl)] = mem()
        if os.environ.get("PART_VERBOSE"):
            print(tbl, times[str(tbl)], mems[str(tbl)], file=sys.stderr, flush=True)
        (Path("out") / tbl / f.name).unlink()  # as CLEAN_UP_LOCAL does; only partition outputs remain
print(json.dumps({"run": str(run), "code": _claims.__file__, "tables": times, "rss_hwm_mb": mems, "maxrss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)}))
