"""What a persistent worker holds after each chunk: process_and_partition (unchanged) on
each given chunk in turn in this one process; after each, RSS and peak RSS, the Arrow
bytes pyarrow's pool still has allocated, the Polars DataFrames and pyarrow Tables still
alive (and what refers to them), then RSS after gc.collect(), after pyarrow's
release_unused(), and after glibc's malloc_trim(0): runtime calls only, no allocator
setting. With RELEASE=1 those releases are made after each chunk, so the next chunk
starts from them. Usage: retain.py <src> <workdir> <chunk>..."""
import ctypes, gc, os, resource, shutil, sys, time
from pathlib import Path

os.environ["WIKIDATA_RELEASE"] = "20261012"
src, work = str(Path(sys.argv[1]).resolve()), Path(sys.argv[2]).resolve()
sys.path.insert(0, src)
import polars as pl  # noqa: E402
import pyarrow as pa  # noqa: E402
import wikidata.main as M  # noqa: E402

libc = ctypes.CDLL("libc.so.6")


def rss():
    return next(int(l.split()[1]) for l in open("/proc/self/status") if l.startswith("VmRSS")) // 1024


def hwm_reset():
    open("/proc/self/clear_refs", "w").write("5")


def live():
    frames = tables = 0
    for o in gc.get_objects():
        if isinstance(o, pl.DataFrame):
            frames += 1
        elif isinstance(o, (pa.Table, pa.RecordBatch, pa.ChunkedArray)):
            tables += 1
    return frames, tables


for i, chunk in enumerate(sys.argv[3:]):
    d = work / f"c{i}"
    shutil.rmtree(d, ignore_errors=True)
    data, out, state = d / "data", d / "out", d / "state"
    for x in (data, out, state):
        x.mkdir(parents=True)
    os.chdir(d)
    (data / f"chunk_{i}.parquet").symlink_to(Path(chunk).resolve())
    saved = os.dup(1); devnull = os.open(os.devnull, os.O_WRONLY); os.dup2(devnull, 1)
    before = rss(); hwm_reset()
    t = time.perf_counter()
    M.process_and_partition(i, data_dir=data, output_dir=out, repo_id="unused", state_dir=state)
    wall = time.perf_counter() - t
    os.dup2(saved, 1)
    peak = next(int(l.split()[1]) for l in open("/proc/self/status") if l.startswith("VmHWM")) // 1024
    after = rss(); arrow = pa.total_allocated_bytes() // 2**20; frames, tables = live()
    after_gc = after_pa = after_trim = frames_gc = tables_gc = -1
    if not os.environ.get("DECAY"):
        gc.collect(); after_gc = rss(); frames_gc, tables_gc = live()
        pa.default_memory_pool().release_unused(); after_pa = rss()
        libc.malloc_trim(0); after_trim = rss()
    print(f"{i} {Path(chunk).parent.parent.name}/{Path(chunk).name}: {wall:5.2f}s, RSS before {before} peak {peak} after {after} MB | "
          f"arrow live {arrow} MB, {frames} DataFrames {tables} Arrow objects | after gc {after_gc} ({frames_gc}, {tables_gc}) | "
          f"after arrow release_unused {after_pa} | after malloc_trim {after_trim}", flush=True)
    if os.environ.get("DECAY"):  # RSS while idle after the chunk (allocators returning memory on their own)
        t0 = time.perf_counter(); marks = []
        for wait in (0.5, 1, 2, 5, 10, 20):
            time.sleep(max(0, wait - (time.perf_counter() - t0))); marks.append(f"{wait}s:{rss()}")
        print("     idle:", " ".join(marks), flush=True)
    os.chdir(work)
    shutil.rmtree(d)
    if not os.environ.get("RELEASE"):
        # leave the next chunk the state this chunk left, before the probes released it:
        # not possible to undo, so the probe runs are diagnostic only
        pass
print("arrow pool:", pa.default_memory_pool().backend_name)
