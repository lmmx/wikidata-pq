"""wikidata-pq's chunk pipeline as `run` drives it, upload aside: pool.process_chunks with
the real process_and_partition per chunk (`workers` at once, `per` chunks a worker), and
groups of `group` chunks closed in the background as close_group does it, merge included,
without the upload and its verification (push_group, verify_group). Chunk i is a link to
the i-th of the given chunk files, cycled.

Per chunk: its wall and CPU (process and partition apart) and its worker's RSS; per group:
its close's wall and CPU (the parent's); every 0.2 s, the summed RSS of the parent and its
workers; for the whole run, the CPU time of the parent and every worker. KEEP=1 keeps
the outputs (staged group files, audit, quarantine, and finalise's sorted part files)
under <out>.keep/ for comparing runs.

Usage: pipeline.py <src> <out.json> <per> <workers> <group> <n chunks> <chunk files...>
       pipeline.py --split-gz <src> <out.json> <workers> <chunk files...>   (see split_gz)
"""

import hashlib
import json
import os
import subprocess
import tempfile
import re
import resource
import shutil
import sys
import threading
import time
from functools import partial
from pathlib import Path

import phases

HERE = Path(__file__).resolve().parent

os.environ["WIKIDATA_RELEASE"] = "20261012"
# finalise runs its local stages only: huggingface_hub refuses the network, and every Hub
# entry point of the finalise modules is replaced with one that raises (main)
os.environ["HF_HUB_OFFLINE"] = "1"


def kb(field: str, pid: str = "self") -> int:
    return next(int(l.split()[1]) for l in open(f"/proc/{pid}/status") if l.startswith(field))


def bench_work(chunk: int, log: str, **kw) -> None:
    from wikidata.main import partition_chunk, process

    t, r = time.perf_counter(), resource.getrusage(resource.RUSAGE_SELF)
    process(chunk_idx=chunk, **kw)
    tp, rp = time.perf_counter(), resource.getrusage(resource.RUSAGE_SELF)
    partition_chunk(chunk, kw["state_dir"], kw["output_dir"])
    r2 = resource.getrusage(resource.RUSAGE_SELF)
    cpu = lambda a, b: b.ru_utime + b.ru_stime - a.ru_utime - a.ru_stime
    with open(log, "a") as f:
        f.write(json.dumps({
            "chunk": chunk, "pid": os.getpid(), "wall": time.perf_counter() - t, "cpu": cpu(r, r2),
            "process_wall": tp - t, "process_cpu": cpu(r, rp),
            "partition_wall": time.perf_counter() - tp, "partition_cpu": cpu(rp, r2),
            "rss_mb": kb("VmRSS") // 1024, "hwm_mb": kb("VmHWM") // 1024,
        }) + "\n")


def no_hub(*args, **kwargs):
    raise RuntimeError("the benchmark never touches the Hub")


def finalise(compact, sort_by_id, Table, compact_dir: Path, hub_copy_dir: Path, state: Path) -> dict:
    """finalise's local stages, as main.finalise runs them for a release, claims first:
    each table compacted (compact_table, its download recorded as done: the group files
    are already where it puts them), then sorted (sort_by_id's sourced, written and done
    stages). Not run: every stage that touches the Hub (download, source_table,
    commit_table, verify_table, the cards' push), which raise if called, and claims_labels,
    which a release builds from both sets' sorted labels. Returns each step's wall time and
    CPU time (this process and the workers it waits for)."""
    for module, names in ((compact, ["download", "snapshot_download"]),
                          (sort_by_id, ["snapshot_download", "HfApi", "source_table",
                                        "commit_table", "verify_table"])):
        for name in names:
            setattr(module, name, no_hub)
    tables = sorted((t for t in Table if t != Table.CLAIMS_LABELS and (compact_dir / "src" / t).is_dir()),
                    key=lambda t: t != Table.CLAIMS)
    usage = lambda: sum(u.ru_utime + u.ru_stime for u in map(resource.getrusage, (resource.RUSAGE_SELF,
                                                                               resource.RUSAGE_CHILDREN)))

    def compact_tables() -> None:
        for table in tables:
            compact.record_stage(state, table, "downloaded")
            compact.compact_table(table, "bench/unused", state)

    def sort_tables() -> None:
        for table in tables:
            (hub_copy_dir / table).mkdir(parents=True, exist_ok=True)
            sort_by_id.check_compacted_copy(table)
            sort_by_id.record_stage(state, table, "sourced")
            sort_by_id.remove_compact_src(table)
            sort_by_id.write_table(table)
            sort_by_id.record_stage(state, table, "written")
            sort_by_id.write_metadata(table)
            sort_by_id.replace_local_copy(table)
            sort_by_id.record_stage(state, table, "done")

    out = {}
    for step, run in (("compact", compact_tables), ("sort", sort_tables)):
        t, c = time.perf_counter(), usage()
        run()
        # Light jobs share a pool for the run (wikidata.parallel); its workers' CPU time is
        # counted when they exit, so each step shuts it down, paying its own workers' start.
        getattr(sys.modules.get("wikidata.parallel"), "shutdown_shared", lambda: None)()
        out[f"{step}_wall"], out[f"{step}_cpu"] = round(time.perf_counter() - t, 2), round(usage() - c, 1)
    out["finalise_wall"] = round(out["compact_wall"] + out["sort_wall"], 2)
    out["finalise_cpu"] = round(out["compact_cpu"] + out["sort_cpu"], 1)
    return out


def finalise_at_scale(compact, sort_by_id, Table, compact_dir: Path, hub_copy_dir: Path, state: Path,
                      n: int, files: list[str], wall: float, workers: int,
                      keep: Path | None = None) -> tuple[dict, dict]:
    """finalise as the release would run it, and its time at the release's size.

    The sample has the release's keys (its languages and sites) with a fraction of their
    rows, so a key that would be over SORT_IN_MEMORY_BYTES in the release is sorted through
    buckets here too: the threshold is scaled by the sample's fraction of the release's
    entities (bucket and file sizes are not, so their per-bucket and per-file costs stand
    as in the release). Then finalise runs again on the group's files three times over, as
    three groups: the same keys with three times the rows, each sorted the way the first
    run sorted it (the threshold tripled as well).

    Of the two, finalise's time at the release's size: a fixed part (per key: the same in
    the release, its keys being the sample's) and a part per byte of source Parquet,
    scaled to the release's bytes. The part per byte is the CPU time the extra rows took,
    over the workers (at the release's size every worker is busy); the fixed part is the
    rest of the first run's wall time. Returns the first run's figures (its outputs are
    the parity's) and the estimates, the pipeline's scaled by entities as before."""
    import pyarrow.parquet as pq
    from release import RELEASE_ENTITIES
    from wikidata import config
    from wikidata.push.groups import chunk_range_name

    entities = sum(pq.read_metadata(files[i % len(files)]).num_rows for i in range(n))
    scale = RELEASE_ENTITIES / entities
    sort_by_id.SORT_IN_MEMORY_BYTES = int(config.SORT_IN_MEMORY_BYTES / scale)
    snap = compact_dir.parent / "compact-src"
    shutil.copytree(compact_dir / "src", snap, copy_function=os.link)
    parts = sorted(snap.rglob("*.parquet"))
    source = sum(p.stat().st_size for p in parts)
    phases.marker("finalise")
    fin = finalise(compact, sort_by_id, Table, compact_dir, hub_copy_dir, state)
    phases.marker("finalise end")
    if keep:  # the first run's sorted part files, what a release publishes (linked)
        for p in sorted(hub_copy_dir.rglob("*.parquet")):
            to = keep / "final" / p.relative_to(hub_copy_dir)
            to.parent.mkdir(parents=True, exist_ok=True)
            os.link(p, to)

    # Three times over: the group's files linked as three groups' (their ids repeated)
    shutil.rmtree(compact_dir)
    shutil.rmtree(hub_copy_dir, ignore_errors=True)
    for p in parts:
        key_dir = compact_dir / "src" / p.relative_to(snap).parent
        key_dir.mkdir(parents=True, exist_ok=True)
        for j in range(3):
            os.link(p, key_dir / f"{chunk_range_name(j * n, j * n + n - 1, 3 * n - 1)}.parquet")
    shutil.rmtree(snap)
    # Its own state, the chunks' state files three times over (compaction reads the run's
    # last chunk from their names)
    state3 = state.parent / "state-x3"
    state3.mkdir()
    for f in state.glob("chunk_*.jsonl"):
        i, rest = re.fullmatch(r"chunk_(\d+)(.*)", f.name).groups()
        for j in range(3):
            shutil.copy(f, state3 / f"chunk_{int(i) + j * n}{rest}")
    # Each key, its rows tripled, sorted the way the first run sorted it: the threshold
    # tripled too, so the extra time is the extra rows', not that of keys moved to another
    # path (more of them sorted through buckets, or as larger keys in a fresh worker each)
    sort_by_id.SORT_IN_MEMORY_BYTES *= 3
    phases.marker("finalise x3")
    fin3 = finalise(compact, sort_by_id, Table, compact_dir, hub_copy_dir, state3)
    phases.marker("finalise x3 end")

    per_byte = max(fin3["finalise_cpu"] - fin["finalise_cpu"], 0) / (2 * source)  # CPU-s a byte
    fixed = fin["finalise_wall"] - per_byte * source / workers
    finalise_eta = fixed + per_byte * source * scale / workers
    pipeline_eta = wall * scale
    return fin, {
        "scale": round(scale, 1), "finalise_source_mb": round(source / 1e6, 1),
        "finalise_x3_wall": fin3["finalise_wall"], "finalise_x3_cpu": fin3["finalise_cpu"],
        "finalise_x3_wall_fit": round(fixed + per_byte * 3 * source / workers, 2),
        "finalise_fixed": round(fixed, 2), "finalise_cpu_per_mb": round(per_byte * 1e6, 3),
        "pipeline_eta": round(pipeline_eta), "finalise_eta": round(finalise_eta),
        "eta": round(pipeline_eta + finalise_eta),
    }


def split_stage(src: Path, files: list[str], workers: int, work: Path, log, fmt: str = "bz2") -> dict:
    """The release's first stage, from the dump to chunk files: split-dump, then
    route-release (dump.split, dump.route), each run as the CLI runs it, in a process of
    its own in a directory of its own, on a slice of a dump made of the sample's own
    entities (dumpslice.py, made once in `work`, untimed), in `fmt`: bz2, or gz with
    WIKIDATA_DUMP_FORMAT=gz (its figures named `gz_...`). Returns each step's wall and
    CPU time (its process and every process it waits for: the decoder, the workers), the
    slice's size, whether the chunks it made hold exactly the sample's rows, the
    scholarly works (the `s` files) in the scholarly set and the rest in the main one,
    and a digest of the manifests it wrote (which split from bz2 and from gz must share)."""
    import polars as pl
    from dumpslice import gz_path

    release = os.environ["WIKIDATA_RELEASE"]
    p = "" if fmt == "bz2" else f"{fmt}_"
    # (BENCH_DUMP_SLICE: where CI caches it between jobs, by the data and dumpslice.py)
    bz2_slice = Path(os.environ.get("BENCH_DUMP_SLICE") or work / "dump-slice.json.bz2")
    dump_slice = bz2_slice if fmt == "bz2" else gz_path(bz2_slice)
    if not dump_slice.exists():
        dump_slice.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, HERE / "dumpslice.py", bz2_slice, *files], check=True)
    out = {f"{p}split_source_mb": round(dump_slice.stat().st_size / 1e6, 1)}
    env = {k: v for k, v in os.environ.items() if k != "WIKIDATA_DUMP_FORMAT"}
    if fmt != "bz2":
        env["WIKIDATA_DUMP_FORMAT"] = fmt
    with tempfile.TemporaryDirectory(dir=work) as tmp:
        dump_dir = Path(tmp) / "releases" / release / "dump"
        dump_dir.mkdir(parents=True)
        os.link(dump_slice, dump_dir / f"wikidata-{release}-all.json.{fmt}")
        code = f"import sys; sys.path.insert(0, {str(src)!r}); from wikidata import dump; dump.{{}}()"
        for step in ("split", "route"):
            phases.marker(p + step)
            r0, t = resource.getrusage(resource.RUSAGE_CHILDREN), time.perf_counter()
            proc = subprocess.Popen([sys.executable, "-c", code.format(step)], cwd=tmp, stdout=log,
                                    stderr=subprocess.STDOUT, env=env)
            peak = 0  # the step's process and its own (the decoder, the workers), summed
            while proc.poll() is None:
                try:
                    peak = max(peak, tree_rss_mb(proc.pid))
                except (OSError, StopIteration):
                    pass
                time.sleep(0.2)
            if proc.returncode:
                raise subprocess.CalledProcessError(proc.returncode, f"dump.{step}()")
            r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
            out[f"{p}{step}_peak_rss_mb"] = peak
            phases.marker(f"{p}{step} end")
            out[f"{p}{step}_wall"] = round(time.perf_counter() - t, 2)
            out[f"{p}{step}_cpu"] = round(r1.ru_utime + r1.ru_stime - r0.ru_utime - r0.ru_stime, 1)
        made = lambda d: pl.concat([pl.read_parquet(f) for f in sorted(d.glob("chunk_*.parquet"))]).sort("id")
        want = lambda s: pl.concat([pl.read_parquet(f) for f in files if Path(f).name.startswith("s") == s]).sort("id")
        sets = [Path(tmp) / "releases" / release / "data", Path(tmp) / "releases" / f"{release}-scholar" / "data"]
        out[f"{p}split_parity_ok"] = made(sets[0]).equals(want(False)) and made(sets[1]).equals(want(True))
        # Each manifest's entries by chunk (the split writes its own in the order its
        # chunks finish)
        manifests = [sets[0] / "manifest.split.jsonl", *(d / "manifest.jsonl" for d in sets)]
        h = hashlib.sha256()
        for m in manifests:
            entries = sorted((json.loads(line) for line in m.read_text().splitlines() if line), key=lambda e: e["chunk"])
            h.update(json.dumps(entries, sort_keys=True).encode())
        out[f"{p}split_manifests_sha256"] = h.hexdigest()
    out[f"{p}split_wall_total"] = round(out[f"{p}split_wall"] + out[f"{p}route_wall"], 2)
    out[f"{p}split_cpu_total"] = round(out[f"{p}split_cpu"] + out[f"{p}route_cpu"], 1)
    out[f"{p}split_peak_rss_mb"] = max(out[f"{p}split_peak_rss_mb"], out.pop(f"{p}route_peak_rss_mb"))
    return out


def decode(cmd: list[str], path: Path) -> tuple[float, float]:
    """The wall and CPU time of a decoder alone, decompressing `path` to nothing."""
    r0, t = resource.getrusage(resource.RUSAGE_CHILDREN), time.perf_counter()
    with open(path, "rb") as f:
        subprocess.run(cmd, stdin=f, stdout=subprocess.DEVNULL, check=True)
    r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    return round(time.perf_counter() - t, 2), round(r1.ru_utime + r1.ru_stime - r0.ru_utime - r0.ru_stime, 2)


def split_gz(src: Path, out: Path, workers: int, files: list[str]) -> None:
    """The split from the gz dump (WIKIDATA_DUMP_FORMAT=gz), as split_stage runs it from
    the bz2, with its ETA: its CPU time per byte of the slice's gz, scaled to the gz dump
    (RELEASE_DUMP_GZ_BYTES) over the cores; and each decoder alone on its slice, the
    decoder the commit's split uses (`gz_decoder`). Run apart from the timed run, after
    it (ab.run_once), so none of the timed run's figures (its disk and device writes
    among them) take it in. A commit whose split reads only bz2 writes no figures."""
    from dumpslice import gz_path
    from release import RELEASE_DUMP_GZ_BYTES

    ask = (f"import sys; sys.path.insert(0, {str(src)!r}); from wikidata import dump; "
           "print(*dump.decoder(sys.argv[1])) if 'gz' in getattr(dump, 'DECODERS', {}) else None")
    cmds = {fmt: subprocess.run([sys.executable, "-c", ask, fmt], capture_output=True, text=True,
                                check=True).stdout.split() for fmt in ("bz2", "gz")}
    res = {}
    if cmds["gz"]:
        with open(out.with_suffix(".log"), "w") as log:
            res = split_stage(src, files, workers, out.parent, log, fmt="gz")
        res["gz_split_eta"] = round(res["gz_split_cpu_total"] / (res["gz_split_source_mb"] * 1e6)
                                    * RELEASE_DUMP_GZ_BYTES / workers)
        res["gz_decoder"] = Path(cmds["gz"][0]).name
        bz2_slice = Path(os.environ.get("BENCH_DUMP_SLICE") or out.parent / "dump-slice.json.bz2")
        for fmt, path in (("bz2", bz2_slice), ("gz", gz_path(bz2_slice))):
            res[f"{fmt}_decode_wall"], res[f"{fmt}_decode_cpu"] = decode(cmds[fmt], path)
    out.write_text(json.dumps(res))


def tree_rss_mb(pid: int) -> int:
    """The RSS of `pid` and its children."""
    total = kb("VmRSS", str(pid))
    for task in Path(f"/proc/{pid}/task").iterdir():
        for child in (task / "children").read_text().split():
            try:
                total += kb("VmRSS", child)
            except (OSError, StopIteration):
                pass
    return total // 1024


def main() -> None:
    src, out = sys.argv[1], Path(sys.argv[2]).resolve()
    per, workers, group_size, n = map(int, sys.argv[3:7])
    files = [str(Path(f).resolve()) for f in sys.argv[7:]]
    run_dir = out.with_suffix(".run")
    shutil.rmtree(run_dir, ignore_errors=True)
    work_dir = run_dir / "releases" / "20261012"
    data, state, results = work_dir / "data", work_dir / "state", work_dir / "results"
    for d in (data, state, results):
        d.mkdir(parents=True)
    for i in range(n):
        (data / f"chunk_{i}.parquet").symlink_to(files[i % len(files)])
    sizes = [os.path.getsize(files[i % len(files)]) for i in range(n)]  # process() deletes its source
    os.chdir(run_dir)
    os.environ["WIKIDATA_FINALISE_WORKERS"] = str(workers)
    # Where the time goes (phases.py), in this process and every worker
    phases_log = out.with_suffix(".phases.jsonl")
    phases_log.unlink(missing_ok=True)
    os.environ["WIKIDATA_PHASES"] = str(phases_log)
    os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [str(HERE / "phases_site"),
                                                            os.environ.get("PYTHONPATH")]))
    phases.install()
    sys.path.insert(0, str(Path(src).resolve()))
    from wikidata import compact, sort_by_id
    from wikidata.config import AUDIT_DIR, COMPACT_DIR, HUB_COPY_DIR, STAGING_DIR, Table
    from wikidata.pool import process_chunks
    from wikidata.push.core import merge_group
    from wikidata.push.groups import Group, record_partitioned, record_stage

    keep = out.with_suffix(".keep") if os.environ.get("KEEP") else None
    if keep:
        shutil.rmtree(keep, ignore_errors=True)
    log = str(out.with_suffix(".chunks.jsonl"))
    Path(log).unlink(missing_ok=True)
    samples, stop, closes = [], threading.Event(), []

    def sample() -> None:
        t0 = time.perf_counter()
        while not stop.wait(0.2):
            samples.append((round(time.perf_counter() - t0, 2), tree_rss_mb(os.getpid())))

    def close(chunks: list[int]) -> None:
        """close_group's local stages: merge, then the staging dir's clean-up."""
        group = Group(chunks[0], chunks[-1], n - 1)
        t, c = time.perf_counter(), time.process_time()
        record_stage(state.resolve(), group, "closed")
        merge_group(group, AUDIT_DIR, STAGING_DIR)
        record_stage(state.resolve(), group, "merged")
        merged = time.perf_counter()
        for table in Table:
            staged = STAGING_DIR / table
            if not staged.exists():
                continue
            if keep:  # linked, not copied: the same files go on to finalise
                (keep / "staging" / group.name).mkdir(parents=True, exist_ok=True)
                shutil.copytree(staged, keep / "staging" / group.name / table, copy_function=os.link)
            # The group's files where finalise's compaction downloads them from the Hub
            for f in staged.rglob("*.parquet"):
                to = COMPACT_DIR / "src" / table / f.relative_to(staged)
                to.parent.mkdir(parents=True, exist_ok=True)
                f.rename(to)
            shutil.rmtree(staged, ignore_errors=True)
        closes.append({"chunks": [chunks[0], chunks[-1]], "start": t - t_start,
                       "merge_wall": merged - t, "wall": time.perf_counter() - t,
                       "cpu_process": time.process_time() - c})

    # The split first, outside the pipeline's wall, CPU and memory figures
    with open(out.with_suffix(".split.log"), "w") as split_log:
        split = split_stage(Path(src).resolve(), sorted(set(files)), workers, out.parent, split_log)
    threading.Thread(target=sample, daemon=True).start()
    finished = {}
    t_start = time.perf_counter()
    usage0 = [resource.getrusage(w) for w in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN)]

    def done(c: int) -> None:
        finished[c] = time.perf_counter() - t_start
        record_partitioned(state.resolve(), c, sizes[c])

    phases.marker("pipeline")
    process_chunks(
        list(range(n)), [], workers=workers, chunks_per_worker=per, start=lambda c: None,
        work=partial(bench_work, log=log, data_dir=data.resolve(), output_dir=results.resolve(),
                     repo_id="unused", state_dir=state.resolve()),
        done=done, ready=lambda g: len(g) >= group_size, close=close,
    )
    wall = time.perf_counter() - t_start
    phases.marker("pipeline end")
    usage = [resource.getrusage(w) for w in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN)]
    stop.set()
    fin, eta = finalise_at_scale(compact, sort_by_id, Table, COMPACT_DIR, HUB_COPY_DIR, state.resolve(),
                                 n, files, wall, workers, keep)
    # The split at the release's size: its CPU time per byte of the dump's bz2, scaled to
    # the release's dump over the cores (lbzip2 and the split's workers share them)
    from release import RELEASE_DUMP_BYTES
    eta["split_eta"] = round(split["split_cpu_total"] / (split["split_source_mb"] * 1e6) * RELEASE_DUMP_BYTES / workers)
    eta["eta"] += eta["split_eta"]
    chunks = [json.loads(l) for l in open(log)]
    res = {
        "per": per, "workers": workers, "group": group_size, "n": n, "wall": round(wall, 2),
        "last_chunk_done": round(max(finished.values()), 2),
        "peak_sum_rss_mb": max(s for _, s in samples),
        "mean_sum_rss_mb": round(sum(s for _, s in samples) / len(samples)),
        "cpu_total": round(sum(u.ru_utime + u.ru_stime - v.ru_utime - v.ru_stime
                               for u, v in zip(usage, usage0)), 1),
        "chunk_cpu": round(sum(c["cpu"] for c in chunks), 1),
        "process_cpu": round(sum(c["process_cpu"] for c in chunks), 1),
        "partition_cpu": round(sum(c["partition_cpu"] for c in chunks), 1),
        "worker_peak_mb": max(c["hwm_mb"] for c in chunks),
        **fin, "total_wall": round(wall + fin["finalise_wall"], 2), **eta, **split,
        "closes": closes, "chunks": chunks, "samples": samples,
        "phases": phases.summarise(str(phases_log), {"split": ("split", "route end"),
                                                     "pipeline": ("pipeline", "pipeline end"),
                                                     "finalise": ("finalise", "finalise end"),
                                                     "finalise_x3": ("finalise x3", "finalise x3 end")}),
    }
    if keep:
        for d in ("audit", "quarantine"):
            if (work_dir / d).exists():
                shutil.copytree(work_dir / d, keep / d)
    out.write_text(json.dumps(res))
    shutil.rmtree(run_dir)
    cl = " ".join(f"{c['wall']:.1f}s@{c['start']:.0f}" for c in closes)
    print(f"wall {wall:.1f}s (last chunk {res['last_chunk_done']:.1f}s) | CPU chunks {res['chunk_cpu']}s "
          f"(process {res['process_cpu']} partition {res['partition_cpu']}) | closes {cl} "
          f"(CPU {sum(c['cpu_process'] for c in closes):.1f}s) | RSS peak {res['peak_sum_rss_mb']} MB "
          f"mean {res['mean_sum_rss_mb']} MB worker peak {res['worker_peak_mb']} MB", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "--split-gz":
        split_gz(Path(sys.argv[2]).resolve(), Path(sys.argv[3]).resolve(), int(sys.argv[4]),
                 sorted({str(Path(f).resolve()) for f in sys.argv[5:]}))
    else:
        main()
