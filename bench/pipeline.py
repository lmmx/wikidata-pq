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
"""

import json
import os
import re
import resource
import shutil
import sys
import threading
import time
from functools import partial
from pathlib import Path

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
    from wikidata._phases import marker
    marker("x1")
    fin = finalise(compact, sort_by_id, Table, compact_dir, hub_copy_dir, state)
    marker("x1 end")
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
    marker("x3")
    fin3 = finalise(compact, sort_by_id, Table, compact_dir, hub_copy_dir, state3)
    marker("x3 end")

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

    threading.Thread(target=sample, daemon=True).start()
    finished = {}
    t_start = time.perf_counter()
    usage0 = [resource.getrusage(w) for w in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN)]

    def done(c: int) -> None:
        finished[c] = time.perf_counter() - t_start
        record_partitioned(state.resolve(), c, sizes[c])

    process_chunks(
        list(range(n)), [], workers=workers, chunks_per_worker=per, start=lambda c: None,
        work=partial(bench_work, log=log, data_dir=data.resolve(), output_dir=results.resolve(),
                     repo_id="unused", state_dir=state.resolve()),
        done=done, ready=lambda g: len(g) >= group_size, close=close,
    )
    wall = time.perf_counter() - t_start
    usage = [resource.getrusage(w) for w in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN)]
    stop.set()
    fin, eta = finalise_at_scale(compact, sort_by_id, Table, COMPACT_DIR, HUB_COPY_DIR, state.resolve(),
                                 n, files, wall, workers, keep)
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
        **fin, "total_wall": round(wall + fin["finalise_wall"], 2), **eta,
        "closes": closes, "chunks": chunks, "samples": samples,
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
    main()
