"""wikidata-pq's chunk pipeline as `run` drives it, upload aside: pool.process_chunks with
the real process_and_partition per chunk (`workers` at once, `per` chunks a worker), and
groups of `group` chunks closed in the background as close_group does it, merge included,
without the upload and its verification (push_group, verify_group). The chunks are the
split's own parts (split_stage, run first on a dump slice of the given chunk files): the
main set's then the scholarly set's, each with its claims partition file where the split
wrote one, cycled as the files would be (bench 12.0.0).

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
import timeline

HERE = Path(__file__).resolve().parent

os.environ["WIKIDATA_RELEASE"] = "20261012"
# finalise runs its local stages only: huggingface_hub refuses the network, and every Hub
# entry point of the finalise modules is replaced with one that raises (main)
os.environ["HF_HUB_OFFLINE"] = "1"


def kb(field: str, pid: str = "self") -> int:
    return next(int(l.split()[1]) for l in open(f"/proc/{pid}/status") if l.startswith(field))


def bench_work(chunk: int, log: str, **kw) -> None:
    from wikidata.main import partition_chunk, process

    t0, t, r = time.time(), time.perf_counter(), resource.getrusage(resource.RUSAGE_SELF)
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
            # (when it ran, and its process's CPU time at its start: a worker's first chunk's
            # is its start, the interpreter and its imports, wikidata.main's among them)
            "t0": round(t0, 4), "t": round(time.time(), 4), "pcpu0": round(r.ru_utime + r.ru_stime, 4),
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


def claims_refs(state: Path, sources: list[Path], copies: int) -> dict:
    """claims_labels' refs stage (collect_refs_stage), as a release's finalise runs it once
    the claims are sorted, timed apart from finalise's figures (bench 13.1.0): its wall
    time and CPU time (this process and its workers). A commit whose claims partition files
    have refs sidecars (claims_refs.py) has them linked into the run's work directory
    first: each part's (the split's, or the pipeline's for a part left as JSON) `copies`
    times, as the part's rows are in the sorted claims, under names of their own (a
    sidecar's name is its part's content, and the union reads every file there). Empty
    for a commit without the stage."""
    from wikidata import claims_labels, config
    if not hasattr(claims_labels, "collect_refs_stage"):
        return {}
    d = config.WORK_DIR / "claims_refs"
    shutil.rmtree(d, ignore_errors=True)
    for p in sources:
        d.mkdir(parents=True, exist_ok=True)
        for j in range(copies):
            os.link(p, d / f"{j}-{p.name}")
    usage = lambda: sum(u.ru_utime + u.ru_stime for u in map(resource.getrusage, (resource.RUSAGE_SELF,
                                                                               resource.RUSAGE_CHILDREN)))
    t, c = time.perf_counter(), usage()
    claims_labels.collect_refs_stage(state)
    getattr(sys.modules.get("wikidata.parallel"), "shutdown_shared", lambda: None)()
    out = {"wall": round(time.perf_counter() - t, 2), "cpu": round(usage() - c, 1)}
    shutil.rmtree(claims_labels.BUILD_DIR, ignore_errors=True)
    shutil.rmtree(d, ignore_errors=True)
    return out


def flushes() -> int:
    """The bucket flushes written so far (sort_by_id._ipc_stream calls, from phases.py's
    log): one IPC stream each."""
    path = os.environ.get("WIKIDATA_PHASES")
    if not path or not os.path.exists(path):
        return 0
    with open(path) as f:
        return sum('"sort_by_id._ipc_stream"' in line for line in f)


def sort_shape(sort_dir: Path, hub_copy_dir: Path, flushed: int, suffix: str = "") -> dict:
    """The claims key's bucket path as this finalise run took it, measured: its compacted
    source files and its part files (a bucketing job each, and a packing job each), its
    buckets and rows a bucket (their mean, and the largest bucket's), the share of its
    rows spilled to fragments rather than read from the sources directly (every row,
    before 2026-10-25's direct reads), and the rows a spill flush (its rows spilled over `flushed`, the run's flushes: the other keys over
    the scaled threshold, labels/en and nl, are a bucket each, read directly)."""
    bdir = sort_dir / "buckets" / "claims" / "all"
    if not (bdir / "buckets.json").exists():
        return {f"sort_{k}{suffix}": None for k in ("sources", "parts", "buckets", "bucket_rows", "bucket_rows_max",
                                                    "spilled_share", "flush_rows")}
    made = json.loads((bdir / "buckets.json").read_text())
    rows = made["rows"]
    direct = sum(c for line in (bdir / "bucketed.jsonl").read_text().splitlines() if line
                 for g in json.loads(line).get("row_groups", [])
                 for b, c in zip(g["buckets"], g["rows"]) if b in g["direct"])
    spilled = sum(rows) - direct
    return {f"sort_sources{suffix}": len(made["sources"]),
            f"sort_parts{suffix}": len(list((hub_copy_dir / "claims" / "all").glob("*.parquet"))),
            f"sort_buckets{suffix}": len(rows), f"sort_bucket_rows{suffix}": round(sum(rows) / len(rows)),
            f"sort_bucket_rows_max{suffix}": max(rows),
            f"sort_spilled_share{suffix}": round(spilled / sum(rows), 4),
            f"sort_flush_rows{suffix}": round(spilled / flushed) if flushed else None}


def finalise_at_scale(compact, sort_by_id, Table, compact_dir: Path, hub_copy_dir: Path, state: Path,
                      n: int, files: list[str], wall: float, workers: int,
                      keep: Path | None = None, refs: tuple[list[Path], int] = ([], 1)) -> tuple[dict, dict]:
    """finalise as the release would run it, and its time at the release's size.

    The sample has the release's keys (its languages and sites) with a fraction of their
    rows, so a key that would be over SORT_IN_MEMORY_BYTES in the release is sorted through
    buckets here too: the threshold is scaled by the sample's fraction of the release's
    entities (the bucket size is not, so its per-bucket costs stand as in the release;
    what the bucket path's shape is here against a release's is in CHANGELOG 7.4.0, and
    each run records it, sort_shape). Then finalise runs again on the groups' files three
    times over, as three times the groups: the same keys with three times the rows, each
    sorted the way the first run sorted it (the threshold tripled as well), the claims
    key in as many compacted and part files as a release's for its bytes (CHANGELOG 8.0.0).

    Of the two, finalise's time at the release's size: a fixed part (per key: the same in
    the release, its keys being the sample's) and a part per byte of source Parquet,
    scaled to the release's bytes. The part per byte is the CPU time the extra rows took,
    over the workers (at the release's size every worker is busy); the fixed part is the
    rest of the first run's wall time. Returns the first run's figures (its outputs are
    the parity's) and the estimates, the pipeline's scaled by entities as before."""
    import pyarrow.parquet as pq
    from release import (RELEASE_CLAIMS_BYTES, RELEASE_CLAIMS_FILES, RELEASE_COMPACT_FILE_BYTES,
                         RELEASE_ENTITIES)
    from wikidata import config
    from wikidata.push.groups import chunk_range_name

    entities = sum(pq.read_metadata(files[i % len(files)]).num_rows for i in range(n))
    scale = RELEASE_ENTITIES / entities
    sort_by_id.SORT_IN_MEMORY_BYTES = int(config.SORT_IN_MEMORY_BYTES / scale)
    snap = compact_dir.parent / "compact-src"
    shutil.copytree(compact_dir / "src", snap, copy_function=os.link)
    parts = sorted(snap.rglob("*.parquet"))
    source = sum(p.stat().st_size for p in parts)
    flushed = flushes()
    phases.marker("finalise")
    fin = finalise(compact, sort_by_id, Table, compact_dir, hub_copy_dir, state)
    phases.marker("finalise end")
    shape = sort_shape(compact_dir / "sort", hub_copy_dir, flushes() - flushed)
    if keep:  # the first run's sorted part files, what a release publishes (linked)
        for p in sorted(hub_copy_dir.rglob("*.parquet")):
            to = keep / "final" / p.relative_to(hub_copy_dir)
            to.parent.mkdir(parents=True, exist_ok=True)
            os.link(p, to)
    # claims_labels' refs from the sorted claims, apart from finalise's figures; the
    # sidecars the pipeline wrote (a part left as JSON) taken with the split's
    from wikidata import config
    refs_src = compact_dir.parent / "claims-refs-src"
    refs_src.mkdir()
    for p in [*refs[0], *sorted((config.WORK_DIR / "claims_refs").glob("*.parquet"))]:
        os.link(p, refs_src / f"{len(list(refs_src.iterdir()))}-{p.name}")
    sidecars = sorted(refs_src.glob("*.parquet"))
    phases.marker("claims refs")
    refs1 = claims_refs(state, sidecars, refs[1])
    phases.marker("claims refs end")

    # Three times over: the groups' files linked as three times as many groups' (their ids
    # repeated), group i's file of a key as groups i, i + groups and i + 2 groups' (chunks
    # first + j n to last + j n)
    shutil.rmtree(compact_dir)
    shutil.rmtree(hub_copy_dir, ignore_errors=True)
    for p in parts:
        key_dir = compact_dir / "src" / p.relative_to(snap).parent
        key_dir.mkdir(parents=True, exist_ok=True)
        first, last = map(int, re.fullmatch(r"chunks-(\d+)-(\d+)\.parquet", p.name).groups())
        for j in range(3):
            os.link(p, key_dir / f"{chunk_range_name(first + j * n, last + j * n, 3 * n - 1)}.parquet")
    # The claims key's files as a release's for its bytes (CHANGELOG 8.0.0): a release's
    # claims group file is larger than COMPACT_FILE_BYTES (44.7 GB in 30, against 500 MiB),
    # so each is a compacted file of its own (linked, not glued), each a bucketing job, and
    # its part files are COMPACT_FILE_BYTES each, about 2.9 a group file. Here the target
    # is set in that ratio to the claims group files' mean size: each a compacted file, and
    # the part files as many for their bytes as a release's, up to one a bucket (a part
    # file is whole buckets). In this process (compaction's and packing's plans), and in
    # every worker started for this run (config, from the environment). Every other key's
    # group files are a few MB, their three times far under the target: compacted and
    # sorted into files as before.
    claims = [p for p in parts if p.relative_to(snap).parts[:2] == ("claims", "all")]
    claims_source = sum(p.stat().st_size for p in claims)  # (the claims key's share of `source`)
    if claims:
        target = int(sum(p.stat().st_size for p in claims) / len(claims)
                     * RELEASE_COMPACT_FILE_BYTES * RELEASE_CLAIMS_FILES / RELEASE_CLAIMS_BYTES)
        os.environ["WIKIDATA_COMPACT_FILE_BYTES"] = str(target)
        compact.COMPACT_FILE_BYTES = sort_by_id.COMPACT_FILE_BYTES = target
        shape["x3_compact_file_mb"] = round(target / 1e6, 1)
        # (and the largest other key at three times its rows, which must stay under it)
        others: dict[Path, int] = {}
        for p in parts:
            if p not in claims:
                others[p.parent] = others.get(p.parent, 0) + 3 * p.stat().st_size
        shape["x3_other_key_max_mb"] = round(max(others.values(), default=0) / 1e6, 1)
    else:
        os.environ["WIKIDATA_COMPACT_FILE_BYTES"] = str(config.COMPACT_FILE_BYTES)
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
    flushed = flushes()
    phases.marker("finalise x3")
    fin3 = finalise(compact, sort_by_id, Table, compact_dir, hub_copy_dir, state3)
    phases.marker("finalise x3 end")
    phases.marker("claims refs x3")
    refs3 = claims_refs(state3, sidecars, 3 * refs[1])
    phases.marker("claims refs x3 end")
    shutil.rmtree(refs_src)
    os.environ.pop("WIKIDATA_COMPACT_FILE_BYTES")
    shape |= sort_shape(compact_dir / "sort", hub_copy_dir, flushes() - flushed, "_x3")

    # (bench 8.0.0's model, every worker's start in the part per byte; release_model's
    # takes the claims key's apart, once each worker's CPU time outside its jobs is known)
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
        "eta": round(pipeline_eta + finalise_eta), **shape,
        **({"refs_wall": refs1["wall"], "refs_cpu": refs1["cpu"], "refs_x3_wall": refs3["wall"],
            "refs_x3_cpu": refs3["cpu"]} if refs1 and refs3 else {}),
        "_model": {"source": source, "scale": scale, "workers": workers, "wall": fin["finalise_wall"],
                   "cpu": fin["finalise_cpu"], "cpu_x3": fin3["finalise_cpu"], "per_byte": per_byte,
                   "claims_source": claims_source},
    }


def release_model(eta: dict, lines: dict) -> dict:
    """finalise at a release's size, with its workers' starts as a release has them, and
    its peak RSS at a release's buckets: modelled (CHANGELOG 9.0.0).

    The ETA. A finalise job that holds much memory runs in a worker of its own
    (parallel.in_parallel, tasks_per_child=1), which starts by importing Polars, pyarrow
    and the package, about a second of CPU. The claims key's bucket path is such jobs
    (a bucketing job a source file, a bucket sort a bucket, a packing job a part file),
    and the run on three times the rows has more of them than the first, so bench
    8.0.0's part per byte carried their starts, scaled to the release's bytes: about ten
    times the release's (CHANGELOG 9.0.0). Here every worker's CPU time outside its jobs
    (its start, its imports included, and its end, or all of it for a worker with no
    job; timeline._workers, from the workers themselves) is taken out of each run's CPU
    time before the part per byte is taken, and the release's claims workers (release.py:
    the main set's measured, the scholarly set's modelled, and its pools' workers started
    with no job) are added back at the mean CPU time a claims worker took outside its
    jobs here. Every other key's workers stay in the fixed part, as before: per key, as
    the release's (its keys, each sorted the same way). The first run's claims workers
    are taken out of its fixed part, the release's count standing for them.

    The peak RSS. The largest bucket-sort worker's own peak (VmHWM at its exit) against
    the largest bucket's rows, in the two runs: the line through them (or, for runs
    under 20% apart in rows, the larger in proportion to its rows) at a release's rows a
    bucket (its rows, modelled, over its buckets, measured: release.py), the bench's
    largest bucket over its mean as the release's; and the process tree's at
    RELEASE_FINALISE_WORKERS such workers at their peak at once, with the pipeline's
    process's own peak over the bucket sorts of three times the rows. A worker's RSS is
    taken to grow with its bucket's rows as here, the release's rows the bench's in
    width; summed RSS counts pages shared between processes once in each, as the
    measured figures do. None where a figure is missing."""
    from release import (RELEASE_CLAIMS_BUCKETS, RELEASE_CLAIMS_ROWS, RELEASE_CLAIMS_WORKERS,
                         RELEASE_FINALISE_WORKERS)
    m = eta["_model"]  # (claims_model takes it after)
    out = {}
    runs = [lines.get(span, {}).get("processes") for span in ("finalise", "finalise_x3")]

    def outside(kinds: list[dict]) -> float | None:
        """The workers' CPU time outside their jobs, a worker whose exit went unseen at
        the mean of those seen."""
        n, seen = sum(k["workers"] for k in kinds), sum(k["exits_seen"] for k in kinds)
        return sum(k["overhead_cpu"] for k in kinds) * n / seen if seen else (0.0 if not n else None)

    if all(runs):
        claims = [[k for kind, k in w.items() if kind.endswith(" claims")] for w in runs]
        n = [sum(k["workers"] for k in c) for c in claims]
        every = [outside(list(w.values())) for w in runs]  # (every worker: with its jobs, or idle)
        ov = [outside(c) for c in claims]
        if all(n) and None not in every + ov:
            mean = sum(ov) / sum(n)  # CPU-s a claims worker, outside its jobs
            source, scale, workers = m["source"], m["scale"], m["workers"]
            per_byte = max((m["cpu_x3"] - every[1]) - (m["cpu"] - every[0]), 0) / (2 * source)
            # (and a pool's workers started with no job: a pool of `workers` for each of the
            # key's three steps in each set, each importing as a job's worker would, as a
            # release's __mp_main__ is its console script, which imports wikidata.main)
            starts = sum(RELEASE_CLAIMS_WORKERS.values()) + len(RELEASE_CLAIMS_WORKERS) * 3 * workers
            fixed = m["wall"] - (per_byte * source + ov[0]) / workers
            finalise_eta = fixed + (per_byte * source * scale + starts * mean) / workers
            m["per_byte"] = per_byte
            out = {
                "finalise_eta_8": eta["finalise_eta"], "finalise_cpu_per_mb_8": eta["finalise_cpu_per_mb"],
                "finalise_eta": round(finalise_eta), "finalise_cpu_per_mb": round(per_byte * 1e6, 3),
                "finalise_fixed": round(fixed, 2),
                "finalise_x3_wall_fit": round(fixed + (per_byte * 3 * source + ov[1]) / workers, 2),
                "claims_workers": n[0], "claims_workers_x3": n[1], "claims_worker_cpu": round(mean, 3),
                "claims_worker_startup_cpu": round(sum(k["startup_cpu"] for c in claims for k in c)
                                                   / max(sum(n), 1), 3),
                "release_claims_workers": starts,
                "release_claims_workers_eta": round(starts * mean / workers),
                "eta": eta["eta"] + round(finalise_eta) - eta["finalise_eta"],
            }
    sorts = [(w or {}).get("_sort_bucket claims", {}).get("hwm_mb") for w in runs]
    rows = [eta.get("sort_bucket_rows_max"), eta.get("sort_bucket_rows_max_x3")]
    main = lines.get("finalise_x3", {}).get("claims_bucket_sorts", {}).get("main_peak_rss_mb")
    if all(sorts) and all(rows) and eta.get("sort_bucket_rows_x3"):
        at = RELEASE_CLAIMS_ROWS / RELEASE_CLAIMS_BUCKETS * rows[1] / eta["sort_bucket_rows_x3"]
        if rows[1] - rows[0] >= 0.2 * rows[1]:
            worker = sorts[1] + (sorts[1] - sorts[0]) / (rows[1] - rows[0]) * (at - rows[1])
        else:
            worker = sorts[1] * at / rows[1]
        out |= {"bucket_worker_rss_mb": sorts[0], "bucket_worker_rss_mb_x3": sorts[1],
                "release_bucket_rows": round(at), "release_bucket_worker_rss_mb": round(worker)}
        if main:
            out["release_finalise_rss_mb"] = round(main + RELEASE_FINALISE_WORKERS * worker)
    return out


def refs_memory(phases_log: Path, cores_log: Path, main: int, lines: dict) -> dict:
    """claims_labels' refs stage's peak RSS in each run (the process tree's and the largest
    worker's), from the timeline's samples in it; for a stage shorter than the sampler's
    interval, with no sample in it, the first sample after it (bench 13.1.0)."""
    marks = {r["marker"]: r["t"] for r in map(json.loads, open(phases_log)) if "marker" in r}
    samples = [json.loads(line) for line in open(cores_log)] if cores_log.exists() else []
    out = {}
    for span, x in (("claims_refs", ""), ("claims_refs_x3", "_x3")):
        got = lines.get(span, {})
        if "peak_rss_mb" not in got and f"{span.replace('_', ' ')} end" in marks:
            after = [s for s in samples if s["t"] >= marks[f"{span.replace('_', ' ')} end"] and "rss" in s]
            if after:
                got = timeline._memory_peaks(samples, main, after[0]["t"], after[0]["t"])
        out[f"refs_peak_rss_mb{x}"] = got.get("peak_rss_mb")
        out[f"refs_worker_peak_mb{x}"] = got.get("worker_peak_rss_mb")
    return out


def refs_model(eta: dict) -> dict:
    """claims_labels' refs stage at a release's size, modelled as finalise's part per byte
    is (bench 13.1.0): its CPU time on three times the rows less the first run's, a byte of
    the claims key's source, times the release's claims bytes (both sets', claims_model's)
    over the cores, and the first run's wall time less its part per byte as a fixed part.
    And the end-to-end ETA with it, the stage not being in `eta`. Empty where a figure is
    missing."""
    m = eta["_model"]
    if "refs_cpu" not in eta or not eta.get("release_claims_mb") or not m.get("claims_source"):
        return {}
    source, workers = m["claims_source"], m["workers"]
    per_byte = max(eta["refs_x3_cpu"] - eta["refs_cpu"], 0) / (2 * source)
    fixed = max(eta["refs_wall"] - per_byte * source / workers, 0)
    refs_eta = round(fixed + per_byte * eta["release_claims_mb"] * 1e6 / workers)
    return {"refs_eta": refs_eta, "refs_cpu_per_mb": round(per_byte * 1e6, 4), "refs_fixed": round(fixed, 2),
            "eta_refs": eta["eta"] + refs_eta}


# (place_built_claims: a chunk whose claims the split built, its claims file placed)
CLAIMS_CALLS = ("process.normalise_claims_direct", "process.place_built_claims", "partitioning.core.place_unsplit")


def claims_measured(phases_log: Path, chunks: list[dict], sets: list[str], audit: Path,
                    sizes: list[int]) -> dict:
    """The pipeline's claims work in each set, measured (CHANGELOG 10.0.0): each chunk's
    claims CPU time (its normalise_claims_direct and place_unsplit calls, every stage they
    call included: genson's decode, the statements and their write, the label lookup and
    the quarantine), its claims partition's rows and bytes (its audit sidecar) and its
    chunk file's bytes, summed by set. A chunk's set is the one the split routed its part
    to (`sets`, by chunk). A worker's k-th
    claims call is the k-th chunk it finished. Empty where a figure is missing (a commit
    without these functions)."""
    import pyarrow.parquet as pq

    rows = [json.loads(line) for line in open(phases_log)]
    marks = {r["marker"]: r["t"] for r in rows if "marker" in r}
    order: dict[int, list[int]] = {}
    for c in chunks:
        order.setdefault(c["pid"], []).append(c["chunk"])
    seen: dict[tuple, int] = {}
    cpu: dict[int, float] = {}
    for r in rows:
        if r.get("fn") in CLAIMS_CALLS and r.get("cpu") is not None and marks["pipeline"] <= r["t"] <= marks["pipeline end"]:
            k = seen[r["pid"], r["fn"]] = seen.get((r["pid"], r["fn"]), -1) + 1
            done = order.get(r["pid"], [])
            if k >= len(done):
                return {}
            cpu[done[k]] = cpu.get(done[k], 0.0) + r["cpu"]
    out: dict[str, dict] = {}
    for c in sorted({c["chunk"] for c in chunks}):
        side = audit / "claims" / f"chunk_{c}.parquet"
        if c not in cpu or not side.exists():
            return {}
        t = pq.read_table(side, columns=["num_rows", "file_size"])
        s = out.setdefault(sets[c], {"cpu": 0.0, "rows": 0, "bytes": 0, "source": 0, "chunks": 0})
        s["cpu"] += cpu[c]
        s["rows"] += sum(t["num_rows"].to_pylist())
        s["bytes"] += sum(t["file_size"].to_pylist())
        s["source"] += sizes[c]
        s["chunks"] += 1
    return out


def claims_model(eta: dict, claims: dict, tables: dict, wall: float, cpu_total: float) -> dict:
    """The pipeline's and finalise's claims parts scaled to a release by the claims' own
    rows and bytes, not by entities (CHANGELOG 10.0.0). Modelled; bench 9.0.0's figures,
    every part scaled by entities, are kept beside them (`_9`).

    The bench's sample has about 2.8 times a release's claims per entity (its chunk c0,
    a third of its entities, has 62 statements an entity against a release's main set's
    10), so scaling the claims' CPU time by entities overstates it. Here:

    - the pipeline: each set's claims CPU time (claims_measured) times the release's
      claims rows in that set over the sample's (the claims' CPU time goes with their
      rows: over the 11 chunk files, 9.2 µs a row on the main set's, with no fixed part
      a chunk beyond 0.03 s), every other part of the pipeline's CPU time by entities as
      before, and the pipeline ETA its wall time scaled by the release's CPU time so
      reckoned over its own;
    - finalise: its part per byte's claims share (the run on three times the rows less
      the first, by table: phases.table_cpu) times the release's claims bytes over the
      sample's claims key, the rest by entities as before; its fixed part and its claims
      workers' starts as in bench 9.0.0.

    The release's claims (release.py): the main set's bytes measured, its rows modelled
    (780 million); the scholarly set's rows and bytes modelled, its chunk files' bytes
    (measured) times the claims rows and bytes a byte of the sample's scholarly chunk
    files. Empty where a figure is missing (a set not in the sample, say)."""
    from release import RELEASE_CLAIMS_BYTES, RELEASE_CLAIMS_ROWS, RELEASE_SOURCE_BYTES
    m = eta["_model"]
    one, three = tables.get("finalise"), tables.get("finalise_x3")
    if set(claims) != {"main", "scholar"} or not one or not three or not m.get("claims_source"):
        return {}
    sch = claims["scholar"]
    rows = {"main": RELEASE_CLAIMS_ROWS, "scholar": RELEASE_SOURCE_BYTES["scholar"] * sch["rows"] / sch["source"]}
    release_bytes = RELEASE_CLAIMS_BYTES + RELEASE_SOURCE_BYTES["scholar"] * sch["bytes"] / sch["source"]
    claims_cpu = sum(c["cpu"] for c in claims.values())
    release_cpu = (sum(c["cpu"] * rows[k] / c["rows"] for k, c in claims.items())
                   + (cpu_total - claims_cpu) * m["scale"])
    pipeline_eta = wall * release_cpu / cpu_total
    per_byte_all = sum(three.values()) - sum(one.values())
    share = (three.get("claims", 0) - one.get("claims", 0)) / per_byte_all if per_byte_all > 0 else 0.0
    per_byte, source, workers = m["per_byte"], m["source"], m["workers"]
    claims_scale = release_bytes / m["claims_source"]
    finalise_eta = eta["finalise_eta"] + per_byte * source * share * (claims_scale - m["scale"]) / workers
    measured = {f"pipeline_claims_{k}_{s}": round(c[k], 3) if k == "cpu" else c[k]
                for s, c in claims.items() for k in ("cpu", "rows", "bytes", "source")}
    return {
        **measured, "pipeline_claims_cpu": round(claims_cpu, 2), "finalise_claims_share": round(share, 4),
        "release_claims_rows_scholar": round(rows["scholar"]), "release_claims_mb": round(release_bytes / 1e6),
        "release_pipeline_cpu": round(release_cpu),
        "claims_rows_scale_main": round(rows["main"] / claims["main"]["rows"], 1),
        "claims_rows_scale_scholar": round(rows["scholar"] / sch["rows"], 1),
        "claims_bytes_scale": round(claims_scale, 1),
        "pipeline_eta_9": eta["pipeline_eta"], "finalise_eta_9": eta["finalise_eta"], "eta_9": eta["eta"],
        "pipeline_eta": round(pipeline_eta), "finalise_eta": round(finalise_eta),
        "eta": eta["eta"] - eta["pipeline_eta"] - eta["finalise_eta"] + round(pipeline_eta) + round(finalise_eta),
    }


def pool_starts(chunks: int, workers: int, per: int) -> int:
    """The worker processes pool.process_chunks starts for `chunks` chunks, `workers` at a
    time, `per` chunks a worker, each chunk taking as long as the others: it starts a
    worker for a chunk when none is idle, and retires one that has taken `per` chunks, or
    that finishes while the queue is empty. The bench's own count is measured
    (pipeline_workers) and checked against this (`pipeline_workers_counted`)."""
    queue, idle, started = chunks, [], 0  # (idle: each idle worker's chunks taken)
    while queue:
        running = []
        while queue and len(running) < workers:
            queue -= 1
            if idle:
                running.append(idle.pop() + 1)
            else:
                started += 1
                running.append(1)
        idle = [taken for taken in running if taken < per and queue]
    return started


def pipeline_workers(phases_log: Path, chunks: list[dict]) -> dict:
    """The pipeline's chunk workers, from the workers themselves: how many, and each one's
    CPU time outside its chunks (its start, from its exec to its first chunk: the
    interpreter, the spawn, its imports, wikidata.main's among them; and its end), its CPU
    time at its exit (phases.track_process, as its atexit handlers run) less its chunks'
    (bench_work, each chunk's process and partition). A worker whose exit went unseen
    counts at the mean of those seen. Empty without the workers' exits (a phases log
    from before bench 9.0.0)."""
    exits = {}
    for line in open(phases_log):
        if line.startswith('{"proc": "exit"'):
            r = json.loads(line)
            exits[r["pid"]] = r["cpu"]
    mine: dict[int, list[dict]] = {}
    for c in chunks:
        mine.setdefault(c["pid"], []).append(c)
    seen = [pid for pid in mine if pid in exits]
    if not seen:
        return {}
    outside = sum(exits[pid] - sum(c["cpu"] for c in mine[pid]) for pid in seen)
    starts = [min(cs, key=lambda c: c.get("t0", 0)).get("pcpu0") for cs in mine.values()]
    return {"pipeline_workers": len(mine), "pipeline_worker_cpu": round(outside / len(seen), 3),
            "pipeline_worker_startup_cpu": round(sum(starts) / len(starts), 3) if None not in starts else None,
            "_outside": outside * len(mine) / len(seen)}


def starts_model(eta: dict, measured: dict, wall: float, cpu_total: float, workers: int, per: int) -> dict:
    """The pipeline's worker starts as a release has them, not scaled by entities
    (CHANGELOG 11.0.0). Modelled; bench 10.0.0's figures are kept beside them (`_10`).

    pool.process_chunks starts a worker process (spawned: the interpreter and the
    package's imports, about 0.8 CPU-s) for every `per` chunks, so a release's starts go
    with its chunks, not its entities: the bench's 4 chunks start 4 workers, and a
    release's 23,035 chunks (release.RELEASE_CHUNKS, measured) about 2,900 at
    CHUNKS_PER_WORKER = 8 (pool_starts, at the bench's workers: the ETA's cores), where
    scaling every part of the CPU time but the claims' by entities stood for about 14,700.
    Here the workers' CPU time outside their chunks (pipeline_workers, measured) is taken
    out of the part scaled by entities, and the release's starts added back at the mean
    CPU time a worker took outside its chunks on the bench. Everything else as in bench
    10.0.0 (claims_model: the claims by their rows, the rest by entities); the pipeline
    ETA is its wall time scaled by the release's CPU time so reckoned over the sample's.
    Empty where a figure is missing."""
    from release import RELEASE_CHUNKS
    if not measured or measured.get("pipeline_worker_cpu") is None:
        return {}
    base = eta.get("release_pipeline_cpu", cpu_total * eta["scale"])  # (bench 10.0.0's, or 9.0.0's)
    starts = {s: pool_starts(n, workers, per) for s, n in RELEASE_CHUNKS.items()}
    mean = measured["pipeline_worker_cpu"]
    release_cpu = base - measured["_outside"] * eta["scale"] + sum(starts.values()) * mean
    pipeline_eta = wall * release_cpu / cpu_total
    return {
        "pipeline_eta_10": eta["pipeline_eta"], "eta_10": eta["eta"], "release_pipeline_cpu_10": round(base),
        "release_pipeline_cpu": round(release_cpu), "release_pipeline_workers": sum(starts.values()),
        "release_pipeline_workers_cpu": round(sum(starts.values()) * mean),
        "pipeline_workers_cpu_scaled_10": round(measured["_outside"] * eta["scale"]),
        "pipeline_eta": round(pipeline_eta), "eta": eta["eta"] - eta["pipeline_eta"] + round(pipeline_eta),
    }


def tables_measured(phases_log: Path, chunks: list[dict], sets: list[str], audit: Path) -> dict:
    """The pipeline's CPU time by table and set, measured: each chunk's calls (its worker's,
    within its run: bench_work's `t0` and `t`) by table (phases.table_cpu's rule: the table
    a call names, its own or its `dst_dir`'s, or that of the innermost call enclosing it,
    the claims' calls naming the claims; `other` else), and each table's rows and
    partition files (its audit sidecars), summed by set. Measurement only: the release
    ETA does not weight the other tables by it (CHANGELOG 11.0.0 says why). Empty for
    chunk logs without `t0` (before bench 11.0.0)."""
    import pyarrow.parquet as pq

    if not chunks or "t0" not in chunks[0]:
        return {}
    calls: dict[int, list[dict]] = {}
    for line in open(phases_log):
        if line.startswith('{"fn"'):
            r = json.loads(line)
            calls.setdefault(r["pid"], []).append(r)
    out: dict[str, dict] = {}
    for c in chunks:
        s = out.setdefault(sets[c["chunk"]], {})
        inside = [r for r in calls.get(c["pid"], []) if c["t0"] <= r["t0"] and r["t"] <= c["t"] + 0.001]
        for t, v in phases.tables_of(inside).items():
            s.setdefault(t, {"cpu": 0.0, "rows": 0, "files": 0})["cpu"] += v
        for side in audit.glob(f"*/chunk_{c['chunk']}.parquet"):
            rows = pq.read_table(side, columns=["num_rows"])["num_rows"].to_pylist()
            k = s.setdefault(side.parent.name, {"cpu": 0.0, "rows": 0, "files": 0})
            k["rows"] += sum(rows)
            k["files"] += len(rows)
    return {s: {t: {**v, "cpu": round(v["cpu"], 3)} for t, v in sorted(by.items())} for s, by in out.items()}


def split_stage(src: Path, files: list[str], workers: int, work: Path, log, fmt: str = "bz2",
                form: str = "dump", keep: Path | None = None) -> dict:
    """The release's first stage, from the dump to chunk files: split-dump, then
    route-release (dump.split, dump.route), each run as the CLI runs it, in a process of
    its own in a directory of its own, on a slice of a dump made of the sample's own
    entities (dumpslice.py, made once in `work`, untimed), in `fmt`: bz2, or gz with
    WIKIDATA_DUMP_FORMAT=gz (its figures named `gz_...`); its lines escaped as the dumps
    escape theirs, or with `form` "orjson" in orjson's form, bench 10.0.0's slice, bz2 (its
    figures named `orjson_...`). Returns each step's wall and
    CPU time (its process and every process it waits for: the decoder, the workers), the
    slice's size, whether the chunks it made hold exactly the sample's rows (`split_parity`),
    and a digest of the manifests it wrote (which split from bz2 and from gz must share).

    With `keep`, the parts it made stay there (linked), each set's in its manifest's order,
    for the pipeline to run on, listed in `_parts` as (set, part file, claims file or None)."""
    from dumpslice import gz_path, orjson_path

    release = os.environ["WIKIDATA_RELEASE"]
    p = "orjson_" if form == "orjson" else "" if fmt == "bz2" else f"{fmt}_"
    # (BENCH_DUMP_SLICE: where CI caches it between jobs, by the data and dumpslice.py)
    bz2_slice = Path(os.environ.get("BENCH_DUMP_SLICE") or work / "dump-slice.json.bz2")
    dump_slice = orjson_path(bz2_slice) if form == "orjson" else bz2_slice if fmt == "bz2" else gz_path(bz2_slice)
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
        sets = [Path(tmp) / "releases" / release / "data", Path(tmp) / "releases" / f"{release}-scholar" / "data"]
        out[f"{p}split_parity_ok"] = split_parity(src, sets, files, Path(tmp))
        if keep is not None:
            out["_parts"] = keep_parts(sets, keep)
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


def set_parts(d: Path) -> list[tuple[Path, Path | None]]:
    """A set's parts in its manifest's order: each chunk file, and its claims partition file
    where the split wrote one (a commit whose split builds its parts' claims)."""
    entries = [json.loads(line) for line in (d / "manifest.jsonl").read_text().splitlines() if line]
    return [(d / e["file"], d / "claims" / e["file"] if (d / "claims" / e["file"]).exists() else None)
            for e in entries]


def split_parity(src: Path, sets: list[Path], files: list[str], tmp: Path) -> bool:
    """Whether the split's parts hold exactly the sample's rows: the scholarly works (the
    `s` files) in the scholarly set and the rest in the main one, every column as the
    sample has it. A part whose claims the split built has no claims column, and its
    claims partition file must be the one the pipeline's claims call
    (process.normalise_claims_direct) makes of the sample's claims of those rows: the same
    rows in the same order, schema (with its metadata) and row groups' rows (bench 13.0.0;
    12.0.0 compared their bytes)."""
    import polars as pl

    built = []
    for i, d in enumerate(sets):
        want = pl.concat([pl.read_parquet(f) for f in files if Path(f).name.startswith("s") == bool(i)])
        parts = set_parts(d)
        made = pl.concat([pl.read_parquet(f) for f, _ in parts], how="diagonal_relaxed")
        if not made.drop("claims", strict=False).sort("id").equals(want.drop("claims").sort("id")):
            return False
        for f, claims in parts:
            part = pl.read_parquet(f)
            sample = part.select("id").join(want.select("id", "claims"), on="id", how="left", maintain_order="left")
            if claims is None:
                if not part.select("id", "claims").equals(sample):
                    return False
                continue
            if "claims" in part.columns:
                return False
            source = tmp / "parity" / f"{d.parent.name}-{f.name}"
            source.parent.mkdir(exist_ok=True)
            sample.write_parquet(source)
            built.append((source, claims))
    return not built or claims_parity(src, built, tmp / "parity")


# The commit's own claims call on each sample (process.normalise_claims_direct), its file
# compared with the split's
CLAIMS_PARITY = """
import sys
sys.path.insert(0, sys.argv[1])
from pathlib import Path
import polars as pl
import pyarrow.parquet as pq
from wikidata import process
work = Path(sys.argv[2])
for source, claims in zip(sys.argv[3::2], sys.argv[4::2]):
    source, claims = Path(source), Path(claims)
    made = process.normalise_claims_direct(source, work / "claims" / source.name, work / "lookup.parquet",
                                           work / "q.parquet", pl.read_parquet(source, columns=["id"]).height)
    a, b = pq.ParquetFile(made), pq.ParquetFile(claims)
    groups = lambda f: [f.metadata.row_group(i).num_rows for i in range(f.metadata.num_row_groups)]
    ta, tb = a.read(), b.read()
    if not (ta.schema.equals(tb.schema, check_metadata=True) and ta.equals(tb) and groups(a) == groups(b)):
        sys.exit(1)
"""


def claims_parity(src: Path, built: list[tuple[Path, Path]], work: Path) -> bool:
    """Whether each claims partition file is the one the commit's own claims call makes of
    its sample (id, claims as JSON): its rows, their order, its schema and its row groups'
    rows, not its bytes (a split may write it a row group at a time, each encoded on its
    own). Run in a process of its own, outside every figure (the phases log and this
    process's memory)."""
    env = {k: v for k, v in os.environ.items() if k != "WIKIDATA_PHASES"}
    args = [str(x) for pair in built for x in pair]
    ok = subprocess.run([sys.executable, "-c", CLAIMS_PARITY, str(src), str(work), *args], env=env).returncode == 0
    shutil.rmtree(work)
    return ok


def keep_parts(sets: list[Path], keep: Path) -> list[tuple[str, str, str | None]]:
    """The split's parts linked into `keep` (to outlive its directory), the main set's then
    the scholarly set's, each in its manifest's order: (set, part file, claims file)."""
    shutil.rmtree(keep, ignore_errors=True)
    kept = []
    for name, d in zip(("main", "scholar"), sets):
        (keep / name / "claims").mkdir(parents=True)
        for f, claims in set_parts(d):
            os.link(f, keep / name / f.name)
            if claims is not None:
                os.link(claims, keep / name / "claims" / f.name)
            kept.append((name, str(keep / name / f.name),
                         None if claims is None else str(keep / name / "claims" / f.name)))
        # (and the split's refs sidecars of the set's claims files, in its work directory:
        # claims_refs.py, bench 13.1.0)
        for f in sorted((d.parent / "claims_refs").glob("*.parquet")):
            (keep / name / "claims_refs").mkdir(exist_ok=True)
            os.link(f, keep / name / "claims_refs" / f.name)
    return kept


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
    among them) take it in. A commit whose split reads only bz2 writes no gz figures.
    And, likewise apart, the split from the bz2 on bench 10.0.0's slice (its lines in
    orjson's form) and its ETA, `orjson_...`."""
    from dumpslice import gz_path
    from release import RELEASE_DUMP_BYTES, RELEASE_DUMP_GZ_BYTES

    ask = (f"import sys; sys.path.insert(0, {str(src)!r}); from wikidata import dump; "
           "print(*dump.decoder(sys.argv[1])) if 'gz' in getattr(dump, 'DECODERS', {}) else None")
    cmds = {fmt: subprocess.run([sys.executable, "-c", ask, fmt], capture_output=True, text=True,
                                check=True).stdout.split() for fmt in ("bz2", "gz")}
    res = {}
    # The split on bench 10.0.0's slice, its lines in orjson's form (dumpslice.orjson_path),
    # and its ETA as bench 10.0.0 made it, so the change to the dumps' escaping shows on the
    # same runs
    with open(out.with_name(out.name.removesuffix(".json") + "-orjson.log"), "w") as log:
        old = split_stage(src, files, workers, out.parent, log, form="orjson")
    res |= {k: v for k, v in old.items() if k != "orjson_split_manifests_sha256"}
    res["orjson_split_eta"] = round(old["orjson_split_cpu_total"] / (old["orjson_split_source_mb"] * 1e6)
                                    * RELEASE_DUMP_BYTES / workers)
    if cmds["gz"]:
        with open(out.with_suffix(".log"), "w") as log:
            res |= split_stage(src, files, workers, out.parent, log, fmt="gz")
        res["gz_split_eta"] = round(res["gz_split_cpu_total"] / (res["gz_split_source_mb"] * 1e6)
                                    * RELEASE_DUMP_GZ_BYTES / workers)
        res["gz_decoder"] = Path(cmds["gz"][0]).name
        bz2_slice = Path(os.environ.get("BENCH_DUMP_SLICE") or out.parent / "dump-slice.json.bz2")
        for fmt, path in (("bz2", bz2_slice), ("gz", gz_path(bz2_slice))):
            res[f"{fmt}_decode_wall"], res[f"{fmt}_decode_cpu"] = decode(cmds[fmt], path)
    out.write_text(json.dumps(res))


def tree_rss_mb(pid: int) -> int:
    """The RSS of `pid` and its children. A thread or child that ends between its listing
    and its read (its /proc entry gone) is left out, as gone."""
    total = kb("VmRSS", str(pid))
    for task in Path(f"/proc/{pid}/task").iterdir():
        try:
            children = (task / "children").read_text().split()
        except OSError:  # the thread has exited
            continue
        for child in children:
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

    # The split first, outside the pipeline's wall, CPU and memory figures; the pipeline
    # runs on its parts (bench 12.0.0): each set's, in order, the main set's first, cycled
    # as the files are (the full tier's 66 chunks are the 11 files 6 times over, so its
    # parts 6 times over)
    kept = out.with_suffix(".parts")
    with open(out.with_suffix(".split.log"), "w") as split_log:
        split = split_stage(Path(src).resolve(), sorted(set(files)), workers, out.parent, split_log, keep=kept)
    parts = split.pop("_parts")
    cycles = max(1, n // len(set(files)))
    chunk_parts = [parts[i % len(parts)] for i in range(cycles * len(parts))]
    n = len(chunk_parts)
    (data / "claims").mkdir()
    for i, (_, part, claims) in enumerate(chunk_parts):
        os.link(part, data / f"chunk_{i}.parquet")
        if claims is not None:
            os.link(claims, data / "claims" / f"chunk_{i}.parquet")
    # (each chunk's source: its file and its claims file, as dump.chunk_sizes counts it;
    # process() deletes its source)
    sizes = [os.path.getsize(part) + (os.path.getsize(claims) if claims else 0) for _, part, claims in chunk_parts]
    sets = [name for name, _, _ in chunk_parts]
    part_files = [part for _, part, _ in chunk_parts]
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
    cpu_total = sum(u.ru_utime + u.ru_stime - v.ru_utime - v.ru_stime for u, v in zip(usage, usage0))
    stop.set()
    # The cores busy over finalise's two runs (timeline.py), sampled by a process of its
    # own, reaped after both runs so its CPU time is in none of their figures
    cores_log = out.with_suffix(".cores.jsonl")
    cores_log.unlink(missing_ok=True)
    sampler = subprocess.Popen([sys.executable, HERE / "timeline.py", str(os.getpid()), cores_log, run_dir])
    while sampler.poll() is None and not (cores_log.exists() and cores_log.stat().st_size):
        time.sleep(0.05)  # its first sample before finalise starts
    fin, eta = finalise_at_scale(compact, sort_by_id, Table, COMPACT_DIR, HUB_COPY_DIR, state.resolve(),
                                 n, part_files, wall, workers, keep,
                                 refs=(sorted(kept.glob("*/claims_refs/*.parquet")), cycles))
    sampler.terminate()
    sampler.wait()
    # The split at the release's size: its CPU time per byte of the dump's bz2, scaled to
    # the release's dump over the cores (lbzip2 and the split's workers share them)
    from release import RELEASE_DUMP_BYTES
    eta["split_eta"] = round(split["split_cpu_total"] / (split["split_source_mb"] * 1e6) * RELEASE_DUMP_BYTES / workers)
    eta["eta"] += eta["split_eta"]
    chunks = [json.loads(l) for l in open(log)]
    # Where finalise's wall time goes (timeline.py), and its peak RSS from the same samples:
    # the process tree's (summed, as peak_sum_rss_mb is the pipeline's) and the largest
    # worker's, in each of its two runs
    lines = timeline.summarise(phases_log, cores_log, os.getpid(),
                               {"finalise": ("finalise", "finalise end"),
                                "finalise_x3": ("finalise x3", "finalise x3 end"),
                                "claims_refs": ("claims refs", "claims refs end"),
                                "claims_refs_x3": ("claims refs x3", "claims refs x3 end")}, os.cpu_count())
    memory = {f"{k}{x}": lines.get(span, {}).get(f)
              for span, x in (("finalise", ""), ("finalise_x3", "_x3"))
              for k, f in (("finalise_peak_rss_mb", "peak_rss_mb"), ("finalise_worker_peak_mb", "worker_peak_rss_mb"))}
    memory |= refs_memory(phases_log, cores_log, os.getpid(), lines)
    # finalise at a release's size with a release's worker starts, and its peak RSS there
    # (modelled, from both runs' workers)
    eta |= release_model(eta, lines)
    # The claims parts scaled by the claims' own rows and bytes (modelled, from the claims'
    # work measured by set)
    claims = claims_measured(phases_log, chunks, sets, work_dir / "audit", sizes)
    tables = phases.table_cpu(str(phases_log), {"finalise": ("finalise", "finalise end"),
                                                "finalise_x3": ("finalise x3", "finalise x3 end")})
    eta |= claims_model(eta, claims, tables, wall, cpu_total)
    # The pipeline's worker starts as a release has them (modelled, from the workers' own
    # CPU time outside their chunks, measured, and the release's chunks and pool settings)
    from wikidata import config
    starts = pipeline_workers(phases_log, chunks)
    release_per = getattr(config, "CHUNKS_PER_WORKER", per)
    eta |= starts_model(eta, starts, wall, cpu_total, workers, release_per)
    eta |= {k: v for k, v in starts.items() if not k.startswith("_")}
    eta["pipeline_workers_counted"] = pool_starts(n, workers, per)  # (what pool_starts counts for the bench's)
    eta["release_chunks_per_worker"] = release_per
    eta |= refs_model(eta)
    eta.pop("_model")
    res = {
        "per": per, "workers": workers, "group": group_size, "n": n, "wall": round(wall, 2),
        "last_chunk_done": round(max(finished.values()), 2),
        "peak_sum_rss_mb": max(s for _, s in samples),
        "mean_sum_rss_mb": round(sum(s for _, s in samples) / len(samples)),
        "cpu_total": round(cpu_total, 1),
        "chunk_cpu": round(sum(c["cpu"] for c in chunks), 1),
        "process_cpu": round(sum(c["process_cpu"] for c in chunks), 1),
        "partition_cpu": round(sum(c["partition_cpu"] for c in chunks), 1),
        "worker_peak_mb": max(c["hwm_mb"] for c in chunks),
        **fin, "total_wall": round(wall + fin["finalise_wall"], 2), **eta, **split, **memory,
        # The sample end to end, measured, nothing modelled: the split (split-dump and
        # route-release), the pipeline and finalise (its first run, on the sample's rows)
        "measured_cpu": round(split["split_cpu_total"] + cpu_total + fin["finalise_cpu"], 1),
        "measured_wall": round(split["split_wall_total"] + wall + fin["finalise_wall"], 2),
        # (and with claims_labels' refs stage, run after finalise's first run: bench 13.1.0)
        **({"measured_cpu_refs": round(split["split_cpu_total"] + cpu_total + fin["finalise_cpu"] + eta["refs_cpu"], 1),
            "measured_wall_refs": round(split["split_wall_total"] + wall + fin["finalise_wall"] + eta["refs_wall"], 2)}
           if "refs_cpu" in eta else {}),
        "pipeline_tables": tables_measured(phases_log, chunks, sets, work_dir / "audit"),
        "closes": closes, "chunks": chunks, "samples": samples,
        "phases": phases.summarise(str(phases_log), {"split": ("split", "route end"),
                                                     "pipeline": ("pipeline", "pipeline end"),
                                                     "finalise": ("finalise", "finalise end"),
                                                     "finalise_x3": ("finalise x3", "finalise x3 end")}),
        "timeline": lines,
    }
    if keep:
        for d in ("audit", "quarantine"):
            if (work_dir / d).exists():
                shutil.copytree(work_dir / d, keep / d)
    out.write_text(json.dumps(res))
    shutil.rmtree(run_dir)
    shutil.rmtree(kept)
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
