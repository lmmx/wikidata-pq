"""The split reading the dump as synchronised sweeps (WIKIDATA_SPLIT_SWEEPS, bench 17.13.0;
src/wikidata/sweeps.py, docs/journal/2026-10-26-progressive-finalise.md, PR 2): run apart
from the timed run, after it (ab.run_once), so none of its figures take this in, and only
for a commit with the sweep reader.

The bench's slice is a few chunks of the dump, not its 8 shards: each of its files is one
shard's run (c0 shard 0, r50 shard 3, s30 shard 2, s90 shard 7). So its entities are
written as a dump of 8 shards one after another (dumpslice.sharded, made once beside the
slice, untimed): its Q ids given anew round-robin over the 8 shards, each shard's ids
rising from Q10,000 as sha1 puts them in it, each file's rows in their order, one bz2
stream written by lbzip2, as Wikimedia's dump job writes the real one. Then, each in a
directory of its own, on that dump:

- **the reader alone,** its lines counted and dropped: today's (`dump._dump_lines`, one
  `lbzip2` on every core) and the sweeps' (`sweeps.Reader`, 8 decoders, finding its cuts
  first), each in a process of its own: `sweeps_read_serial_cpu`, `_wall` and
  `sweeps_read_cpu`, `_wall` (the CPU of the process and its decoders);
- **the split** (split-dump, routing at its end), in byte order and as sweeps:
  `sweeps_serial_split_cpu`, `_wall`, `_rss_mb` and `sweeps_split_cpu`, `_wall`, `_rss_mb`;
  the cuts' probes and seconds (`sweeps_count`, `sweeps_plan_probes`, `sweeps_plan_s`);
- **the census of each split's order** (dump.split_census): the frontier (the least of the
  shards' running maxima) at most `sweeps_frontier_lag` below the largest id read in the
  sweeps' order, against `sweeps_serial_frontier_lag` in byte order; the chunks after
  which it rose (`sweeps_frontier_rises`, `sweeps_serial_frontier_rises`); and the most a
  Q id came below its shard's running maximum (`sweeps_lag`, past the heads), which in
  the sweeps' order bounds how far below the frontier it came;
- **each split's parts:** every entity in exactly one part, the same entities in each set
  (`sweeps_entities_ok`);
- **parity of the published files** (`sweeps_published_ok`): both sets' runs (as
  overlap.py's sequential release runs them, with its stand-in Hub) and finalise's three
  runs (link.finalise_stage, no link) on each split; finalise replaces each key's group
  files on the build branch with its sorted files, so the build branch's files after it
  are what the release publishes, and they must be the same, byte for byte. ab.py exits 1
  if they are not.

Usage: pipeline.py --sweeps <src> <out.json> <workers> <chunk files...>
"""

import json
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
FACTOR = 8

# The reader alone: its entity lines counted (today's, or the sweeps' with WIKIDATA_SPLIT_SWEEPS)
READ = """
import os, sys
from pathlib import Path
sys.path.insert(0, {src!r})
from wikidata import dump
path = Path({dump!r})
if {factor}:
    from wikidata import sweeps
    reader = sweeps.Reader(path, {factor}, Path({state!r}), {{}})
    n = sum(1 for _ in reader.lines())
else:
    n = sum(1 for _ in dump._dump_lines(path, [0]))
print(n)
"""


def _tree_rss_mb(pid: int) -> int:
    from overlap import _tree_rss_mb as rss

    return rss(pid)


def timed(cmd: list[str], cwd: Path, env: dict, log) -> dict:
    """A process's wall, its CPU time and its children's (RUSAGE_CHILDREN: it is waited for
    here, and it waits for its own), and the peak RSS of it and its children."""
    r0, t = resource.getrusage(resource.RUSAGE_CHILDREN), time.perf_counter()
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
    peak = 0
    while proc.poll() is None:
        try:
            peak = max(peak, _tree_rss_mb(proc.pid))
        except (OSError, StopIteration):
            pass
        time.sleep(0.2)
    if proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, cmd[:3])
    r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {"wall": round(time.perf_counter() - t, 2), "cpu": round(r1.ru_utime + r1.ru_stime - r0.ru_utime - r0.ru_stime, 1),
            "rss_mb": peak}


def env_of(workers: int, factor: int | None) -> dict:
    """The split's environment (overlap.py's, a release's defaults), reading the dump in
    byte order, or as `factor` sweeps."""
    from overlap import _env

    env = _env("split", workers, False)
    env.pop("WIKIDATA_SPLIT_SWEEPS", None)
    if factor:
        env["WIKIDATA_SPLIT_SWEEPS"] = str(factor)
    return env


def release_dir(root: Path, dump: Path) -> Path:
    """A release's directory at `root`, with `dump` linked in as its dump."""
    release = os.environ["WIKIDATA_RELEASE"]
    d = root / "releases" / release / "dump"
    d.mkdir(parents=True)
    os.link(dump, d / f"wikidata-{release}-all.json.bz2")
    return d / f"wikidata-{release}-all.json.bz2"


def split(src: Path, root: Path, env: dict, log) -> dict:
    code = f"import sys; sys.path.insert(0, {str(src)!r}); from wikidata import dump; dump.split()"
    return timed([sys.executable, "-c", code], root, env, log)


def census(src: Path, root: Path, env: dict) -> dict:
    code = (f"import sys, json; sys.path.insert(0, {str(src)!r}); from wikidata import dump; "
            "print(json.dumps(dump.split_census()))")
    out = subprocess.run([sys.executable, "-c", code], cwd=root, env=env, capture_output=True, text=True, check=True)
    return json.loads(out.stdout.splitlines()[-1])


def entities(root: Path) -> dict[str, list[str]]:
    """Each set's entities in its split's parts, sorted (with any repeated)."""
    import polars as pl

    release = os.environ["WIKIDATA_RELEASE"]
    out = {}
    for name, d in (("main", root / "releases" / release / "data"),
                    ("scholar", root / "releases" / f"{release}-scholar" / "data")):
        ids = [i for f in sorted(d.glob("chunk_*.parquet")) for i in pl.read_parquet(f, columns=["id"])["id"].to_list()]
        out[name] = sorted(ids)
    return out


def published(src: Path, root: Path, workers: int, log) -> dict:
    """Both sets' runs on a split's parts, then finalise's three runs (link.finalise_stage,
    no link), through the stand-in Hub: the digest of the build branch's files after it
    (the files the release publishes), its bytes by set and table, and the walls."""
    from link import finalise_stage
    from overlap import _env, _run

    steps = [[("scholar", _env("scholar", workers, False), 0)], [("main", _env("main", workers, False), 0)]]
    for step in steps:
        for _, env, _ in step:
            env.pop("WIKIDATA_SPLIT_SWEEPS", None)
    runs_wall, _ = _run(src, root, steps, log)
    work = Path(tempfile.mkdtemp(dir=root.parent))
    try:
        f = finalise_stage(src, root, work, None, {}, 1, workers, log)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return {"digest": f["digest"], "published": f["published"], "runs_wall": runs_wall, "finalise_wall": f["wall"]}


def sweeps_stage(src: Path, out: Path, workers: int, files: list[str]) -> None:
    from dumpslice import sharded, sharded_path

    if not (src / "wikidata" / "sweeps.py").exists():
        out.write_text("{}")  # (a commit without the sweep reader)
        return
    dump_slice = Path(os.environ.get("BENCH_DUMP_SLICE") or out.parent / "dump-slice.json.bz2")
    if not dump_slice.exists():
        from dumpslice import make

        dump_slice.parent.mkdir(parents=True, exist_ok=True)
        make(dump_slice, files)
    dump = sharded_path(dump_slice)
    if not dump.exists():
        sharded(dump_slice, dump)
    res = {"sweeps_source_mb": round(dump.stat().st_size / 1e6, 1)}
    stage = Path(tempfile.mkdtemp(dir=out.parent))
    figures, digests, ids = {}, {}, {}
    try:
        with open(out.with_suffix(".log"), "w") as log:
            for mode, factor in (("serial", None), ("sweeps", FACTOR)):
                p = "sweeps_" if mode == "sweeps" else "sweeps_serial_"
                env = env_of(workers, factor)
                # the reader alone
                root = stage / f"read-{mode}"
                path = release_dir(root, dump)
                code = READ.format(src=str(src), dump=str(path), factor=factor or 0, state=str(root))
                r = timed([sys.executable, "-c", code], root, env, log)
                key = "sweeps_read" if factor else "sweeps_read_serial"
                res |= {f"{key}_cpu": r["cpu"], f"{key}_wall": r["wall"]}
                shutil.rmtree(root)
                # the split
                root = stage / mode
                release_dir(root, dump)
                s = split(src, root, env, log)
                res |= {f"{p}split_{k}": v for k, v in s.items()}
                c = census(src, root, env)
                res |= {f"{p}frontier_lag": c["frontier"]["lag"], f"{p}frontier_rises": c["frontier"]["rises"],
                        f"{p}chunks": c["chunks"], f"{p}lag": c["all"]["lag"], f"{p}late": c["all"]["late"]}
                if factor:
                    plan = json.loads((root / "releases" / os.environ["WIKIDATA_RELEASE"] / "data"
                                       / "sweeps.json").read_text())
                    res |= {"sweeps_count": len(plan["sweeps"]), "sweeps_plan_probes": plan["probes"],
                            "sweeps_plan_s": plan["seconds"]}
                ids[mode] = entities(root)
                pub = published(src, root, workers, log)
                digests[mode] = pub["digest"]
                figures[mode] = pub["published"]
                shutil.rmtree(root)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    once = all(len(set(v)) == len(v) for m in ids.values() for v in m.values())
    res["sweeps_entities_ok"] = once and ids["serial"] == ids["sweeps"]
    res["sweeps_published_ok"] = digests["serial"] == digests["sweeps"]
    res["sweeps_published_mb"] = round(sum(sum(t.values()) for t in figures["sweeps"].values()) / 1e6, 1)
    res["sweeps_parity_ok"] = res["sweeps_entities_ok"] and res["sweeps_published_ok"]
    out.write_text(json.dumps(res))
    print(f"sweeps: {res['sweeps_count']} sweeps; read CPU {res['sweeps_read_cpu']}s against "
          f"{res['sweeps_read_serial_cpu']}s; split CPU {res['sweeps_split_cpu']}s against "
          f"{res['sweeps_serial_split_cpu']}s; frontier lag {res['sweeps_frontier_lag']:,} against "
          f"{res['sweeps_serial_frontier_lag']:,}; entities {'once' if res['sweeps_entities_ok'] else 'NOT once'}; "
          f"published {'the same' if res['sweeps_published_ok'] else 'DIFFERENT'}", flush=True)
