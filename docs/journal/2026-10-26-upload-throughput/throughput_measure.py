"""THE USER'S TO RUN. No agent runs this: every subcommand but `logs` uploads to the Hub.

Measures what limits our uploads' rate (docs/journal/2026-10-26-upload-throughput.md):
whether the 7 MB/s on record is the uplink, or one connection's rate, or our calls'
fixed costs. Companion to #177's `preupload_measure.py` (same scratch repo conventions).

Subcommands:

- `sweep` (the main one): fresh random bytes (incompressible like zstd Parquet pages, and
  new each time, so Xet deduplicates nothing) preuploaded in one `preupload_lfs_files`
  call (no commit), in a fresh process for each point, as hf_xet reads its settings when
  its session starts. The points: a fixed number of parallel xorb uploads
  (`HF_XET_FIXED_UPLOAD_CONCURRENCY` = 1, 2, 4, 8, 16, 32), then the adaptive controller
  at its defaults (start 2, up to 64) and in high-performance mode
  (`HF_XET_HIGH_PERFORMANCE=1`: start 16, 4-124, 100 files at once). Each point at each
  `--sizes` (MiB a file; our parts are 500 MiB, our group claims files ~1.4 GB, our row
  groups 128 MiB), `--total-mb` of files a point. Each point first preuploads one 1 MiB
  file (the session's start, the token, a call's fixed cost), timed apart.
  - **Reading it:** MB/s rising with N, then flat, is one connection's rate times N up
    to the uplink; flat from N = 1 or 2 is the uplink (or a cap on the Hub's side). The
    adaptive points' last logged concurrency and predicted bandwidth (hf_xet's own log,
    captured per point) say where the controller settled.
- `small`: our small files' cost. Commits (`create_commit`, to `--revision`) of K files
  of S bytes, K in `--counts` (1, 50 = finalise's batch, 250 = upload_folder's), S in
  `--small-kb` (64 KiB, 1 MiB: the build branch's per-key files are mostly this size), at
  the defaults (8 files ingested at once) and with `--hp` also in high-performance mode
  (100). The intercept over K is a commit's fixed cost (the model's c0), the slope the
  per-file cost (q/F: each new file's first chunk is one global-dedup query).
- `parallel`: k calls at once, each preuploading `--size-mb` MiB in files of 128 MiB, as
  threads of one process (one Xet session and controller, as a run's committer and its
  pool share) and as k processes (k controllers, as the two sets' runs and finalise
  are), k in `--ks` (1, 2, 4).
- `logs` (**offline**, no Hub call): summarises the controller's lines in hf_xet's own
  log files (by default `~/.cache/huggingface/xet/logs`, kept up to 2 weeks and 250 MB;
  hf_xet 1.6.0 writes them at INFO level by default). Run it first on the release
  machine: any upload of the last two weeks shows the concurrency it settled at and its
  predicted bandwidth, with no new upload at all.

Bytes sent are the machine's (psutil's count, else /proc/net/dev's): run nothing else on
the link meanwhile. Results print as tables and go to `--out` (JSON).

Before running:
- a private scratch dataset repo, e.g. `hf repos create <you>/upload-scratch --type
  dataset --private`, or `--create` (refused unless the name has "scratch");
- free disk in `--dir` for one point's files (`--total-mb`, default 2048 MB);
- time: at 7 MB/s, `sweep` at its defaults is ~16 points of 2 GB: about 80 min.
  `--total-mb 512` takes about a quarter (but N = 32 then has only 8 xorbs of 64 MiB to
  send at once, so the high points read low);
- memory: each in-flight xorb is up to 64 MiB, so N = 32 holds ~2 GiB, and
  high-performance mode up to ~8 GiB of xorbs plus 100 files' buffers (peak RSS is
  reported per point).

Examples:

    python throughput_measure.py logs
    python throughput_measure.py sweep --repo <you>/upload-scratch
    python throughput_measure.py sweep --repo <you>/upload-scratch --sizes 500 --points 1,4,16,adaptive,hp
    python throughput_measure.py small --repo <you>/upload-scratch --revision measure --hp
    python throughput_measure.py parallel --repo <you>/upload-scratch

Afterwards: `sweep` and `parallel` commit nothing (their uploads are never referenced);
`small` commits its files to `--revision` (default `measure`, created if missing). Delete
the scratch repo when done (`hf repos delete <you>/upload-scratch --type dataset`);
nothing here deletes anything on the Hub.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import resource
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

MB, MIB = 1_000_000, 1024**2
POINTS = ["1", "2", "4", "8", "16", "32", "adaptive", "hp"]
KNOBS = ("HF_XET_HIGH_PERFORMANCE", "HF_XET_HP", "HF_XET_FIXED_UPLOAD_CONCURRENCY",
         "HF_XET_CLIENT_AC_INITIAL_UPLOAD_CONCURRENCY", "HF_XET_CLIENT_AC_MIN_UPLOAD_CONCURRENCY",
         "HF_XET_CLIENT_AC_MAX_UPLOAD_CONCURRENCY", "HF_XET_DATA_MAX_CONCURRENT_FILE_INGESTION",
         "HF_HUB_DISABLE_XET")


# -- offline: hf_xet's own logs ----------------------------------------------------------

LINE = re.compile(r"Concurrency control for upload: Current concurrency = (\d+); predicted bandwidth = ([\d.]+)"
                  r".*?observed bytes sent so far = (\d+); completed transmissions = (\d+)")
MOVE = re.compile(r"Concurrency control for upload: (Increased|Decreased) concurrency from (\d+) to (\d+)")
INIT = re.compile(r"Adaptive Concurrency Controller for upload with starting concurrency = (\d+); min = (\d+), max = (\d+)")
STAMP = re.compile(r'"timestamp":"([^"]+)"|^(\d{4}-\d\d-\d\dT[\d:.]+Z?)')


def read_logs(paths: list[Path]) -> list[dict]:
    """Each log file's upload-controller lines: its start, its moves, its periodic state."""
    out = []
    for p in sorted(paths):
        rec = {"file": str(p), "init": None, "states": [], "up": 0, "down": 0}
        try:
            text = p.read_text(errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            if m := INIT.search(line):
                rec["init"] = [int(x) for x in m.groups()]
            elif m := MOVE.search(line):
                rec["up" if m.group(1) == "Increased" else "down"] += 1
            elif m := LINE.search(line):
                ts = STAMP.search(line)
                rec["states"].append({"t": (ts.group(1) or ts.group(2)) if ts else None,
                                      "concurrency": int(m.group(1)), "bandwidth": float(m.group(2)),
                                      "sent": int(m.group(3)), "completed": int(m.group(4))})
        if rec["init"] or rec["states"]:
            out.append(rec)
    return out


def summarise_logs(recs: list[dict]) -> None:
    print("| log file | start/min/max | moves up/down | concurrency (median, last) | predicted MB/s (median, last) | GB sent | first, last |")
    print("|---|---|---|---|---|---|---|")
    for r in recs:
        s = r["states"]
        if not s:
            print(f"| {Path(r['file']).name} | {r['init']} | {r['up']}/{r['down']} | | | | |")
            continue
        c = [x["concurrency"] for x in s]
        b = [x["bandwidth"] / MB for x in s]
        print(f"| {Path(r['file']).name} | {r['init']} | {r['up']}/{r['down']} | {statistics.median(c):g}, {c[-1]} | "
              f"{statistics.median(b):.1f}, {b[-1]:.1f} | {s[-1]['sent'] / 1e9:.2f} | {s[0]['t']}, {s[-1]['t']} |")


def logs(a) -> None:
    if a.log_dir:
        root = a.log_dir
    elif os.environ.get("HF_XET_CACHE"):  # (hf_xet's own order: xet_runtime cache_dir.rs)
        root = Path(os.environ["HF_XET_CACHE"]) / "logs"
    elif os.environ.get("HF_HOME"):
        root = Path(os.environ["HF_HOME"]) / "xet" / "logs"
    else:
        cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
        root = cache / "huggingface" / "xet" / "logs"
    paths = [p for p in root.rglob("*") if p.is_file()] if root.exists() else []
    print(f"{len(paths)} log files under {root}")
    recs = read_logs(paths)
    summarise_logs(recs)
    a.out.write_text(json.dumps(recs, indent=2) + "\n")
    print(f"written {a.out}")


# -- the measurements (Hub uploads) ------------------------------------------------------

def sent() -> int:
    """The machine's bytes sent so far (psutil, else /proc/net/dev; 0 if neither)."""
    try:
        import psutil

        return psutil.net_io_counters().bytes_sent
    except ImportError:
        pass
    try:
        lines = Path("/proc/net/dev").read_text().splitlines()[2:]
    except OSError:
        return 0
    return sum(int(x.split(":", 1)[1].split()[8]) for x in lines if not x.strip().startswith("lo:"))


def write_files(d: Path, n: int, size: int, prefix: str) -> list[Path]:
    d.mkdir(parents=True, exist_ok=True)
    out = []
    for i in range(n):
        p = d / f"{prefix}-{i:05d}.bin"
        with p.open("wb") as f:
            left = size
            while left:
                block = os.urandom(min(left, 8 << 20))
                f.write(block)
                left -= len(block)
        out.append(p)
    return out


def ops(files: list[Path], tag: str):
    from huggingface_hub import CommitOperationAdd

    return [CommitOperationAdd(path_in_repo=f"measure/{tag}/{p.name}", path_or_fileobj=str(p)) for p in files]


def timed_preupload(api, repo: str, revision: str, operations) -> dict:
    nbytes = sum(op.upload_info.size for op in operations)
    s0, t0 = sent(), time.perf_counter()
    api.preupload_lfs_files(repo, additions=operations, repo_type="dataset", revision=revision, free_memory=False)
    wall = time.perf_counter() - t0
    return {"files": len(operations), "bytes": nbytes, "wall_s": round(wall, 3),
            "mb_s": round(nbytes / MB / wall, 2), "sent_mb": round((sent() - s0) / MB, 1)}


def point_env(point: str, logdir: Path) -> dict:
    """The environment for a sweep point: its knobs, and hf_xet's log to `logdir`."""
    env = {k: v for k, v in os.environ.items() if k not in KNOBS}
    env["HF_XET_LOG_DEST"] = str(logdir) + os.sep
    if point == "hp":
        env["HF_XET_HIGH_PERFORMANCE"] = "1"
    elif point != "adaptive":
        env["HF_XET_FIXED_UPLOAD_CONCURRENCY"] = point
    return env


def worker(a) -> None:
    """(_worker) One point in a fresh process: a 1 MiB call, then the files, timed."""
    from huggingface_hub import HfApi

    spec = json.loads(a.spec)
    api, d = HfApi(), Path(spec["dir"])
    tag = f"{spec['tag']}-{os.getpid()}"
    first = timed_preupload(api, spec["repo"], spec["revision"], ops(write_files(d / "first", 1, MIB, "first"), tag))
    out = {"first_call": first}
    if spec["threads"] <= 1:
        files = write_files(d / "files", spec["n"], spec["size"], "f")
        if go := spec.get("go"):  # (k processes: every one's files written before any call)
            (Path(go).parent / f"ready-{os.getpid()}").touch()
            while not Path(go).exists():
                time.sleep(0.2)
        out["call"] = timed_preupload(api, spec["repo"], spec["revision"], ops(files, tag))
    else:  # (k calls at once in this process: one session, one controller)
        sets = [write_files(d / f"t{i}", spec["n"], spec["size"], f"t{i}") for i in range(spec["threads"])]
        res: list = [None] * len(sets)
        s0, t0 = sent(), time.perf_counter()

        def go(i):
            res[i] = timed_preupload(api, spec["repo"], spec["revision"], ops(sets[i], f"{tag}-t{i}"))

        ts = [threading.Thread(target=go, args=(i,)) for i in range(len(sets))]
        [t.start() for t in ts]
        [t.join() for t in ts]
        wall = time.perf_counter() - t0
        nbytes = sum(r["bytes"] for r in res)
        out["call"] = {"calls": res, "bytes": nbytes, "wall_s": round(wall, 3),
                       "mb_s": round(nbytes / MB / wall, 2), "sent_mb": round((sent() - s0) / MB, 1)}
    out["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    print("RESULT " + json.dumps(out), flush=True)


def spawn(a, point: str, size: int, n: int, work: Path, threads: int = 1,
          go: Path | None = None) -> tuple[subprocess.Popen, Path]:
    d = Path(tempfile.mkdtemp(prefix=f"p{point}-", dir=work))
    (d / "xetlog").mkdir()
    spec = {"repo": a.repo, "revision": a.revision, "dir": str(d), "size": size, "n": n,
            "threads": threads, "go": str(go) if go else None,
            "tag": time.strftime("%Y%m%dT%H%M%S") + f"-{point}-{size // MIB}"}
    return subprocess.Popen([sys.executable, __file__, "_worker", "--spec", json.dumps(spec)],
                            env=point_env(point, d / "xetlog"), stdout=subprocess.PIPE, text=True), d


def collect(proc: subprocess.Popen, d: Path) -> dict:
    stdout, _ = proc.communicate()
    line = next((x for x in stdout.splitlines() if x.startswith("RESULT ")), None)
    res = json.loads(line[7:]) if line else {"error": f"exit {proc.returncode}"}
    recs = read_logs([p for p in (d / "xetlog").rglob("*") if p.is_file()])
    states = [s for r in recs for s in r["states"]]
    if states:
        res["xet"] = {"concurrency_last": states[-1]["concurrency"],
                      "concurrency_max": max(s["concurrency"] for s in states),
                      "predicted_mb_s_last": round(states[-1]["bandwidth"] / MB, 2),
                      "moves_up": sum(r["up"] for r in recs), "moves_down": sum(r["down"] for r in recs)}
    shutil.rmtree(d, ignore_errors=True)
    return res


def row(label: str, r: dict) -> str:
    c, f, x = r.get("call", {}), r.get("first_call", {}), r.get("xet", {})
    return (f"| {label} | {c.get('bytes', 0) / MB:.0f} | {c.get('wall_s', '')} | **{c.get('mb_s', '')}** | "
            f"{c.get('sent_mb', '')} | {f.get('wall_s', '')} | {x.get('concurrency_last', '')}/{x.get('concurrency_max', '')} | "
            f"{x.get('predicted_mb_s_last', '')} | {r.get('peak_rss_mb', '')} | {r.get('error', '')} |")


HEAD = ("| point | MB | wall s | MB/s | MB sent | 1 MiB call s | xet concurrency last/max | xet predicted MB/s | peak RSS MB | error |\n"
        "|---|---|---|---|---|---|---|---|---|---|")


def setup(a):
    from huggingface_hub import HfApi, __version__

    if a.create and "scratch" not in a.repo:
        sys.exit("refusing --create for a repo whose name has no 'scratch' in it")
    api = HfApi()
    if a.create:
        api.create_repo(a.repo, repo_type="dataset", private=True, exist_ok=True)
    if a.revision != "main":
        api.create_branch(a.repo, repo_type="dataset", branch=a.revision, exist_ok=True)
    try:
        from importlib.metadata import version

        xet = version("hf_xet")
    except Exception:
        xet = None
    print(f"huggingface_hub {__version__}, hf_xet {xet}; knobs in the environment (cleared per point): "
          f"{ {k: os.environ[k] for k in KNOBS if k in os.environ} or 'none'}")
    return api, {"huggingface_hub": __version__, "hf_xet": xet, "repo": a.repo}


def sweep(a) -> None:
    _, report = setup(a)
    report["points"] = []
    work = Path(tempfile.mkdtemp(prefix="uprate-", dir=a.dir))
    try:
        for size_mib in (int(s) for s in a.sizes.split(",")):
            size = size_mib * MIB
            n = max(1, round(a.total_mb * MB / size))
            print(f"\n{n} files of {size_mib} MiB a point\n\n{HEAD}", flush=True)
            for point in a.points.split(","):
                proc, d = spawn(a, point, size, n, work)
                r = collect(proc, d) | {"point": point, "size_mib": size_mib}
                report["points"].append(r)
                print(row(point, r), flush=True)
                a.out.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print(f"\nwritten {a.out}")


def parallel(a) -> None:
    _, report = setup(a)
    report["parallel"] = []
    size, n = 128 * MIB, max(1, round(a.size_mb / 128))
    work = Path(tempfile.mkdtemp(prefix="uprate-", dir=a.dir))
    try:
        print(f"\neach call {n} files of 128 MiB, defaults\n\n{HEAD}", flush=True)
        for k in (int(x) for x in a.ks.split(",")):
            proc, d = spawn(a, "adaptive", size, n, work, threads=k)
            r = collect(proc, d) | {"k": k, "how": "threads"}
            report["parallel"].append(r)
            print(row(f"{k} calls, one process", r), flush=True)
            if k > 1:
                gate = Path(tempfile.mkdtemp(prefix="gate-", dir=work))
                procs = [spawn(a, "adaptive", size, n, work, go=gate / "go") for _ in range(k)]
                while len(list(gate.glob("ready-*"))) < k and all(p.poll() is None for p, _ in procs):
                    time.sleep(0.2)
                t0 = time.perf_counter()
                (gate / "go").touch()
                rs = [collect(p, d) for p, d in procs]
                wall = time.perf_counter() - t0
                nbytes = sum(x.get("call", {}).get("bytes", 0) for x in rs)
                r = {"k": k, "how": "processes", "each": rs,
                     "call": {"bytes": nbytes, "wall_s": round(wall, 1), "mb_s": round(nbytes / MB / wall, 2)},
                     "peak_rss_mb": max(x.get("peak_rss_mb", 0) for x in rs)}
                report["parallel"].append(r)
                print(row(f"{k} processes", r), flush=True)
            a.out.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print(f"\nwritten {a.out}")


def small_worker(a) -> None:
    """(_small) Commits of K files of S bytes in this process's settings."""
    from huggingface_hub import HfApi

    spec = json.loads(a.spec)
    api, d, out = HfApi(), Path(spec["dir"]), []
    for kb in spec["kbs"]:
        for k in spec["counts"]:
            for rep in range(spec["repeat"]):
                tag = f"{spec['tag']}-{kb}k-{k}-{rep}"
                files = write_files(d / tag, k, kb * 1024, "s")
                operations = ops(files, tag)
                s0, t0 = sent(), time.perf_counter()
                api.create_commit(spec["repo"], repo_type="dataset", revision=spec["revision"],
                                  operations=operations, commit_message=f"measure {tag}")
                wall = time.perf_counter() - t0
                modes = [getattr(op, "_upload_mode", None) for op in operations]
                out.append({"kb": kb, "files": k, "rep": rep, "wall_s": round(wall, 3),
                            "sent_mb": round((sent() - s0) / MB, 2),
                            "lfs": modes.count("lfs"), "regular": modes.count("regular")})
                print(f"  {k} files of {kb} KiB: {wall:.2f} s", file=sys.stderr, flush=True)
                shutil.rmtree(d / tag, ignore_errors=True)
    print("RESULT " + json.dumps(out), flush=True)


def small(a) -> None:
    _, report = setup(a)
    report["small"] = {}
    work = Path(tempfile.mkdtemp(prefix="uprate-", dir=a.dir))
    try:
        for point in ["adaptive"] + (["hp"] if a.hp else []):
            spec = {"repo": a.repo, "revision": a.revision, "dir": str(work), "repeat": a.repeat,
                    "kbs": [int(x) for x in a.small_kb.split(",")], "counts": [int(x) for x in a.counts.split(",")],
                    "tag": time.strftime("%Y%m%dT%H%M%S") + f"-{point}"}
            env = point_env(point, work / f"xetlog-{point}")
            stdout = subprocess.run([sys.executable, __file__, "_small", "--spec", json.dumps(spec)],
                                    env=env, stdout=subprocess.PIPE, text=True).stdout
            line = next((x for x in stdout.splitlines() if x.startswith("RESULT ")), None)
            rows = json.loads(line[7:]) if line else []
            report["small"][point] = rows
            modes = (sum(r["lfs"] for r in rows), sum(r["regular"] for r in rows))
            print(f"\n{point}: a commit of K files of S KiB (upload modes: {modes[0]} via Xet/LFS, {modes[1]} regular; "
                  f"regular files skip Xet, so their cost is not ours)\n\n| S KiB | K | wall s (median) | s a file |\n|---|---|---|---|")
            for kb in spec["kbs"]:
                walls = {k: statistics.median(r["wall_s"] for r in rows if r["kb"] == kb and r["files"] == k)
                         for k in spec["counts"] if any(r["kb"] == kb and r["files"] == k for r in rows)}
                ks = sorted(walls)
                for k in ks:
                    per = (walls[k] - walls[ks[0]]) / (k - ks[0]) if k != ks[0] else float("nan")
                    print(f"| {kb} | {k} | {walls[k]:.2f} | {per:.3f} |")
            a.out.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print(f"\nwritten {a.out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    lg = sub.add_parser("logs", help="offline: hf_xet's own logs")
    lg.add_argument("--log-dir", type=Path, default=None)
    lg.add_argument("--out", type=Path, default=Path("xet_logs.json"))
    for name in ("sweep", "small", "parallel"):
        s = sub.add_parser(name)
        s.add_argument("--repo", required=True, help="a scratch dataset repo of yours")
        s.add_argument("--revision", default="main" if name != "small" else "measure")
        s.add_argument("--create", action="store_true", help="create the repo (private) if missing")
        s.add_argument("--dir", type=Path, default=None, help="where the synthetic files go")
        s.add_argument("--out", type=Path, default=Path(f"throughput_{name}.json"))
        if name == "sweep":
            s.add_argument("--sizes", default="128,512", help="MiB a file, comma-separated")
            s.add_argument("--total-mb", type=int, default=2048, help="MB of files a point")
            s.add_argument("--points", default=",".join(POINTS))
        if name == "small":
            s.add_argument("--small-kb", default="64,1024")
            s.add_argument("--counts", default="1,50,250")
            s.add_argument("--repeat", type=int, default=2)
            s.add_argument("--hp", action="store_true", help="also in high-performance mode")
        if name == "parallel":
            s.add_argument("--ks", default="1,2,4")
            s.add_argument("--size-mb", type=int, default=1024, help="MiB a call")
    for name in ("_worker", "_small"):
        s = sub.add_parser(name)
        s.add_argument("--spec", required=True)
    a = p.parse_args()
    {"logs": logs, "sweep": sweep, "small": small, "parallel": parallel,
     "_worker": worker, "_small": small_worker}[a.cmd](a)


if __name__ == "__main__":
    main()
