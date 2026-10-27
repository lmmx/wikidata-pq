"""The compute benchmark: a release's processing of a real Wikidata dump on this machine,
end to end, with the network out of the timed path (bench/full/README.md).

    run.py fetch  --release R [--prefix-gb N | --windows K --window-mb W]
    run.py run    --release R [--prefix-gb N | --windows K --window-mb W]
                  [--source local|stream] [--orchestration overlap|serial] [...]
    run.py report OUT_DIR [OTHER_OUT_DIR]

`fetch` caches the dump (the whole release's, by `download-dump`, md5 and sha1 checked
against Wikimedia's published sums; or a scaled one, by splice.py), timed on its own.

`run` processes it as a release's recipes do, every Hub call answered by the on-disk
stand-in (standin.py), so nothing is uploaded or downloaded on the timed path:

- `--source local` (the default): the cached dump, read from disk. No network at all.
- `--source stream`: the dump's bytes fetched from dumps.wikimedia.org as the run goes
  (`download-dump` for the whole dump, splice.py for a scaled one), the split following
  the growing .part as a release's does (`just download-split`), so the download and the
  decode overlap. Its wall against a local run's is how much of the download the
  processing hides.
- `--orchestration overlap` (the default): `just overlap-release` and then `just
  release` without its promotion: the split and both sets' runs at once, finalise's runs
  (finalise-release-runs) as the main run ends. `serial`: `just download-split` and then
  `just release` without its promotion: the split, then the scholarly set's run, the
  main set's, finalise's runs, one after another, so each stage's figures are its own.

It writes OUT_DIR/result.json (every figure, machine-readable) and OUT_DIR/summary.md;
`report` rewrites them from OUT_DIR's records, or compares two runs (a local and a
streaming run, say).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SITE = HERE / "hook"  # (not "site": .gitignore ignores site/, the docs build)
WRAP = HERE / "wrap.py"
SPLICE = HERE / "splice.py"
BIN = Path(sys.executable).parent
VERSION = "1.0.0"  # (this tool's: bench/full/README.md; the CI bench's is bench/version.py)

# The run's peak disk beside the dump, over the dump's bytes, by default: ~300 GB for a
# release's 103.5 GB dump. The record (docs/journal/2026-10-25-release-critical-path.md,
# "Peak memory and disk") puts a release's peak at ~235 GB with the dump, ~300 GB when
# the group files stay on disk until finalise (WIKIDATA_KEEP_GROUP_FILES=1). The stand-in
# Hub keeps the build branch on this disk, as kept group files are, so the latter is the
# floor here; the factor, 2.9, takes the record's upper end of ~400 GB in all with the
# dump (its claims sort's ~3x the claims, and the published files beside the group files
# while finalise replaces them). A scaled run reports its own measured factor.
DISK_FACTOR = 2.9
DISK_MARGIN = 10e9  # (bytes kept free besides)
RELEASE_DUMP_BYTES = 103_488_697_311  # (bench/release.py: wikidata-20261019-all.json.bz2)


def default_cache() -> Path:
    return Path(os.environ.get("WIKIDATA_BENCH_CACHE") or Path.home() / ".cache" / "wikidata-pq" / "bench-full")


def scale_args(a) -> list[str]:
    if a.prefix_gb:
        return ["--prefix-gb", f"{a.prefix_gb:g}"]
    if a.windows:
        return ["--windows", str(a.windows), "--window-mb", f"{a.window_mb:g}"]
    return []


def label_of(a) -> str:
    if getattr(a, "custom", None):
        return f"custom-{a.custom}"
    if a.prefix_gb:
        return f"prefix{a.prefix_gb:g}g"
    if a.windows:
        return f"windows{a.windows}x{a.window_mb:g}m"
    return "full"


def full_root(cache: Path, release: str) -> Path:
    """Where `fetch` downloads the whole dump: a release root (download-dump's cwd)."""
    return cache / release / "root"


def cached_dump(cache: Path, release: str, label: str) -> Path:
    if label == "full":
        return full_root(cache, release) / "releases" / release / "dump" / f"wikidata-{release}-all.json.bz2"
    return cache / release / f"{label}.json.bz2"


def _hash(path: Path, algo: str) -> tuple[str, float]:
    h, t = hashlib.new(algo), time.perf_counter()
    with open(path, "rb") as f:
        while b := f.read(16 << 20):
            h.update(b)
    return h.hexdigest(), round(time.perf_counter() - t, 1)


def json_bytes(path: Path) -> int:
    """The dump's decompressed bytes (lbzip2 -dc | counted), outside any timed figure."""
    p = subprocess.Popen(["lbzip2", "-dc", str(path)], stdout=subprocess.PIPE)
    n = 0
    while b := p.stdout.read(16 << 20):
        n += len(b)
    if p.wait():
        raise RuntimeError(f"lbzip2 -dc {path} failed ({p.returncode})")
    return n


# -- fetch
def fetch(a) -> Path:
    cache, label = a.cache, label_of(a)
    dst = cached_dump(cache, a.release, label)
    meta_path = dst.with_name(dst.name + ".meta.json")
    if dst.exists() and meta_path.exists():
        print(f"[fetch] {dst} is cached", flush=True)
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if label.startswith("custom-"):  # (any local bz2 dump, as it is: the CI bench's slice, say)
        if not a.file:
            raise SystemExit("[fetch] --custom NAME takes --file PATH, a bz2 dump to cache as it is")
        tmp = dst.with_name(dst.name + ".part")
        shutil.copyfile(a.file, tmp)
        tmp.replace(dst)
        meta = {"label": label, "release": a.release, "source": {"path": str(a.file.resolve())},
                "bytes": dst.stat().st_size, "sha256": _hash(dst, "sha256")[0]}
        print("[fetch] checking it decodes (lbzip2 -dc), and counting its JSON", flush=True)
        meta["json_bytes"] = json_bytes(dst)
    elif label == "full":
        need = RELEASE_DUMP_BYTES + DISK_MARGIN
        free = shutil.disk_usage(dst.parent).free
        if free < need and not a.force:
            raise SystemExit(f"[fetch] the dump needs ~{need / 1e9:.0f} GB free at {dst.parent}; "
                             f"{free / 1e9:.0f} GB are (--force to try anyway)")
        root = full_root(cache, a.release)
        env = {**os.environ, "WIKIDATA_RELEASE": a.release}
        env.pop("WIKIDATA_SCHOLAR", None)
        rec = root / "download.rusage.json"
        if subprocess.run([sys.executable, "-I", str(WRAP), str(rec), "--", str(BIN / "download-dump")],
                          cwd=root, env=env).returncode:
            raise SystemExit("[fetch] download-dump failed")
        sys.path.insert(0, str(HERE))
        from splice import remote_info

        info = remote_info(a.release)
        sha1, sha1_s = _hash(dst, "sha1")
        if info.get("sha1") and sha1 != info["sha1"]:
            raise SystemExit(f"[fetch] {dst}: sha1 {sha1} is not the published {info['sha1']}")
        meta = {"label": label, "release": a.release, "source": info, "bytes": dst.stat().st_size,
                "md5_checked_by": "download-dump", "sha1": sha1, "sha1_checked": bool(info.get("sha1")),
                "sha1_s": sha1_s, "download": json.loads(rec.read_text())}
    else:
        net = dst.with_name(dst.name + ".net.jsonl")
        net.unlink(missing_ok=True)
        cmd = [sys.executable, "-I", str(SPLICE), str(dst), "--release", a.release, *scale_args(a),
               "--meta", str(meta_path) + ".tmp", "--log", str(net)]
        if a.from_file:
            cmd += ["--from-file", str(a.from_file)]
        if subprocess.run(cmd).returncode:
            raise SystemExit("[fetch] splice.py failed")
        meta = json.loads(Path(str(meta_path) + ".tmp").read_text())
        Path(str(meta_path) + ".tmp").unlink()
        print("[fetch] checking it decodes (lbzip2 -dc), and counting its JSON", flush=True)
        meta["json_bytes"] = json_bytes(dst)
    meta["fetch_wall_s"] = round(time.time() - t0, 1)
    meta["fetched_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    meta_path.write_text(json.dumps(meta, indent=1))
    print(f"[fetch] {dst}: {meta['bytes'] / 1e9:.3f} GB in {meta['fetch_wall_s']} s", flush=True)
    return dst


# -- the record of the machine and the code
def _sh(cmd: list[str], cwd: Path | None = None) -> str | None:
    try:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=30).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def machine(work: Path) -> dict:
    cpu = {}
    try:
        info = Path("/proc/cpuinfo").read_text()
        cpu["model"] = next((l.split(":", 1)[1].strip() for l in info.splitlines() if l.startswith("model name")), None)
        cpu["mhz"] = next((float(l.split(":", 1)[1]) for l in info.splitlines() if l.startswith("cpu MHz")), None)
        cpu["flags_sha_ni"] = " sha_ni " in info
        cpu["flags_avx2"] = " avx2 " in info
        cpu["physical_cores"] = len({l for l in info.splitlines() if l.startswith("core id")}) or None
    except OSError:
        pass
    cpu["logical_cpus"] = os.cpu_count()
    cpu["affinity"] = len(os.sched_getaffinity(0))
    mem = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, v = line.split(":", 1)
            if k in ("MemTotal", "SwapTotal"):
                mem[k] = int(v.split()[0]) * 1024
    except OSError:
        pass
    disk = {"path": str(work)}
    df = _sh(["df", "-PT", str(work)])
    if df:
        p = df.splitlines()[-1].split()
        disk |= {"device": p[0], "fs": p[1], "size": int(p[2]) * 1024}
        name = Path(p[0]).name
        base = name.rstrip("0123456789") if not name.startswith("nvme") else name.split("p")[0]
        for k, f in (("model", f"/sys/block/{base}/device/model"), ("rotational", f"/sys/block/{base}/queue/rotational")):
            try:
                disk[k] = Path(f).read_text().strip()
            except OSError:
                pass
    osr = {}
    try:
        for line in Path("/etc/os-release").read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                osr[k] = v.strip('"')
    except OSError:
        pass
    return {"hostname": platform.node(), "cpu": cpu, "memory": mem, "disk": disk,
            "os": osr.get("PRETTY_NAME"), "kernel": platform.release(), "arch": platform.machine(),
            "virtualised": _sh(["systemd-detect-virt"]), "python": sys.version.split()[0],
            "load_at_start": os.getloadavg()}


PACKAGES = ("wikidata", "wikidata-reshape", "polars", "pyarrow", "pq-glue", "polars-genson", "orjson", "numpy",
            "huggingface-hub", "hf-xet")


def code() -> dict:
    from importlib import metadata

    versions = {}
    for p in PACKAGES:
        try:
            versions[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            versions[p] = None
    tools = {t: (_sh([t, "--version"]) or "").splitlines()[0] if shutil.which(t) else None
             for t in ("lbzip2", "pigz", "igzip")}
    return {"commit": _sh(["git", "rev-parse", "HEAD"], REPO), "describe": _sh(["git", "describe", "--always", "--dirty"], REPO),
            "dirty": bool(_sh(["git", "status", "--porcelain", "--", "src", "reshape", "pyproject.toml", "uv.lock"], REPO)),
            "packages": versions, "tools": tools, "tool_version": VERSION}


ENV_PREFIXES = ("WIKIDATA_", "POLARS_", "RAYON_", "HF_XET_", "OMP_", "MALLOC_", "PYTHON")
# What the pipeline's config resolves to under the run's environment (each set's)
CONFIG = ("CHUNK_WORKERS", "CHUNKS_PER_WORKER", "FINALISE_WORKERS", "FINALISE_LARGE_WORKERS", "FINALISE_MEMORY_BYTES",
          "FINALISE_PENDING_UPLOAD_BYTES", "FINALISE_COMMITTERS", "FINALISE_CLAIMS_LAST", "FINALISE_EARLY",
          "FINALISE_CROSS_RUN", "FINALISE_CROSS_RUN_BYTES", "KEEP_GROUP_FILES", "GROUP_SOURCE_GB", "GROUP_SINK_ROWS",
          "PUSH_TABLES_AT_ONCE", "COMPACT_FILE_BYTES", "COMPACT_COMMIT_MAX_ADDS", "SORT_IN_MEMORY_BYTES",
          "SORT_BUCKET_BYTES", "DUMP_FORMAT", "SPLIT_WRITER_THREADS")


def config_of(env: dict, cwd: Path) -> dict:
    ask = f"""
import json, os
from wikidata import config, dump
out = {{k: getattr(config, k, None) for k in {list(CONFIG)!r}}}
out["bz2_decoder_env"] = os.environ.get("WIKIDATA_BZ2_DECODER")  # (#197: lbzip2 by default, or native)
out["bz2_threads_env"] = os.environ.get("WIKIDATA_BZ2_THREADS")  # (#197: the native decoder's threads)
try:  # (the decoder the split would start; its signature may change with WIKIDATA_BZ2_DECODER)
    out["decoder"] = dump.decoder(config.DUMP_FORMAT)
except BaseException as e:
    out["decoder"] = f"unknown ({{type(e).__name__}}: {{e}})"
for k, f in (("bz2_decoder", "bz2_decoder"), ("bz2_threads", "bz2_threads")):  # (#197's, resolved)
    try:
        out[k] = getattr(dump, f)() if hasattr(dump, f) else None
    except BaseException as e:
        out[k] = f"unknown ({{type(e).__name__}}: {{e}})"
print(json.dumps(out, default=str))
"""
    out = {}
    for name, scholar in (("main", ""), ("scholar", "1")):
        r = subprocess.run([sys.executable, "-c", ask], cwd=cwd, env=env | {"WIKIDATA_SCHOLAR": scholar},
                           capture_output=True, text=True)
        out[name] = json.loads(r.stdout) if r.returncode == 0 else {"error": r.stderr[-2000:]}
    return out


# -- run
class Stages:
    """Each stage a process under wrap.py (its rusage to OUT/stage_{name}.json), its tree
    watched by the monitor."""

    def __init__(self, out: Path, monitor, env: dict, cwd: Path, log):
        self.out, self.monitor, self.env, self.cwd, self.log = out, monitor, env, cwd, log
        self.procs: dict[str, subprocess.Popen] = {}
        self.order: list[str] = []

    def start(self, name: str, cmd: list[str], extra: dict | None = None, unset: tuple = (), nice: int = 0,
              set_: str | None = None) -> subprocess.Popen:
        env = {k: v for k, v in (self.env | (extra or {})).items() if k not in unset}
        if set_:
            env["WIKIDATA_CARD_FIGURES_DIR"] = str(Path(env["BENCH_FULL_CARD_FIGURES"]) / set_)
        full = [sys.executable, "-I", str(WRAP), str(self.out / f"stage_{name}.json"), "--",
                *(["nice", "-n", str(nice)] if nice else []), *cmd]
        print(f"[bench-full] {time.strftime('%H:%M:%S')} start {name}", flush=True)
        p = subprocess.Popen(full, cwd=self.cwd, env=env, stdout=self.log, stderr=subprocess.STDOUT)
        self.procs[name] = p
        self.order.append(name)
        self.monitor.register(name, p.pid)
        return p

    def wait(self, name: str) -> int:
        rc = self.procs[name].wait()
        self.monitor.unregister(name)
        print(f"[bench-full] {time.strftime('%H:%M:%S')} end {name} ({rc})", flush=True)
        return rc

    def kill(self, *names: str) -> None:
        """Stop each stage and every process under it (a worker pool's workers outlive
        their parent's SIGTERM otherwise), SIGTERM first, then SIGKILL."""
        from monitor import tree

        pids = [q for n in names if (p := self.procs.get(n)) and p.poll() is None for q in tree(p.pid)]
        for sig in (signal.SIGTERM, signal.SIGKILL):
            for q in pids:
                try:
                    os.kill(q, sig)
                except OSError:
                    pass
            if sig == signal.SIGTERM:
                time.sleep(3)
        for n in names:
            if n in self.procs:
                self.wait(n)


def preflight(a, work: Path, dump_bytes: int, dump_on_disk: bool) -> dict:
    free = shutil.disk_usage(work).free
    need = dump_bytes * a.disk_factor + (0 if dump_on_disk else dump_bytes) + DISK_MARGIN * (a.label == "full")
    mem = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    rep = {"free_bytes": free, "needed_bytes": int(need), "disk_factor": a.disk_factor, "dump_bytes": dump_bytes,
           "dump_already_on_disk": dump_on_disk, "memory_bytes": mem}
    print(f"[bench-full] disk: {free / 1e9:.1f} GB free at {work}; the run needs ~{need / 1e9:.1f} GB "
          f"({a.disk_factor:g}x the dump's {dump_bytes / 1e9:.2f} GB beside it"
          f"{'' if dump_on_disk else ', and the dump itself'})", flush=True)
    if free < need and not a.force:
        raise SystemExit(
            f"[bench-full] refusing: {free / 1e9:.1f} GB free at {work}, the run needs ~{need / 1e9:.1f} GB at its peak "
            f"(the dump, the chunk files, the build branch's group files held by the stand-in Hub, finalise's sort "
            f"scratch; bench/full/README.md, Disk). Free some space, put --work on a larger disk, or pass --force "
            f"(or a smaller --disk-factor) to try anyway.")
    if mem < 15e9:
        print(f"[bench-full] warning: {mem / 1e9:.1f} GB of memory, under the 16 GB the pipeline is sized for",
              flush=True)
    if (load := os.getloadavg()[0]) > (os.cpu_count() or 1):
        print(f"[bench-full] warning: load {load:.1f} on {os.cpu_count()} CPUs: other work will skew the figures",
              flush=True)
    return rep


def run(a) -> Path:
    sys.path.insert(0, str(HERE))
    from monitor import Monitor
    import report

    a.label = label_of(a)
    release = a.release
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"{stamp}-{a.label}-{a.source}-{a.orchestration}"
    out = (a.out or a.cache / "results" / name).resolve()
    out.mkdir(parents=True, exist_ok=True)
    work = (a.work or a.cache / "work").resolve() / name
    work.mkdir(parents=True)
    dump_dir = work / "releases" / release / "dump"
    dump_dir.mkdir(parents=True)
    dump = dump_dir / f"wikidata-{release}-all.json.bz2"
    cached = cached_dump(a.cache, release, a.label)
    meta: dict = {}
    if a.source == "local":
        if not cached.exists():
            raise SystemExit(f"[bench-full] {cached} is not cached: run `run.py fetch` with the same scale first")
        meta = json.loads(cached.with_name(cached.name + ".meta.json").read_text())
        try:
            os.link(cached, dump)
            linked = "hard link"
        except OSError:
            dump.symlink_to(cached)
            linked = "symlink"
        dump_bytes, on_disk = cached.stat().st_size, True
        print(f"[bench-full] the dump: {cached} ({linked})", flush=True)
    else:
        if cached.exists():  # (to compare: the same bytes, by sha256, once the run is done)
            meta = json.loads(cached.with_name(cached.name + ".meta.json").read_text())
        if a.label == "full":
            dump_bytes = RELEASE_DUMP_BYTES
        else:
            sys.path.insert(0, str(HERE))
            from splice import plan, remote_info

            info = remote_info(release)
            dump_bytes = sum(b - s for s, b in plan(info["bytes"], a.prefix_gb, a.windows, a.window_mb)[1])
        on_disk = False
    pre = preflight(a, work, dump_bytes, on_disk)

    hub = work / "fakehub"
    env = {**os.environ, "WIKIDATA_RELEASE": release, "BENCH_FULL_HUB": str(hub),
           "BENCH_FULL_SPANS": str(out / "spans.jsonl"), "BENCH_FULL_CARD_FIGURES": str(work / "card-figures"),
           "BENCH_FULL_CARDS": "1" if a.cards else "0", "WIKIDATA_CARD_FIGURES_DIR": str(work / "card-figures" / "main"),
           "PYTHONPATH": os.pathsep.join([str(SITE), *filter(None, [os.environ.get("PYTHONPATH")])]),
           # (no Hub: the stand-in answers every call; anything else is refused)
           "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1", "HF_ENDPOINT": "http://127.0.0.1:9"}
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "WIKIDATA_SCHOLAR", "WIKIDATA_FOLLOW_SPLIT", "WIKIDATA_PHASES"):
        env.pop(k, None)
    record = {"tool": "bench/full", "tool_version": VERSION, "name": name, "label": a.label, "release": release,
              "source": a.source, "orchestration": a.orchestration, "cards": a.cards,
              "started": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
              "work": str(work), "dump": {"path": str(cached) if a.source == "local" else str(dump), **meta},
              "preflight": pre, "machine": machine(work), "code": code(),
              "env": {k: v for k, v in os.environ.items() if k.startswith(ENV_PREFIXES) and "TOKEN" not in k},
              "config": config_of(env, work)}
    (out / "meta.json").write_text(json.dumps(record, indent=1, default=str))

    monitor = Monitor(work, out / "timeline.jsonl", interval=a.interval)
    if a.source == "stream":
        monitor.watch = {"part": dump.with_name(dump.name + ".part"), "dump": dump}
    # (the chunks the split has written, and each set's parts numbered for its run, as
    # they come: the time to the first processed chunk, and the work done meanwhile)
    data = work / "releases" / release / "data"
    monitor.lines = {"split_chunks": data / "manifest.split.jsonl", "main_parts": data / "manifest.jsonl",
                     "scholar_parts": work / "releases" / f"{release}-scholar" / "data" / "manifest.jsonl"}
    log = open(out / "run.log", "w")
    st = Stages(out, monitor, env, work, log)
    split_cmd = [str(BIN / "split-dump")]
    if a.source == "stream":
        if a.label == "full":
            download_cmd = [str(BIN / "download-dump")]
        else:
            download_cmd = [sys.executable, "-I", str(SPLICE), str(dump), "--release", release, *scale_args(a),
                            "--log", str(out / "net.jsonl"), "--meta", str(out / "stream_source.json")]
            if a.connections:
                download_cmd += ["--connections", str(a.connections)]
    run_cmd, fin_cmd = [str(BIN / "process-wikidata")], [sys.executable, str(HERE / "finalise_runs.py"), str(out)]
    early = (os.environ.get("WIKIDATA_FINALISE_EARLY") or "1") != "0"
    cross = (os.environ.get("WIKIDATA_FINALISE_CROSS_RUN") or "1") != "0"
    t0 = time.time()
    monitor.start()
    ok = False
    try:
        def download_split(also=()) -> None:
            """`just download-split`: a download that fails stops the split (and `also`)."""
            if a.source == "stream":
                if st.wait("download"):
                    st.kill("split", *also)
                    raise SystemExit("[bench-full] the download failed (see run.log)")
            if st.wait("split"):
                st.kill(*also)
                raise SystemExit("[bench-full] the split failed (see run.log)")
            # (a release has both sets; a small sample may not: the dump's first GB, its
            # early ids, holds no scholarly works, and finalise stops on an empty set)
            for set_, d in (("main", release), ("scholarly", f"{release}-scholar")):
                m = work / "releases" / d / "data" / "manifest.jsonl"
                if not (m.exists() and m.stat().st_size):
                    st.kill(*also)
                    raise SystemExit(f"[bench-full] the sample has no {set_} entities, and finalise needs both "
                                     "sets: a prefix's early ids hold no scholarly works. Take --windows, or a "
                                     "longer prefix")

        # (`just overlap-release` exports these for every process it starts, the
        # download's and the split's too)
        follow = ({"WIKIDATA_FOLLOW_SPLIT": "1", "WIKIDATA_WORKERS": os.environ.get("WIKIDATA_OVERLAP_WORKERS") or "2"}
                  if a.orchestration == "overlap" else {})
        if a.source == "stream":
            st.start("download", download_cmd, follow)
        st.start("split", split_cmd, follow)
        if a.orchestration == "overlap":  # (`just overlap-release`)
            st.start("run_scholar", run_cmd, follow | {"WIKIDATA_SCHOLAR": "1"}, nice=10, set_="scholar")
            st.start("run_main", run_cmd, follow | {"WIKIDATA_SCHOLAR": ""}, nice=10, set_="main")
            download_split(also=("run_scholar", "run_main"))
            code_ = 0 if st.wait("run_main") == 0 else 1
            unset = ("WIKIDATA_FOLLOW_SPLIT", "WIKIDATA_WORKERS")
            if code_ == 0 and early and cross:
                gate = work / ".gate"
                gate.mkdir()
                st.start("finalise", fin_cmd + ["finalise", "--after", str(gate / "scholar")],
                         {"WIKIDATA_SCHOLAR": ""}, unset=unset, nice=10)
                s = st.wait("run_scholar")
                (gate / "scholar.tmp").write_text("0" if s == 0 else "1")
                (gate / "scholar.tmp").replace(gate / "scholar")
                code_ |= int(s != 0) | int(st.wait("finalise") != 0)
            else:
                if code_ == 0 and early:
                    st.start("finalise_early", [str(BIN / "finalise-wikidata")], {"WIKIDATA_SCHOLAR": ""},
                             unset=unset, nice=10, set_="main")
                    code_ |= int(st.wait("finalise_early") != 0)
                code_ |= int(st.wait("run_scholar") != 0)
            if code_:
                raise SystemExit("[bench-full] overlap-release failed (see run.log)")
        else:  # (`just download-split`)
            download_split()
        # `just release` without its promotion: both sets' runs (after overlap-release,
        # they find their work done), then finalise's runs
        tail = "tail_" if a.orchestration == "overlap" else ""
        for set_, scholar in (("scholar", "1"), ("main", "")):
            st.start(f"{tail}run_{set_}", run_cmd, {"WIKIDATA_SCHOLAR": scholar}, set_=set_)
            if st.wait(f"{tail}run_{set_}"):
                raise SystemExit(f"[bench-full] run-release {set_} failed (see run.log)")
        st.start(f"{tail}finalise", fin_cmd + [f"{tail}finalise"], {"WIKIDATA_SCHOLAR": ""})
        if st.wait(f"{tail}finalise"):
            raise SystemExit("[bench-full] finalise failed (see run.log)")
        ok = True
    finally:
        wall = time.time() - t0
        monitor.stop()
        log.close()
        st.kill(*[n for n, p in st.procs.items() if p.poll() is None])
        record["ok"] = ok
        record["wall_s"] = round(wall, 2)
        record["stages_order"] = st.order
        record["t0"] = t0
        record["ended"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        done = {s: (work / "releases" / d / "state" / "finalise.done").exists()
                for s, d in (("main", release), ("scholar", f"{release}-scholar"))}
        record["finalised"] = done
        # The stand-in's logs, and after the run (untimed): its linked files unchanged,
        # the published files' digest
        for f in (hub / ".bench").glob("*.jsonl") if (hub / ".bench").is_dir() else []:
            shutil.copy(f, out / f"hub_{f.name}")
        record["post"] = report.post_checks(work, hub, release, digest=a.digest and ok)
        if a.source == "stream" and dump.exists():
            record["post"]["stream_dump_sha256"] = _hash(dump, "sha256")[0]
            if meta.get("sha256"):
                record["post"]["stream_matches_cached"] = record["post"]["stream_dump_sha256"] == meta["sha256"]
        if not meta.get("json_bytes") and ok and a.label != "full" and dump.exists():
            record["dump"]["json_bytes"] = json_bytes(dump)
        record["outputs"] = report.outputs(work, release)
        (out / "meta.json").write_text(json.dumps(record, indent=1, default=str))
        if a.source == "stream" and a.keep_dump and ok and dump.exists() and not cached.exists():
            cached.parent.mkdir(parents=True, exist_ok=True)
            if a.label != "full":  # (its record, as `fetch` writes one)
                kept = json.loads((out / "stream_source.json").read_text())
                kept |= {"json_bytes": record["dump"].get("json_bytes"), "kept_from_stream_run": name}
                cached.with_name(cached.name + ".meta.json").write_text(json.dumps(kept, indent=1))
            dump.replace(cached)
        if not a.keep:
            shutil.rmtree(work, ignore_errors=True)
    report.build(out)
    print(f"[bench-full] {out / 'result.json'}\n{(out / 'summary.md').read_text()}", flush=True)
    if not ok:
        raise SystemExit(1)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("fetch", "run"):
        p = sub.add_parser(name)
        p.add_argument("--release", required=True, help="the dump's date, e.g. 20261019")
        p.add_argument("--prefix-gb", type=float, help="only the dump's first N GB (shard 0's head)")
        p.add_argument("--windows", type=int, help="K evenly spaced windows of whole blocks across the dump")
        p.add_argument("--window-mb", type=float, default=32.0, help="each window's MB (default 32)")
        p.add_argument("--cache", type=Path, default=default_cache(),
                       help="the cached dumps and the results (default $WIKIDATA_BENCH_CACHE or ~/.cache/wikidata-pq/bench-full)")
        p.add_argument("--custom", help="a local bz2 dump cached as it is under this name (fetch it with --file)")
        p.add_argument("--force", action="store_true", help="run though the disk check refuses")
        if name == "fetch":
            p.add_argument("--from-file", type=Path, help="a scaled dump cut from a local copy of the dump")
            p.add_argument("--file", type=Path, help="with --custom: the bz2 dump to cache")
        else:
            p.add_argument("--source", choices=("local", "stream"), default="local")
            p.add_argument("--orchestration", choices=("overlap", "serial"), default="overlap")
            p.add_argument("--work", type=Path, help="where the run's files go (default CACHE/work); needs the disk")
            p.add_argument("--out", type=Path, help="the results' directory (default CACHE/results/NAME)")
            p.add_argument("--disk-factor", type=float, default=DISK_FACTOR,
                           help=f"the peak disk beside the dump, over its bytes (default {DISK_FACTOR})")
            p.add_argument("--cards", action=argparse.BooleanOptionalAction, default=None,
                           help="finalise's card figures and cards (default: on for the full dump, off scaled)")
            p.add_argument("--digest", action=argparse.BooleanOptionalAction, default=True,
                           help="hash the published files after the run (untimed)")
            p.add_argument("--keep", action="store_true", help="keep the run's work directory")
            p.add_argument("--keep-dump", action="store_true", help="streaming: cache the dump it fetched, for local runs")
            p.add_argument("--connections", type=int, help="streaming a scaled dump: its connections (1-3)")
            p.add_argument("--interval", type=float, default=1.0, help="the monitor's sampling interval, s")
    p = sub.add_parser("report")
    p.add_argument("out", type=Path)
    p.add_argument("other", type=Path, nargs="?")
    a = ap.parse_args(argv)
    if a.cmd == "report":
        sys.path.insert(0, str(HERE))
        import report

        report.build(a.out)
        if a.other:
            report.build(a.other)
            print(report.compare(a.out, a.other))
        else:
            print((a.out / "summary.md").read_text())
        return
    if sum(bool(x) for x in (a.prefix_gb, a.windows, a.custom)) > 1:
        raise SystemExit("--prefix-gb, --windows or --custom: one of them")
    if a.cmd == "run" and a.custom and a.source == "stream":
        raise SystemExit("--custom is a local dump: --source local")
    if a.cmd == "fetch":
        fetch(a)
    else:
        if a.cards is None:
            a.cards = label_of(a) == "full"
        run(a)


if __name__ == "__main__":
    main()
