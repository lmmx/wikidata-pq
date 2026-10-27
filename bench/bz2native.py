"""The split with an in-process bz2 decoder (WIKIDATA_BZ2_DECODER=native, bench 17.14.0;
crates/mibz2, docs/journal/2026-10-27-bz2-decoder.md; or BENCH_BZ2_DECODER, another of
dump.IN_PROCESS_BZ2, `bz2native_decoder`) against lbzip2's: run apart
from the timed run, after it (ab.run_once), so none of its figures take this in, and only
for a commit with the native decoder (`dump.BZ2_DECODERS`). The timed run's split is
lbzip2's (the default), with its parity as ever.

On the timed run's slice, each in a directory of its own and a process of its own, as
pipeline.split_stage runs it (split-dump, then route-release):

- **the split by lbzip2 and by the native decoder,** in an order the round's seed picks:
  `bz2native_off_split_cpu`, `_wall`, `_rss_mb` and `bz2native_on_split_cpu`, `_wall`,
  `_rss_mb` (the CPU of the process and every process it waits for, lbzip2 among them;
  the peak RSS of it and its children);
- **the decoders alone,** the slice decoded to nothing: `bz2native_lbzip2_decode_cpu`,
  `_wall` and `bz2native_decode_cpu`, `_wall`; with the native decoder's threads and
  kernels (`bz2native_threads`, `bz2native_kernels`);
- **parity** (`bz2native_parity_ok`): every file the two splits wrote (both sets' chunks,
  built tables and manifests) the same, byte for byte (`bz2native_files`). ab.py exits 1
  if not.

Usage: pipeline.py --bz2-native <src> <out.json> <seed> (the slice is BENCH_DUMP_SLICE,
or dump-slice.json.bz2 beside out.json, made by the timed run)
"""

import hashlib
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# The in-process decoder alone (`backend`): the slice in, its bytes counted (its threads as
# dump.bz2_threads gives them; the wikidata package not imported, so its imports are not
# counted)
DECODE = """
import os, sys, threading
from wikidata_reshape import Bz2Decoder
d = Bz2Decoder(int(os.environ.get("WIKIDATA_BZ2_THREADS") or os.cpu_count() or 1), backend={backend!r})
def feed():
    with open({path!r}, "rb") as f, d:
        while b := f.read(8 << 20):
            d.write(b)
t = threading.Thread(target=feed)
t.start()
buf = memoryview(bytearray(16 << 20))
n = 0
while k := d.readinto(buf):
    n += k
t.join()
d.join()
print(n, d.threads, d.kernels)
"""


def timed(cmd: list[str], cwd: Path, env: dict, out, rss, stdin=None) -> dict:
    """A process's wall, its CPU time and its children's (RUSAGE_CHILDREN: it is waited for
    here, and it waits for its own), and the peak RSS of it and its children (`rss`:
    pipeline.tree_rss_mb)."""
    r0, t = resource.getrusage(resource.RUSAGE_CHILDREN), time.perf_counter()
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdin=stdin, stdout=out, stderr=subprocess.STDOUT)
    peak = 0
    while proc.poll() is None:
        try:
            peak = max(peak, rss(proc.pid))
        except (OSError, StopIteration):
            pass
        time.sleep(0.2)
    if proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, cmd[:3])
    r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {"wall": round(time.perf_counter() - t, 2),
            "cpu": round(r1.ru_utime + r1.ru_stime - r0.ru_utime - r0.ru_stime, 1), "rss_mb": peak}


def has_native(src: Path) -> bool:
    code = f"import sys; sys.path.insert(0, {str(src)!r}); from wikidata import dump; print(hasattr(dump, 'BZ2_DECODERS'))"
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip() == "True"


def files(root: Path) -> dict[str, str]:
    """Every file a split wrote under `root` (the dump aside): its sha256, by its path"""
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file() and p.parent.name != "dump"}


def split(src: Path, dump_slice: Path, work: Path, decoder: str, log, rss) -> tuple[dict, dict]:
    """split-dump then route-release (as pipeline.split_stage runs them) by `decoder`: its
    figures (both steps' CPU and wall summed, the larger peak RSS) and its files"""
    release = os.environ["WIKIDATA_RELEASE"]
    env = {k: v for k, v in os.environ.items() if k not in ("WIKIDATA_DUMP_FORMAT", "WIKIDATA_PHASES")}
    env["WIKIDATA_BZ2_DECODER"] = decoder
    d = work / "releases" / release / "dump"
    d.mkdir(parents=True)
    os.link(dump_slice, d / f"wikidata-{release}-all.json.bz2")
    code = f"import sys; sys.path.insert(0, {str(src)!r}); from wikidata import dump; dump.{{}}()"
    out = {"cpu": 0.0, "wall": 0.0, "rss_mb": 0}
    for step in ("split", "route"):
        t = timed([sys.executable, "-c", code.format(step)], work, env, log, rss)
        out = {"cpu": round(out["cpu"] + t["cpu"], 1), "wall": round(out["wall"] + t["wall"], 2),
               "rss_mb": max(out["rss_mb"], t["rss_mb"])}
    return out, files(work / "releases")


def bz2native_stage(src: Path, out: Path, seed: int, rss) -> None:
    if not has_native(src):
        out.write_text("{}")
        return
    native = os.environ.get("BENCH_BZ2_DECODER") or "native"
    dump_slice = Path(os.environ.get("BENCH_DUMP_SLICE") or out.parent / "dump-slice.json.bz2")
    res: dict = {}
    made = {}
    with open(out.with_suffix(".log"), "w") as log, tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        order = ["lbzip2", native] if seed % 2 else [native, "lbzip2"]
        for decoder in order:
            figures, made[decoder] = split(src, dump_slice, Path(tmp) / decoder, decoder, log, rss)
            p = "bz2native_on" if decoder == native else "bz2native_off"
            res |= {f"{p}_split_{k}": v for k, v in figures.items()}
        res["bz2native_files"] = len(made["lbzip2"])
        res["bz2native_decoder"] = native
        res["bz2native_parity_ok"] = made["lbzip2"] == made[native] and bool(made["lbzip2"])
        if not res["bz2native_parity_ok"]:
            diff = sorted(k for k in made["lbzip2"].keys() | made[native].keys()
                          if made["lbzip2"].get(k) != made[native].get(k))
            print(f"bz2native parity: {len(diff)} files differ: {diff[:20]}", file=sys.stderr)
        lbzip2 = subprocess.run([sys.executable, "-c", f"import sys; sys.path.insert(0, {str(src)!r}); "
                                 "from wikidata import dump; print(*dump.decoder('bz2'))"],
                                capture_output=True, text=True, check=True).stdout.split()
        with open(dump_slice, "rb") as f:
            t = timed(lbzip2, Path(tmp), dict(os.environ), subprocess.DEVNULL, rss, stdin=f)
        res["bz2native_lbzip2_decode_cpu"], res["bz2native_lbzip2_decode_wall"] = t["cpu"], t["wall"]
        decode_out = Path(tmp) / "decode.out"
        with open(decode_out, "w") as o:
            t = timed([sys.executable, "-c", DECODE.format(path=str(dump_slice), backend=native)], Path(tmp),
                      dict(os.environ), o, rss)
        n, threads, kernels = decode_out.read_text().split()
        res["bz2native_decode_cpu"], res["bz2native_decode_wall"] = t["cpu"], t["wall"]
        res["bz2native_threads"], res["bz2native_kernels"] = int(threads), kernels
    out.write_text(json.dumps(res))
