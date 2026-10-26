"""THE USER'S TO RUN. No agent runs this: it uploads to the Hugging Face Hub.

A one-off measurement for L1b (docs/journal/2026-10-26-l1b-preupload-evidence.md). It
times `HfApi.preupload_lfs_files` of synthetic files into a scratch repo of yours, and
checks the semantics the journal entry reads from the source:

1. **rate:** N files of random bytes (incompressible, like zstd Parquet pages, and new
   each run, so nothing deduplicates), preuploaded `--batch` files a call, each call
   timed: its wall, its bytes, its MB/s;
2. **a call's fixed cost:** `--small` files of 1 MB, one a call, so the per-call cost
   (the /preupload request, the Xet token, the shard upload, the finalise) shows apart
   from the bytes (the model's `o`);
3. **visibility:** the repo's files at the revision listed after the preuploads and
   before the commit (none of them should be there);
4. **a rerun:** the same files preuploaded again from fresh operations, as a rerun of
   finalise after a kill would (Xet should deduplicate them: little sent);
5. **a kill mid-call** (`--kill-after S`): a child process preuploads fresh files and is
   SIGKILLed after S seconds; the parent then preuploads the same files from fresh
   operations, and the bytes it sends show what the kill cost (Xet's resumed session,
   HF_XET_CACHE's staging, or a re-send);
6. **the commit:** one `create_commit` of every preuploaded file on the scratch revision,
   timed, then the files' sha256 and size on the Hub compared with the local ones (as
   verify_table does). `--no-commit` skips it (the uploads are then never referenced).

Bytes sent are the machine's (psutil's count, else /proc/net/dev's), so run nothing else
on the link meanwhile. Each figure is printed and written to `--out` (JSON).

Before running:
- create a private scratch dataset repo for it, e.g. `hf repos create <you>/l1b-scratch
  --type dataset --private`, or pass `--create` (refused unless the name has "scratch");
- have about N x size of free disk in `--dir` (default: a temporary directory);
- note that HF_XET_CACHE (default ~/.cache/huggingface/xet) gets the session's shards.

Examples (each a separate run; the knobs are read when huggingface_hub / hf_xet load):

    python preupload_measure.py --repo <you>/l1b-scratch --n 8 --size-mb 256
    HF_XET_HIGH_PERFORMANCE=1 python preupload_measure.py --repo <you>/l1b-scratch
    HF_HUB_DISABLE_XET=1 python preupload_measure.py --repo <you>/l1b-scratch --n 4
    python preupload_measure.py --repo <you>/l1b-scratch --kill-after 20

Afterwards, delete the scratch repo (`hf repos delete <you>/l1b-scratch --type dataset`)
if you no longer want it; nothing here deletes anything on the Hub.
"""

import argparse
import hashlib
import json
import os
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, __version__
from huggingface_hub.utils._runtime import is_xet_available

MB = 1_000_000


def sent() -> int:
    """The machine's bytes sent so far: psutil's count if installed, else /proc/net/dev's
    (Linux), else 0 (no byte counts then)."""
    try:
        import psutil

        return psutil.net_io_counters().bytes_sent
    except ImportError:
        pass
    try:
        lines = Path("/proc/net/dev").read_text().splitlines()[2:]
    except OSError:
        return 0
    return sum(int(line.split(":", 1)[1].split()[8]) for line in lines if not line.strip().startswith("lo:"))


def write_files(d: Path, n: int, size: int, prefix: str) -> list[Path]:
    """n files of `size` random bytes each, written in 8 MiB blocks."""
    d.mkdir(parents=True, exist_ok=True)
    out = []
    for i in range(n):
        p = d / f"{prefix}-{i:04d}.bin"
        with p.open("wb") as f:
            left = size
            while left:
                block = os.urandom(min(left, 8 << 20))
                f.write(block)
                left -= len(block)
        out.append(p)
    return out


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def ops(files: list[Path], tag: str) -> list[CommitOperationAdd]:
    return [CommitOperationAdd(path_in_repo=f"l1b/{tag}/{p.name}", path_or_fileobj=str(p)) for p in files]


def preupload(api: HfApi, a, operations: list[CommitOperationAdd], batch: int) -> list[dict]:
    """Preupload `operations` `batch` at a time: each call's wall, bytes, MB/s, bytes sent."""
    calls = []
    for i in range(0, len(operations), batch):
        chunk = operations[i:i + batch]
        nbytes = sum(op.upload_info.size for op in chunk)
        s0, t0 = sent(), time.perf_counter()
        api.preupload_lfs_files(a.repo, additions=chunk, repo_type="dataset", revision=a.revision,
                                free_memory=False)
        wall = time.perf_counter() - t0
        calls.append({"files": len(chunk), "bytes": nbytes, "wall_s": round(wall, 3),
                      "mb_s": round(nbytes / MB / wall, 2), "sent_bytes": sent() - s0,
                      "uploaded": [op._is_uploaded for op in chunk].count(True)})
        print(f"  call {len(calls)}: {len(chunk)} files, {nbytes / MB:.0f} MB in {wall:.1f} s "
              f"({nbytes / MB / wall:.1f} MB/s), {calls[-1]['sent_bytes'] / MB:.0f} MB sent", flush=True)
    return calls


def summary(calls: list[dict]) -> dict:
    nbytes, wall = sum(c["bytes"] for c in calls), sum(c["wall_s"] for c in calls)
    return {"calls": len(calls), "bytes": nbytes, "wall_s": round(wall, 3),
            "mb_s": round(nbytes / MB / wall, 2) if wall else None,
            "sent_bytes": sum(c["sent_bytes"] for c in calls)}


def child(a) -> None:
    """(--_child) Preupload the given files, to be killed mid-call by the parent."""
    api = HfApi()
    files = [Path(p) for p in a.files]
    api.preupload_lfs_files(a.repo, additions=ops(files, "kill"), repo_type="dataset", revision=a.revision)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--repo", required=True, help="a scratch dataset repo of yours")
    p.add_argument("--revision", default="main", help="the scratch repo's branch to commit to")
    p.add_argument("--create", action="store_true", help="create the repo (private) if missing")
    p.add_argument("--n", type=int, default=8, help="files for the rate")
    p.add_argument("--size-mb", type=int, default=256, help="each file's MB")
    p.add_argument("--batch", type=int, default=4, help="files a preupload call")
    p.add_argument("--small", type=int, default=16, help="1 MB files, one a call (0: skip)")
    p.add_argument("--kill-after", type=float, default=0, help="seconds before the child's SIGKILL (0: skip)")
    p.add_argument("--no-commit", action="store_true")
    p.add_argument("--dir", type=Path, default=None, help="where the synthetic files go")
    p.add_argument("--out", type=Path, default=Path("preupload_measure.json"))
    p.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("files", nargs="*", help=argparse.SUPPRESS)
    a = p.parse_args()
    if a._child:
        return child(a)

    if a.create and "scratch" not in a.repo:
        sys.exit("refusing --create for a repo whose name has no 'scratch' in it")
    api = HfApi()
    if a.create:
        api.create_repo(a.repo, repo_type="dataset", private=True, exist_ok=True)
    if a.revision != "main":
        api.create_branch(a.repo, repo_type="dataset", branch=a.revision, exist_ok=True)

    knobs = {k: v for k, v in os.environ.items() if k.startswith(("HF_XET_", "HF_HUB_"))}
    report = {"huggingface_hub": __version__, "xet": is_xet_available(), "env": knobs,
              "repo": a.repo, "revision": a.revision, "n": a.n, "size_mb": a.size_mb, "batch": a.batch}
    print(f"huggingface_hub {__version__}, Xet path: {is_xet_available()}, knobs: {knobs or 'defaults'}")

    work = Path(tempfile.mkdtemp(prefix="l1b-", dir=a.dir))
    try:
        tag = time.strftime("%Y%m%dT%H%M%S")
        print(f"writing {a.n} x {a.size_mb} MB of random bytes to {work}", flush=True)
        big = write_files(work / "big", a.n, a.size_mb * MB, "big")
        hashes = {f"l1b/{tag}/{p.name}": (sha256(p), p.stat().st_size) for p in big}

        print("1. the rate", flush=True)
        big_ops = ops(big, tag)
        report["rate"] = {"calls": preupload(api, a, big_ops, a.batch)}
        report["rate"]["all"] = summary(report["rate"]["calls"])

        committed = list(big_ops)
        if a.small:
            print(f"2. a call's fixed cost ({a.small} files of 1 MB, one a call)", flush=True)
            small = write_files(work / "small", a.small, MB, "small")
            small_ops = ops(small, tag)
            hashes.update({f"l1b/{tag}/{p.name}": (sha256(p), p.stat().st_size) for p in small})
            report["small"] = {"calls": preupload(api, a, small_ops, 1)}
            report["small"]["all"] = summary(report["small"]["calls"])
            committed += small_ops

        print("3. visibility before the commit", flush=True)
        listed = set(api.list_repo_files(a.repo, repo_type="dataset", revision=a.revision))
        report["visible_before_commit"] = sorted(set(hashes) & listed)
        print(f"  preuploaded files listed at {a.revision}: {len(report['visible_before_commit'])}")

        print("4. a rerun: the same files from fresh operations", flush=True)
        report["rerun"] = {"calls": preupload(api, a, ops(big, tag), a.batch)}
        report["rerun"]["all"] = summary(report["rerun"]["calls"])

        if a.kill_after:
            print(f"5. a kill {a.kill_after:g} s into a call, then the same files again", flush=True)
            kill = write_files(work / "kill", a.n, a.size_mb * MB, "kill")
            s0 = sent()
            proc = subprocess.Popen([sys.executable, __file__, "--repo", a.repo, "--revision", a.revision,
                                     "--_child", *map(str, kill)], start_new_session=True)
            time.sleep(a.kill_after)
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            before = sent() - s0
            calls = preupload(api, a, ops(kill, "kill"), a.batch)
            report["kill"] = {"after_s": a.kill_after, "sent_before_kill_bytes": before,
                              "bytes": sum(p.stat().st_size for p in kill), "rerun": summary(calls)}
            print(f"  sent before the kill {before / MB:.0f} MB; the rerun sent "
                  f"{report['kill']['rerun']['sent_bytes'] / MB:.0f} MB of {report['kill']['bytes'] / MB:.0f} MB")

        if not a.no_commit:
            print("6. the commit", flush=True)
            s0, t0 = sent(), time.perf_counter()
            info = api.create_commit(a.repo, repo_type="dataset", revision=a.revision, operations=committed,
                                     commit_message=f"l1b measure {tag}")
            report["commit"] = {"wall_s": round(time.perf_counter() - t0, 3), "sent_bytes": sent() - s0,
                                "oid": info.oid, "files": len(committed)}
            remote = {f.path: f for f in api.list_repo_tree(a.repo, repo_type="dataset", revision=a.revision,
                                                            path_in_repo=f"l1b/{tag}", recursive=True)
                      if hasattr(f, "lfs")}
            bad = [path for path, (sha, size) in hashes.items()
                   if path not in remote or remote[path].size != size
                   or (remote[path].lfs.sha256 if remote[path].lfs else None) != sha]
            report["commit"]["mismatched"] = bad
            print(f"  committed {len(committed)} files in {report['commit']['wall_s']:.1f} s, "
                  f"{report['commit']['sent_bytes'] / MB:.1f} MB sent; mismatched on the Hub: {bad or 'none'}")
    finally:
        shutil.rmtree(work, ignore_errors=True)
        report["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
        a.out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"written {a.out}")


if __name__ == "__main__":
    main()
