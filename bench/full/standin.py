"""The compute benchmark's stand-in Hub: every Hub call the pipeline and finalise make,
answered from a directory, with no transfer and nothing copied.

It answers the calls bench/overlap.py's FakeHub and bench/link.py's LinkHub answer (the
CI bench's stand-in), for the same layout (a repo's branch's files at
root/{owner}__{repo}/{revision}/{path}), but it is built for a whole dump on a laptop's
disk:

- **Nothing is copied.** A file uploaded (upload_folder, create_commit) or downloaded
  (snapshot_download) is hard-linked into place, so the pipeline's bytes are on the disk
  once, and a push or a download costs a link, not a copy. (A file on another file system
  is copied, and the log says so.) The pipeline never writes into a file it has
  uploaded: it replaces files. `placed.jsonl` records each linked file's inode, size and
  mtime, and run.py checks at the end that none changed.
- **Every transfer is logged, not timed.** `uploads.jsonl` has a line a call (its set,
  repo, kind, bytes, files) and `downloads.jsonl` a line for each download (the bytes a
  real Hub would have sent: the files not already local). run.py reports them as the
  bytes a release would move, apart from the timed figures.
- **The Hub's own hashing** (the sha256 the Hub reports for each file in list_repo_tree
  and get_paths_info, which a real Hub has computed on its side) is done here lazily, only
  when the pipeline reads it, once an inode, shared by every process through a cache
  directory. It is the stand-in's one cost in the timed path: `hashes.jsonl` logs each
  hash's bytes, wall and CPU time, and run.py reports their sum as the stand-in's
  overhead.

`hook(root)` installs it in a process before huggingface_hub is imported (from
hook/sitecustomize.py, on PYTHONPATH, when BENCH_FULL_HUB names the root): every
`HfApi()`, `snapshot_download` and `hf_hub_download` is then this. HF_HUB_OFFLINE=1 and an
unroutable HF_ENDPOINT (run.py sets both) refuse the network to anything else.
"""

from __future__ import annotations

import fnmatch
import hashlib
import importlib.abc
import importlib.machinery
import json
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

BENCH = ".bench"  # (the stand-in's own files, beside the repos)


def _set(repo_id: str) -> str:
    return "scholar" if "/wikidata-scholar-" in repo_id else "main"


def _table(repo_id: str) -> str:
    return repo_id.split("/", 1)[-1].removeprefix("wikidata-").removeprefix("scholar-")


class _Lfs:
    """A file's LFS info, its sha256 computed only when read."""

    def __init__(self, hub: "LocalHub", path: Path, size: int):
        self._hub, self._path, self.size = hub, path, size

    @property
    def sha256(self) -> str:
        return self._hub._sha256(self._path)


class LocalHub:
    """The HfApi calls of close_group and finalise, against `root` (see the module
    docstring)."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.logs = self.root / BENCH
        (self.logs / "sha256").mkdir(parents=True, exist_ok=True)
        self._hashes: dict[str, str] = {}
        self._lock = threading.Lock()
        self._pool: ThreadPoolExecutor | None = None

    # -- the stand-in's logs
    def _log(self, name: str, rec: dict) -> None:
        line = (json.dumps(rec) + "\n").encode()
        fd = os.open(self.logs / f"{name}.jsonl", os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, line)  # (one write, O_APPEND: whole lines from every process)
        finally:
            os.close(fd)

    def _upload(self, repo_id: str, kind: str, nbytes: int, files: int, **extra) -> None:
        self._log("uploads", {"t": time.time(), "set": _set(repo_id), "repo": repo_id, "table": _table(repo_id),
                              "kind": kind, "bytes": nbytes, "files": files, "pid": os.getpid(), **extra})

    def _download(self, repo_id: str, kind: str, nbytes: int, files: int) -> None:
        self._log("downloads", {"t": time.time(), "set": _set(repo_id), "repo": repo_id, "table": _table(repo_id),
                                "kind": kind, "bytes": nbytes, "files": files, "pid": os.getpid()})

    def _place(self, src: Path, dst: Path) -> None:
        """`src` at `dst`, hard-linked (copied across file systems), atomically."""
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".part")
        tmp.unlink(missing_ok=True)
        try:
            os.link(src, tmp)
            how = "link"
        except OSError:
            shutil.copyfile(src, tmp)
            how = "copy"
        tmp.replace(dst)
        st = dst.stat()
        self._log("placed", {"path": str(dst), "ino": st.st_ino, "size": st.st_size,
                             "mtime_ns": st.st_mtime_ns, "how": how})

    def _sha256(self, p: Path) -> str:
        st = p.stat()
        key = f"{st.st_dev}-{st.st_ino}-{st.st_size}-{st.st_mtime_ns}"
        if key in self._hashes:
            return self._hashes[key]
        cached = self.logs / "sha256" / key
        try:
            digest = cached.read_text()
        except OSError:
            w, c = time.perf_counter(), time.thread_time()
            h = hashlib.sha256()
            with open(p, "rb") as f:
                while b := f.read(1 << 20):
                    h.update(b)
            digest = h.hexdigest()
            self._log("hashes", {"bytes": st.st_size, "s": round(time.perf_counter() - w, 4),
                                 "cpu": round(time.thread_time() - c, 4), "pid": os.getpid()})
            tmp = cached.with_name(f"{key}.{os.getpid()}.{threading.get_ident()}")
            tmp.write_text(digest)
            tmp.replace(cached)
        self._hashes[key] = digest
        return digest

    def _dir(self, repo_id, revision) -> Path:
        return self.root / repo_id.replace("/", "__") / (revision or "main")

    def _info(self, d: Path, p: Path) -> SimpleNamespace:
        size = p.stat().st_size
        return SimpleNamespace(path=str(p.relative_to(d)), size=size, blob_id=None, lfs=_Lfs(self, p, size))

    # -- the calls (FakeHub's and LinkHub's)
    def create_repo(self, repo_id, **kw):
        self._dir(repo_id, None).mkdir(parents=True, exist_ok=True)

    def list_repo_refs(self, repo_id, **kw):
        d = self.root / repo_id.replace("/", "__")
        return SimpleNamespace(branches=[SimpleNamespace(name=b.name) for b in d.iterdir()] if d.is_dir() else [],
                               tags=[])

    def create_branch(self, repo_id, branch, **kw):
        self._dir(repo_id, branch).mkdir(parents=True, exist_ok=True)

    def list_repo_files(self, repo_id, revision=None, **kw):
        d = self._dir(repo_id, revision)
        return [str(p.relative_to(d)) for p in d.rglob("*") if p.is_file() and not p.name.endswith(".part")]

    def file_exists(self, repo_id, path, revision=None, **kw):
        return (self._dir(repo_id, revision) / path).exists()

    def upload_file(self, path_or_fileobj, path_in_repo, repo_id, revision=None, **kw):
        if isinstance(path_or_fileobj, (bytes, bytearray)):
            data = bytes(path_or_fileobj)
        elif isinstance(path_or_fileobj, (str, Path)):
            data = Path(path_or_fileobj).read_bytes()
        else:
            data = path_or_fileobj.read()
        dst = self._dir(repo_id, revision) / path_in_repo
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".part")
        tmp.write_bytes(data)
        tmp.replace(dst)
        self._upload(repo_id, "file", len(data), 1)

    def upload_folder(self, repo_id, folder_path, allow_patterns, revision=None, **kw):
        folder = Path(folder_path)
        files = sorted(p for p in folder.glob(allow_patterns) if p.is_file())
        for p in files:
            self._place(p, self._dir(repo_id, revision) / p.relative_to(folder))
        self._upload(repo_id, "group", sum(p.stat().st_size for p in files), len(files))

    def get_paths_info(self, repo_id, paths, revision=None, **kw):
        d = self._dir(repo_id, revision)
        return [self._info(d, d / p) for p in paths if (d / p).is_file()]

    def list_repo_tree(self, repo_id, path_in_repo=None, recursive=False, revision=None, **kw):
        d = self._dir(repo_id, revision)
        base = d / path_in_repo if path_in_repo else d
        for p in sorted(base.rglob("*") if recursive else base.iterdir()):
            if p.is_file() and not p.name.endswith(".part"):
                yield self._info(d, p)

    def preupload_lfs_files(self, repo_id, additions, revision=None, **kw):
        todo = [op for op in additions if not getattr(op, "_bench_sent", False)]
        self._upload(repo_id, "preupload", sum(_size(op.path_or_fileobj) for op in todo), len(todo))
        for op in todo:
            op._bench_sent = True

    def create_commit(self, repo_id, operations, commit_message=None, revision=None, run_as_future=False, **kw):
        if run_as_future:  # (as HfApi: one background thread, in order)
            with self._lock:
                self._pool = self._pool or ThreadPoolExecutor(1)
            return self._pool.submit(self.create_commit, repo_id, operations, commit_message, revision, **kw)
        adds = [op for op in operations if hasattr(op, "path_or_fileobj")]
        new = [op for op in adds if not getattr(op, "_bench_sent", False)]
        d = self._dir(repo_id, revision)
        deletes = 0
        for op in operations:
            dst = d / op.path_in_repo
            if hasattr(op, "path_or_fileobj"):
                src = op.path_or_fileobj
                if isinstance(src, (str, Path)):
                    self._place(Path(src), dst)
                else:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    dst.write_bytes(src if isinstance(src, (bytes, bytearray)) else src.read())
            elif hasattr(op, "src_path_in_repo"):  # (a copy: promote's, not run here)
                self._place(d / op.src_path_in_repo, dst)
            else:
                deletes += 1
                if dst.is_dir():
                    shutil.rmtree(dst)
                else:
                    dst.unlink(missing_ok=True)
        self._upload(repo_id, "commit", sum(_size(op.path_or_fileobj) for op in new), len(new), deletes=deletes)
        oid = hashlib.sha1(repr(sorted(op.path_in_repo for op in operations)).encode()).hexdigest()
        return SimpleNamespace(oid=oid, commit_message=commit_message)

    def hf_hub_download(self, repo_id, filename, revision=None, **kw):
        p = self._dir(repo_id, revision) / filename
        self._download(repo_id, "file", p.stat().st_size, 1)
        return str(p)

    def snapshot_download(self, repo_id, revision=None, local_dir=None, allow_patterns=None, **kw):
        """The branch's files matching `allow_patterns` placed in `local_dir`: a file there
        already (the same inode, or the same size and sha256) kept, as snapshot_download
        keeps one, the rest linked and logged as downloaded."""
        d, out = self._dir(repo_id, revision), Path(local_dir)
        pats = [allow_patterns] if isinstance(allow_patterns, str) else (allow_patterns or ["*"])
        moved = files = 0
        for p in sorted(d.rglob("*")) if d.is_dir() else []:
            rel = str(p.relative_to(d))
            if not p.is_file() or rel.endswith(".part") or not any(fnmatch.fnmatch(rel, pat) for pat in pats):
                continue
            dst = out / rel
            if dst.is_file() and (os.path.samefile(dst, p) or (
                    dst.stat().st_size == p.stat().st_size and self._sha256(dst) == self._sha256(p))):
                continue
            self._place(p, dst)
            moved += p.stat().st_size
            files += 1
        self._download(repo_id, "snapshot", moved, files)
        return str(out)


def _size(obj) -> int:
    if isinstance(obj, (bytes, bytearray)):
        return len(obj)
    if isinstance(obj, (str, Path)):
        return Path(obj).stat().st_size
    at = obj.tell()
    size = obj.seek(0, os.SEEK_END) - at
    obj.seek(at)
    return size


def patch_hub(root: Path) -> LocalHub:
    """The stand-in in place of huggingface_hub's Hub calls (as bench/link.py's install)."""
    import huggingface_hub
    import huggingface_hub.hf_api

    hub = LocalHub(root)

    class HfApi:  # (a class: annotations take `HfApi | None`)
        def __new__(cls, *args, **kwargs):
            return hub

    huggingface_hub.HfApi = huggingface_hub.hf_api.HfApi = HfApi
    huggingface_hub.snapshot_download = lambda repo_id, **kw: hub.snapshot_download(repo_id, **kw)
    huggingface_hub.hf_hub_download = lambda repo_id, filename, **kw: hub.hf_hub_download(repo_id, filename, **kw)
    return hub


# -- spans: the finalise steps' and group closes' wall and CPU, from wikidata.main
SPANS = ("close_group", "compact_table", "sort_table", "collect_refs_stage", "build_claims_labels",
         "update_stats", "write_cards")


_TICK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100


def _tree_cpu() -> float:
    """The CPU time of this process's descendants: those reaped (RUSAGE_CHILDREN) and
    every live one's, its own and its reaped descendants' (/proc/PID/stat utime, stime,
    cutime, cstime), at any depth. Its change over a step is the step's workers' CPU,
    a pool's live workers included."""
    import resource

    r = resource.getrusage(resource.RUSAGE_CHILDREN)
    total, todo = r.ru_utime + r.ru_stime, [os.getpid()]
    first = True
    while todo:
        pid = todo.pop()
        if not first:
            try:
                with open(f"/proc/{pid}/stat") as fh:
                    v = fh.read().rsplit(")", 1)[1].split()
                total += sum(int(x) for x in v[11:15]) / _TICK
            except (OSError, IndexError, ValueError):
                pass
        first = False
        try:
            for task in os.scandir(f"/proc/{pid}/task"):
                try:
                    with open(f"{task.path}/children") as fh:
                        todo += [int(c) for c in fh.read().split()]
                except OSError:
                    pass
        except OSError:
            pass
    return total


def _span(name: str, f, out: str):
    import functools

    @functools.wraps(f)
    def timed(*args, **kwargs):
        t0, c0, k0 = time.time(), time.process_time(), _tree_cpu()
        ok = False
        try:
            result = f(*args, **kwargs)
            ok = True
            return result
        finally:
            k1 = _tree_cpu()
            of = next((str(a) for a in args if isinstance(a, str)), None)
            if name == "close_group" and args:
                of = getattr(args[0], "name", of)
            rec = {"fn": name, "of": of, "set": "scholar" if os.environ.get("WIKIDATA_SCHOLAR") == "1" else "main",
                   "pid": os.getpid(), "t0": t0, "t1": time.time(), "ok": ok,
                   "cpu_self": round(time.process_time() - c0, 3), "cpu_children": round(max(0.0, k1 - k0), 3)}
            fd = os.open(out, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
            try:
                os.write(fd, (json.dumps(rec) + "\n").encode())
            finally:
                os.close(fd)
    return timed


def _patch_main(module) -> None:
    if out := os.environ.get("BENCH_FULL_SPANS"):
        for name in SPANS:
            if callable(f := module.__dict__.get(name)):
                module.__dict__[name] = _span(name, f, out)
    if os.environ.get("BENCH_FULL_CARDS") == "0":  # (as the CI bench: no cards' figures, no cards)
        module.update_stats, module.write_cards = (lambda: None), (lambda: {})


class _Finder(importlib.abc.MetaPathFinder):
    """Patch huggingface_hub (the stand-in) and wikidata.main (spans, cards) as each is
    imported, so a process that imports neither pays nothing."""

    def __init__(self, root: Path):
        self.root = root

    def find_spec(self, name, path, target=None):
        if name not in ("huggingface_hub", "wikidata.main"):
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None:
            return spec
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            if name == "huggingface_hub":
                patch_hub(self.root)
            else:
                _patch_main(module)
        spec.loader.exec_module = exec_and_patch
        return spec


def hook(root: Path) -> None:
    if "huggingface_hub" in sys.modules:  # (already imported: patch it now)
        patch_hub(root)
    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder(root))
