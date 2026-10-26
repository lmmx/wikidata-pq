"""An emulated network link for the bench's stand-in Hub (bench 17.2.0), and finalise run
over it, as `just release` runs it, after the timed run and outside its figures.

**The link.** The stand-in Hub (overlap.FakeHub: each repo's branches on disk) with
transfers that take time: each upload and download sleeps for its bytes at a release's
size over a stated rate, compressed to the sample's time. A transfer of `b` sample bytes
of a set sleeps

    b * scale[set] / rate / time_scale

- `scale[set]`, bytes: a release's chunk files' bytes over the sample's
  (release.RELEASE_SOURCE_BYTES over the tier's chunk files, the scholarly set's those
  whose names start with "s", as overlap.upload_eta_modelled scales the uploads). The
  sample's group files and its published files are each scaled by it: both are about
  their chunk files' bytes, at the sample as at a release.
- `rate`: the release's link, in bytes a second: an upload rate (LINK_RATES), or
  DOWNLOAD_RATE for a download.
- `time_scale`: a release's seconds a sample second stands for, in the stage the link
  serves (TIME_SCALE): a release's modelled compute in it over the sample's wall of it
  without the link, at master. With it, the sample's transfers are about the same share
  of its wall as a release's are of a release's.

One link each way, shared by every process and thread of a stage (a file lock): two
transfers the same way at once take as long as one after the other, as on one link; an
upload and a download don't wait for each other. Every transfer is logged (its bytes and
sleep) so the stage reads the link's busy time. Nothing here reaches the Hub: the
stand-in replaces huggingface_hub's HfApi, snapshot_download and hf_hub_download before
wikidata is imported, and HF_HUB_OFFLINE=1 refuses the network to anything else.

**Finalise over the link** (`finalise_stage`): `finalise-wikidata` (main.finalise) as
`just release` runs it after `just overlap-release`, the main set, the scholarly set, the
main set again (claims_labels), each a process of its own, on a copy of the overlap
stage's output (both sets' runs done, their groups on the stand-in's build branches, the
group files deleted locally as WIKIDATA_KEEP_GROUP_FILES' default does): compaction's
download of the group files from the build branch, the sort, each table's commit and
verification through the stand-in, claims_labels' build (not the cards' figures or the
cards: their figures sample entities the bench's sample doesn't have). Whatever order
master's finalise runs its transfers in (today, `sort_table`'s per-table write, commit,
verify, the CPU idle during each commit), the stage runs them in, so a change to it
shows in the stage's wall.
"""

import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent

# The upload rates the stages run at, in bytes a second at a release: 7 MB/s, the one on
# record (docs/journal/2026-10-05-sort-speed.md), and 60 MB/s, a fast link
LINK_RATES = (7e6, 60e6)
# The download rate: none on record. 75 MB/s, the middle of the 50-100 MB/s assumed for
# the build branch's download (docs/journal/2026-10-25-release-critical-path.md, E1)
DOWNLOAD_RATE = 75e6
# A release's seconds a sample second stands for, by stage, at master (c45c0cd), on the
# quick tier: a release's modelled compute in the stage over the sample's wall of it
# without the link, means over CI runs' jobs, both arms (CHANGELOG 17.2.0):
# - "overlap": the split and both sets' runs, release_compute_end_modelled (bench
#   17.2.0's model) over overlap_wall: 1,046 in #165's run 40241412346, 994 in #163's
#   40248725557 (12 jobs each);
# - "finalise": finalise's three runs, finalise_eta + refs_eta + CLAIMS_LABELS_BUILD
#   (7,840 s in those runs) over finalise_link_0_wall (this stage's, on CI).
# The full tier's sample is larger: each is scaled by the quick tier's chunk files' bytes
# over the tier's (TIME_SCALE_BYTES; a stage's compute on the sample taken in proportion
# to its bytes, as no full-tier run at bench 17 is on record)
TIME_SCALE = {"overlap": 1020.0, "finalise": 217.0}
TIME_SCALE_BYTES = 76_465_617  # the quick tier's chunk files (c0, r50, s30, s90)
# claims_labels' build at a release, in neither ETA: 13-22 CPU-min for both sets (#164),
# about 5 min on 4 cores (docs/journal/2026-10-25-release-critical-path.md, E3)
CLAIMS_LABELS_BUILD = 300


def time_scale(stage: str, sample_bytes: int) -> float:
    return TIME_SCALE[stage] * TIME_SCALE_BYTES / sample_bytes


def _set(repo_id: str) -> str:
    return "scholar" if "/wikidata-scholar-" in repo_id else "main"


def _size(obj) -> int:
    if isinstance(obj, (bytes, bytearray)):
        return len(obj)
    if isinstance(obj, (str, Path)):
        return Path(obj).stat().st_size
    at = obj.tell()  # (a file object: its size, left where it was)
    size = obj.seek(0, os.SEEK_END) - at
    obj.seek(at)
    return size


class Link:
    """A link each way at a release's rate, its time compressed to the sample's
    (`cfg`: up, down, time_scale, scale, dir; None: no link, no time)."""

    def __init__(self, cfg: dict | None):
        self.cfg = cfg

    def transfer(self, way: str, nbytes: int, repo_id: str, kind: str) -> None:
        if not self.cfg or not nbytes:
            return
        c, set_ = self.cfg, _set(repo_id)
        s = nbytes * c["scale"][set_] / c[way] / c["time_scale"]
        d = Path(c["dir"])
        with open(d / f"{way}.lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            t = time.perf_counter()
            time.sleep(s)
            t = time.perf_counter() - t
        with open(d / "transfers.jsonl", "a") as log:
            log.write(json.dumps({"way": way, "set": set_, "repo": repo_id, "kind": kind,
                                  "bytes": nbytes, "s": round(t, 4), "end": time.time()}) + "\n")


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    shutil.copyfile(src, tmp)
    tmp.replace(dst)


class LinkHub:
    """overlap.FakeHub (its calls: close_group's) with every transfer over a Link, and
    finalise's calls besides: list_repo_tree, create_commit (and run_as_future),
    preupload_lfs_files, hf_hub_download, snapshot_download."""

    def __init__(self, root: Path, cfg: dict | None):
        from overlap import FakeHub

        self.hub, self.link, self._hashes = FakeHub(root), Link(cfg), {}
        self._pool = None

    def __getattr__(self, name):  # (close_group's calls without a transfer: FakeHub's)
        return getattr(self.hub, name)

    def _dir(self, repo_id, revision):
        return self.hub._dir(repo_id, revision)

    def upload_file(self, path_or_fileobj, path_in_repo, repo_id, revision=None, **kw):
        data = path_or_fileobj if isinstance(path_or_fileobj, (bytes, bytearray)) else Path(path_or_fileobj).read_bytes()
        self.link.transfer("up", len(data), repo_id, "file")
        dst = self._dir(repo_id, revision) / path_in_repo
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".part")
        tmp.write_bytes(data)
        tmp.replace(dst)

    def upload_folder(self, repo_id, folder_path, allow_patterns, revision=None, **kw):
        files = sorted(Path(folder_path).glob(allow_patterns))
        self.link.transfer("up", sum(p.stat().st_size for p in files), repo_id, "group")
        self.hub.upload_folder(repo_id, folder_path, allow_patterns, revision=revision, **kw)

    def _sha256(self, p: Path) -> str:
        st = p.stat()
        key = (str(p), st.st_ino, st.st_size, st.st_mtime_ns)
        if key not in self._hashes:
            h = hashlib.sha256()
            with open(p, "rb") as f:
                while b := f.read(1 << 20):
                    h.update(b)
            self._hashes[key] = h.hexdigest()
        return self._hashes[key]

    def list_repo_tree(self, repo_id, path_in_repo=None, recursive=False, revision=None, **kw):
        d = self._dir(repo_id, revision)
        base = d / path_in_repo if path_in_repo else d
        for p in sorted(base.rglob("*") if recursive else base.iterdir()):
            if p.is_file() and not p.name.endswith(".part"):
                yield SimpleNamespace(path=str(p.relative_to(d)), size=p.stat().st_size, blob_id=None,
                                      lfs=SimpleNamespace(sha256=self._sha256(p), size=p.stat().st_size))

    def preupload_lfs_files(self, repo_id, additions, revision=None, **kw):
        todo = [op for op in additions if not getattr(op, "_is_uploaded", False)]
        self.link.transfer("up", sum(_size(op.path_or_fileobj) for op in todo), repo_id, "commit")
        for op in todo:
            op._is_uploaded = True

    def create_commit(self, repo_id, operations, commit_message=None, revision=None, run_as_future=False, **kw):
        if run_as_future:  # (as HfApi: one background thread, in order)
            self._pool = self._pool or ThreadPoolExecutor(1)
            return self._pool.submit(self.create_commit, repo_id, operations, commit_message, revision, **kw)
        adds = [op for op in operations if hasattr(op, "path_or_fileobj")]
        self.link.transfer("up", sum(_size(op.path_or_fileobj) for op in adds
                                     if not getattr(op, "_is_uploaded", False)), repo_id, "commit")
        d = self._dir(repo_id, revision)
        for op in operations:
            dst = d / op.path_in_repo
            if hasattr(op, "path_or_fileobj"):
                src = op.path_or_fileobj
                if isinstance(src, (str, Path)):
                    _copy(Path(src), dst)
                else:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    dst.write_bytes(src if isinstance(src, (bytes, bytearray)) else src.read())
            elif dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink(missing_ok=True)
        return SimpleNamespace(oid=hashlib.sha1(repr(sorted(op.path_in_repo for op in operations)).encode()).hexdigest(),
                               commit_message=commit_message)

    def hf_hub_download(self, repo_id, filename, revision=None, **kw):
        p = self._dir(repo_id, revision) / filename
        self.link.transfer("down", p.stat().st_size, repo_id, "file")
        return str(p)

    def snapshot_download(self, repo_id, revision=None, local_dir=None, allow_patterns=None, **kw):
        """The branch's files matching `allow_patterns` copied to `local_dir`, a file
        already there with the branch's size and sha256 kept, as snapshot_download keeps
        one, the rest over the link."""
        import fnmatch

        d, out = self._dir(repo_id, revision), Path(local_dir)
        pats = [allow_patterns] if isinstance(allow_patterns, str) else (allow_patterns or ["*"])
        todo = []
        for p in sorted(d.rglob("*")):
            rel = str(p.relative_to(d))
            if p.is_file() and not rel.endswith(".part") and any(fnmatch.fnmatch(rel, pat) for pat in pats):
                dst = out / rel
                if not (dst.is_file() and dst.stat().st_size == p.stat().st_size
                        and self._sha256(dst) == self._sha256(p)):
                    todo.append((p, dst))
        self.link.transfer("down", sum(p.stat().st_size for p, _ in todo), repo_id, "snapshot")
        for p, dst in todo:
            _copy(p, dst)
        return str(out)


def install(root: Path) -> LinkHub:
    """The stand-in in place of huggingface_hub's Hub calls, before wikidata is imported
    (so every `from huggingface_hub import HfApi` takes it), its link from BENCH_LINK."""
    import huggingface_hub
    import huggingface_hub.hf_api

    cfg = json.loads(os.environ["BENCH_LINK"]) if os.environ.get("BENCH_LINK") else None
    hub = LinkHub(root, cfg)

    class HfApi:  # (a class: annotations take `HfApi | None`)
        def __new__(cls, *args, **kwargs):
            return hub

    huggingface_hub.HfApi = huggingface_hub.hf_api.HfApi = HfApi
    huggingface_hub.snapshot_download = lambda repo_id, **kw: hub.snapshot_download(repo_id, **kw)
    huggingface_hub.hf_hub_download = lambda repo_id, filename, **kw: hub.hf_hub_download(repo_id, filename, **kw)
    return hub


FINALISE = """
import sys
from pathlib import Path
sys.path.insert(0, {src!r})
sys.path.insert(0, {here!r})
from link import install
install(Path("fakehub").resolve())
from wikidata import main
# (the cards' figures sample entities the bench's sample doesn't have; at a release they
# and the cards are a few small files)
main.update_stats, main.write_cards = (lambda: None), (lambda: {{}})
main.finalise()
"""

# `just release`'s finalise runs, in order
RUNS = ("main", "scholar", "main")


def _copy_tree(src: Path, dst: Path) -> None:
    """A copy of an overlap run's directory to finalise: Parquet files linked (finalise
    replaces files, never writes into one), the rest (ledgers, appended to) copied."""
    def copy(s, d):
        (os.link if s.endswith(".parquet") or s.endswith(".bz2") else shutil.copy2)(s, d)
    shutil.copytree(src, dst, copy_function=copy, symlinks=True)


def _published(hub: Path, revision: str) -> dict[str, dict[str, int]]:
    """The bytes of each set's published (sorted) files on its build branch, by table."""
    out: dict[str, dict[str, int]] = {"main": {}, "scholar": {}}
    for repo in hub.iterdir():
        d = repo / revision
        if not d.is_dir():
            continue
        name = repo.name.split("__", 1)[1]
        set_ = "scholar" if name.startswith("wikidata-scholar-") else "main"
        table = name.removeprefix("wikidata-").removeprefix("scholar-")
        out[set_][table] = sum(p.stat().st_size for p in d.rglob("*.parquet"))
    return out


def _digest(hub: Path, revision: str) -> str:
    h = hashlib.sha256()
    for p in sorted(hub.rglob("*.parquet")):
        if revision in p.parts:
            h.update(f"{p.relative_to(hub)}\0".encode())
            h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def finalise_stage(src: Path, base: Path, work: Path, rate: float | None, scale: dict[str, float],
                   sample_bytes: int, workers: int, log) -> dict:
    """finalise's three runs (RUNS) on a copy of `base` (an overlap run's directory), over
    the link at upload `rate` (None: no link), in `work`: their wall, each run's, the
    link's busy time each way, and the bytes published."""
    from overlap import _env

    release = os.environ["WIKIDATA_RELEASE"]
    root = work / "root"
    _copy_tree(base, root)
    linkdir = work / "link"
    linkdir.mkdir()
    cfg = None if rate is None else {"up": rate, "down": DOWNLOAD_RATE, "scale": scale, "dir": str(linkdir),
                                     "time_scale": time_scale("finalise", sample_bytes)}
    walls = []
    for i, name in enumerate(RUNS):
        env = _env(name, workers, False) | {"WIKIDATA_FINALISE_WORKERS": str(workers),
                                          "WIKIDATA_CARD_FIGURES_DIR": str(root / "card-figures" / name)}
        if cfg:
            env["BENCH_LINK"] = json.dumps(cfg)
        code = FINALISE.format(src=str(src), here=str(HERE))
        t = time.perf_counter()
        p = subprocess.run([sys.executable, "-c", code], cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
        walls.append(round(time.perf_counter() - t, 2))
        if p.returncode:
            raise RuntimeError(f"finalise over the link: {name}'s run {i + 1} failed (see {log.name})")
    for name, d in (("main", release), ("scholar", f"{release}-scholar")):  # (main.FINALISE_DONE)
        if not (root / "releases" / d / "state" / "finalise.done").exists():
            raise RuntimeError(f"finalise over the link: the {name} set not finalised (see {log.name})")
    busy = {"up": 0.0, "down": 0.0}
    moved = {"up": 0, "down": 0}
    if (linkdir / "transfers.jsonl").exists():
        for line in (linkdir / "transfers.jsonl").read_text().splitlines():
            r = json.loads(line)
            busy[r["way"]] += r["s"]
            moved[r["way"]] += r["bytes"]
    return {"wall": round(sum(walls), 2), "walls": walls, "up_s": round(busy["up"], 2),
            "down_s": round(busy["down"], 2), "up_mb": round(moved["up"] / 1e6, 1),
            "down_mb": round(moved["down"] / 1e6, 1),
            "published": _published(root / "fakehub", f"build-{release}"),
            "digest": _digest(root / "fakehub", f"build-{release}")}
