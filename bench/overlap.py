"""The split and both sets' pipelines, overlapped and not, as a release runs them (bench
16.0.0): run apart from the timed run, after it (ab.run_once), so none of its figures
take this in.

- **Sequential** (`just release` after `just download-split`): the split (split-dump,
  routing at its end), then the scholarly set's run, then the main set's (main.run, as
  `process-wikidata` runs it, `workers` chunk workers each).
- **Overlapped** (`just overlap-release`): the split and both sets' runs at once, each run
  following the split as it numbers its parts (WIKIDATA_FOLLOW_SPLIT=1), at nice 10, with
  half the workers each. Only for a commit whose pipeline can follow the split
  (dump.SetManifest).

Each on the dump slice the timed run's split reads (made by split_stage, or cached in
BENCH_DUMP_SLICE), in a directory of its own, the Hub a stand-in that keeps each repo's
build branch on disk (FakeHub; no transfer, so no upload time), and a group a chunk (so
groups close during the split, as a release's do). Measured: each one's wall
from the split's start to its last run's end, and the peak RSS of every process at once
(the split's, the runs', their workers' and decoders', summed); and the bytes each set's
runs upload. Modelled, apart from them: those bytes at a release's size uploaded at
7 MB/s, the one upload rate on record (docs/journal/2026-10-05-sort-speed.md), each set's
scaled by its chunk files' bytes, the release's (release.RELEASE_SOURCE_BYTES) over the
sample's (its files whose names start with "s" the scholarly set's, as pipeline.py's
parity takes them), not by entities: the sample's c0 has about 8 times a release's mean
bytes a main entity, so entities would scale its uploads to about 3 times a release's.

Usage: pipeline.py --overlap <src> <out.json> <workers> <chunk files...>
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

# The one upload rate on record (docs/journal/2026-10-05-sort-speed.md), in bytes a second
UPLOAD_RATE = 7e6

RUN = """
import sys
from functools import partial
from pathlib import Path
sys.path.insert(0, {src!r})
sys.path.insert(0, {here!r})
from overlap import FakeHub
from wikidata import main
main.close_group = partial(main.close_group, api=FakeHub(Path("fakehub").resolve()))
main.run()
"""


class FakeHub:
    """The HfApi calls close_group makes, against a directory: a repo's branch's files at
    root/{repo}/{revision}/{path} (as scripts/test_group_restart.py's)."""

    def __init__(self, root: Path):
        self.root = root

    def _dir(self, repo_id, revision):
        return self.root / repo_id.replace("/", "__") / (revision or "main")

    def create_repo(self, repo_id, **kw):
        self._dir(repo_id, None).mkdir(parents=True, exist_ok=True)

    def list_repo_refs(self, repo_id, **kw):
        d = self.root / repo_id.replace("/", "__")
        return SimpleNamespace(branches=[SimpleNamespace(name=b.name) for b in d.iterdir()], tags=[])

    def create_branch(self, repo_id, branch, **kw):
        self._dir(repo_id, branch).mkdir(parents=True, exist_ok=True)

    def list_repo_files(self, repo_id, revision=None, **kw):
        d = self._dir(repo_id, revision)
        return [str(p.relative_to(d)) for p in d.rglob("*") if p.is_file()]

    def file_exists(self, repo_id, path, revision=None, **kw):
        return (self._dir(repo_id, revision) / path).exists()

    def upload_file(self, path_or_fileobj, path_in_repo, repo_id, revision=None, **kw):
        dst = self._dir(repo_id, revision) / path_in_repo
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(path_or_fileobj)

    def upload_folder(self, repo_id, folder_path, allow_patterns, revision=None, **kw):
        folder = Path(folder_path)
        for p in sorted(folder.glob(allow_patterns)):
            dst = self._dir(repo_id, revision) / p.relative_to(folder)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, dst.with_suffix(".part"))
            dst.with_suffix(".part").replace(dst)

    def get_paths_info(self, repo_id, paths, revision=None, **kw):
        d = self._dir(repo_id, revision)
        return [
            SimpleNamespace(path=p, size=(d / p).stat().st_size,
                            lfs=SimpleNamespace(sha256=hashlib.sha256((d / p).read_bytes()).hexdigest()))
            for p in paths if (d / p).exists()
        ]


def _tree_rss_mb(pid: int) -> int:
    """The RSS of `pid` and of every process under it, at any depth."""
    def rss(p: str) -> int:
        try:
            for line in Path(f"/proc/{p}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
        except OSError:
            pass
        return 0

    total, todo = 0, [str(pid)]
    while todo:
        p = todo.pop()
        total += rss(p)
        try:
            for task in Path(f"/proc/{p}/task").iterdir():
                try:
                    todo += (task / "children").read_text().split()
                except OSError:
                    pass
        except OSError:
            pass
    return total // 1024


def _start(src: Path, root: Path, name: str, env: dict, nice: int, log) -> subprocess.Popen:
    code = (f"import sys; sys.path.insert(0, {str(src)!r}); from wikidata import dump; dump.split()"
            if name == "split" else RUN.format(src=str(src), here=str(Path(__file__).parent)))
    cmd = [*(["nice", "-n", str(nice)] if nice else []), sys.executable, "-c", code]
    return subprocess.Popen(cmd, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)


def _env(name: str, workers: int, follow: bool) -> dict:
    # (a group a chunk, WIKIDATA_GROUP_SOURCE_GB at 1 MB: a release's groups are about 400
    # of its chunks and a set's slice two to four, so with a group a chunk the groups
    # close as the split runs, as at a release; in either mode, and ignored by a commit
    # before the fixed group rule, whose groups are then one a set)
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "WIKIDATA_WORKERS": str(workers), "WIKIDATA_GROUP_SOURCE_GB": "0.001"}
    for k in ("WIKIDATA_SCHOLAR", "WIKIDATA_FOLLOW_SPLIT", "WIKIDATA_PHASES", "KEEP"):
        env.pop(k, None)
    if name == "scholar":
        env["WIKIDATA_SCHOLAR"] = "1"
    if follow:
        env["WIKIDATA_FOLLOW_SPLIT"] = "1"
    return env


def _run(src: Path, root: Path, steps: list[list[tuple[str, dict, int]]], log) -> tuple[float, int]:
    """Run each step's processes (name, env, niceness) at once, the steps one after
    another: the wall from the first's start to the last's end, and the peak of every live
    process's RSS (and its children's, at any depth), summed."""
    t, peak = time.perf_counter(), 0
    for step in steps:
        procs = [_start(src, root, name, env, nice, log) for name, env, nice in step]
        while any(p.poll() is None for p in procs):
            peak = max(peak, sum(_tree_rss_mb(p.pid) for p in procs if p.poll() is None))
            time.sleep(0.2)
        if failed := [name for (name, _, _), p in zip(step, procs) if p.returncode]:
            raise RuntimeError(f"overlap stage: {failed} failed (see {log.name})")
    return round(time.perf_counter() - t, 2), peak


def _uploaded(hub: Path) -> dict[str, int]:
    """The bytes each set's repos got (a scholarly repo is wikidata-scholar-{table})."""
    out = {"main": 0, "scholar": 0}
    for f in hub.rglob("*.parquet"):
        out["scholar" if "__wikidata-scholar-" in str(f.relative_to(hub)) else "main"] += f.stat().st_size
    return out


def overlap(src: Path, out: Path, workers: int, dump_slice: Path, files: list[str]) -> None:
    """Both orchestrations' figures (see the module docstring) to `out`; `files` the
    sample's chunk files, for the uploads' scale."""
    from release import RELEASE_SOURCE_BYTES

    release = os.environ["WIKIDATA_RELEASE"]
    ask = (f"import sys; sys.path.insert(0, {str(src)!r}); from wikidata import dump; "
           "print(hasattr(dump, 'SetManifest'))")
    follows = subprocess.run([sys.executable, "-c", ask], capture_output=True, text=True,
                             check=True).stdout.strip() == "True"
    half = max(1, workers // 2)
    modes = {"overlap_seq": [[("split", _env("split", workers, False), 0)],
                             [("scholar", _env("scholar", workers, False), 0)],
                             [("main", _env("main", workers, False), 0)]]}
    if follows:
        modes["overlap"] = [[("split", _env("split", workers, False), 0),
                             ("scholar", _env("scholar", half, True), 10), ("main", _env("main", half, True), 10)]]
    res, by_set = {}, {}
    with open(out.with_suffix(".log"), "w") as log:
        for p, steps in modes.items():
            with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
                root = Path(tmp)
                dump_dir = root / "releases" / release / "dump"
                dump_dir.mkdir(parents=True)
                os.link(dump_slice, dump_dir / f"wikidata-{release}-all.json.bz2")
                res[f"{p}_wall"], res[f"{p}_peak_rss_mb"] = _run(src, root, steps, log)
                by_set[p] = _uploaded(root / "fakehub")
                res[f"{p}_upload_mb"] = round(sum(by_set[p].values()) / 1e6, 1)
    # Modelled, apart from the measured figures: a release's uploads at the rate on record,
    # each set's bytes scaled by its chunk files' (the release's over the sample's)
    sample = {"main": 0, "scholar": 0}
    for f in files:
        sample["scholar" if Path(f).name.startswith("s") else "main"] += Path(f).stat().st_size
    release_bytes = sum(n * RELEASE_SOURCE_BYTES[k] / sample[k] for k, n in by_set["overlap_seq"].items() if sample[k])
    res["upload_eta_modelled"] = round(release_bytes / UPLOAD_RATE)
    out.write_text(json.dumps(res))
