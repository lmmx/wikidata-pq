"""The split and both sets' pipelines, overlapped and not, as a release runs them (bench
17.0.0): run apart from the timed run, after it (ab.run_once), so none of its figures
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

Then (bench 17.2.0, link.py), apart from those: the overlapped runs again with their
pushes over an emulated link, and finalise's three runs on the overlapped runs' output
through the stand-in, without the link and over it; and in release_wall, a release's wall
with finalise's transfers at several upload rates.

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
# (bench 17.2.0: the same, its pushes over the emulated link, link.py)
RUN_LINK = """
import json, os, sys
from functools import partial
from pathlib import Path
sys.path.insert(0, {src!r})
sys.path.insert(0, {here!r})
from link import LinkHub
from wikidata import main
main.close_group = partial(main.close_group, api=LinkHub(Path("fakehub").resolve(), json.loads(os.environ["BENCH_LINK"])))
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
            if name == "split" else (RUN_LINK if "BENCH_LINK" in env else RUN).format(src=str(src), here=str(Path(__file__).parent)))
    cmd = [*(["nice", "-n", str(nice)] if nice else []), sys.executable, "-c", code]
    return subprocess.Popen(cmd, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)


def _env(name: str, workers: int, follow: bool) -> dict:
    # (a group a chunk, WIKIDATA_GROUP_SOURCE_GB at 1 MB: a release's groups are about 400
    # of its chunks and a set's slice two to four, so with a group a chunk the groups
    # close as the split runs, as at a release; in either mode, and ignored by a commit
    # before the fixed group rule, whose groups are then one a set)
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "WIKIDATA_WORKERS": str(workers), "WIKIDATA_GROUP_SOURCE_GB": "0.001"}
    # (and a release's defaults: the group files not kept, finalise's commits as the
    # commit's default orders them; no link unless a mode sets one, bench 17.2.0)
    for k in ("WIKIDATA_SCHOLAR", "WIKIDATA_FOLLOW_SPLIT", "WIKIDATA_PHASES", "KEEP", "BENCH_LINK",
              "WIKIDATA_KEEP_GROUP_FILES", "WIKIDATA_FINALISE_PENDING_UPLOAD_GB", "WIKIDATA_FINALISE_CLAIMS_LAST",
              "WIKIDATA_FINALISE_EARLY"):
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
    sample's chunk files, for the uploads' scale. Then (bench 17.2.0, link.py) the
    overlapped runs again with their pushes over the emulated link, and finalise's three
    runs on the overlapped runs' output, without the link and over it."""
    from link import LINK_RATES, finalise_stage, time_scale
    from link import _published as branch_bytes
    from release import RELEASE_SOURCE_BYTES

    release = os.environ["WIKIDATA_RELEASE"]
    # (what the commit's release runs: whether its runs follow the split, and finalise's
    # defaults, which the model takes: the group files kept, the commits' unsent bound, and
    # whether the main set's first finalise starts once its run ends, bench 17.4.0)
    ask = (f"import json, sys; sys.path.insert(0, {str(src)!r}); from wikidata import config, dump; "
           "print(json.dumps([hasattr(dump, 'SetManifest'), config.KEEP_GROUP_FILES is not False, "
           "getattr(config, 'FINALISE_PENDING_UPLOAD_BYTES', 0), getattr(config, 'FINALISE_EARLY', False)]))")
    follows, kept, pending, early = json.loads(subprocess.run([sys.executable, "-c", ask], capture_output=True, text=True,
                                                      env=_env("main", workers, False), check=True).stdout)
    half = max(1, workers // 2)
    modes = {"overlap_seq": [[("split", _env("split", workers, False), 0)],
                             [("scholar", _env("scholar", workers, False), 0)],
                             [("main", _env("main", workers, False), 0)]]}
    if follows:
        modes["overlap"] = [[("split", _env("split", workers, False), 0),
                             ("scholar", _env("scholar", half, True), 10), ("main", _env("main", half, True), 10)]]
    sample = {"main": 0, "scholar": 0}
    for f in files:
        sample["scholar" if Path(f).name.startswith("s") else "main"] += Path(f).stat().st_size
    scale = {k: RELEASE_SOURCE_BYTES[k] / n for k, n in sample.items() if n}
    # (the emulated link at the rate on record, in the overlapped runs: link.py)
    linked = {}
    if follows:
        rate = LINK_RATES[0]
        linked[f"overlap_link_{rate / 1e6:.0f}"] = rate
        link = {"up": rate, "down": rate, "scale": scale, "time_scale": time_scale("overlap", sum(sample.values()))}
    base = "overlap" if follows else "overlap_seq"  # (finalise's input: a release's, overlapped)
    res, by_set, stage = {}, {}, Path(tempfile.mkdtemp(dir=out.parent))
    try:
        with open(out.with_suffix(".log"), "w") as log:
            for p, steps in [*modes.items(), *((p, None) for p in linked)]:
                root = stage / p
                dump_dir = root / "releases" / release / "dump"
                dump_dir.mkdir(parents=True)
                os.link(dump_slice, dump_dir / f"wikidata-{release}-all.json.bz2")
                if steps is None:  # (the overlapped runs, over the link)
                    (root / "link").mkdir()
                    env = {"BENCH_LINK": json.dumps(link | {"dir": str(root / "link")})}
                    steps = [[(n, e | env if n != "split" else e, nice) for n, e, nice in modes["overlap"][0]]]
                    res[f"{p}_wall"], _ = _run(src, root, steps, log)
                    res[f"{p}_up_s"] = round(sum(json.loads(line)["s"] for line in
                                                 (root / "link" / "transfers.jsonl").read_text().splitlines()), 2)
                    shutil.rmtree(root)
                    continue
                res[f"{p}_wall"], res[f"{p}_peak_rss_mb"] = _run(src, root, steps, log)
                by_set[p] = _uploaded(root / "fakehub")
                res[f"{p}_upload_mb"] = round(sum(by_set[p].values()) / 1e6, 1)
                if p != base:
                    shutil.rmtree(root)
            # finalise's three runs on the overlapped runs' output: without the link, then
            # over it at each rate (link.py)
            groups = branch_bytes(stage / base / "fakehub", f"build-{release}")
            digests = set()
            for rate in (None, *LINK_RATES):
                name = f"finalise_link_{0 if rate is None else round(rate / 1e6)}"
                work = stage / name
                work.mkdir()
                f = finalise_stage(src, stage / base, work, rate, scale, sum(sample.values()), workers, log)
                shutil.rmtree(work)
                digests.add(f["digest"])
                res |= {f"{name}_wall": f["wall"], **{f"{name}_run{i + 1}_wall": w for i, w in enumerate(f["walls"])}}
                if rate is not None:
                    res |= {f"{name}_up_s": f["up_s"], f"{name}_down_s": f["down_s"]}
                    # (the share of the link's time not added to the wall: under the compute)
                    if f["up_s"] + f["down_s"]:
                        res[f"{name}_hidden"] = round(1 - (f["wall"] - res["finalise_link_0_wall"])
                                                      / (f["up_s"] + f["down_s"]), 3)
                else:
                    res["finalise_link_published_mb"] = round(sum(sum(t.values()) for t in f["published"].values()) / 1e6, 1)
                    # (what the release model takes, by set and table: bytes on the sample)
                    res["finalise_link_bytes"] = {"group": groups, "published": f["published"], "sample": sample,
                                                  "kept": kept, "pending": pending, "early": early}
            if len(digests) > 1:
                raise RuntimeError("finalise over the link: its runs published different files")
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    # Modelled, apart from the measured figures: a release's uploads at the rate on record,
    # each set's bytes scaled by its chunk files' (the release's over the sample's)
    release_bytes = sum(n * RELEASE_SOURCE_BYTES[k] / sample[k] for k, n in by_set["overlap_seq"].items() if sample[k])
    res["upload_eta_modelled"] = round(release_bytes / UPLOAD_RATE)
    out.write_text(json.dumps(res))


# The upload rates a release's wall is modelled at (bench 17.2.0), in bytes a second: the
# one on record and three faster, the Hub's own being unknown
RELEASE_LINK_RATES = (7e6, 30e6, 60e6, 120e6)


# A nice-10 process's share of a core a nice-0 process also wants: the kernel's CFS
# weights (sched_prio_to_weight: nice 10, 110; nice 0, 1024)
NICE_10_SHARE = 110 / (110 + 1024)


def _phase(run: dict, fixed: bool) -> SimpleNamespace:
    """The download, the split and both sets' runs (release_wall), and when each share of
    the runs' CPU time is done. `fixed`: bench 17.2.0's model; else 17.1.0's."""
    from release import RELEASE_DOWNLOAD_RATE, RELEASE_DUMP_BYTES

    cores = run["workers"]
    split, pipeline = run["split_eta"], run["pipeline_eta"]
    download = RELEASE_DUMP_BYTES / RELEASE_DOWNLOAD_RATE
    split_cores = min(cores, run["split_cpu_total"] / run["split_wall_total"])
    pipeline_cores = min(cores, run["cpu_total"] / run["wall"])
    pipeline_cpu = pipeline * pipeline_cores
    follows = run.get("overlap_wall") is not None
    if not fixed:  # (17.1.0)
        split_end = max(split, download)
        free = pipeline_cores
        if follows:
            idle = (cores - split_cores) * split + cores * (split_end - split)
            during = min(pipeline_cpu, idle, free * split_end)
    else:
        # The split's CPU time is its ETA's (its CPU time a byte over every core), and it
        # uses at most its cores on the sample, no faster than the download. With the runs
        # (nice 10) wanting more than the cores it leaves, it keeps all but their CFS share
        # of its own; the runs follow its parts, so they go no faster than it makes them.
        split_cpu = split * cores
        pace = pipeline_cpu / split_cpu  # (the runs' CPU time a second of the split's)
        x = min(split_cores, split_cpu / download)
        if follows:
            x = min(x, max(split_cores * (1 - NICE_10_SHARE), cores / (1 + pace)))
        split_end = split_cpu / x
        # (with nothing else running, the runs use every core: both sets' workers and
        # their groups' closes; 17.1.0 took the sample's pipeline's, about 1.9 of 4)
        free = cores if follows else pipeline_cores
        if follows:
            during = min(pipeline_cpu, min(pace * x, cores - x) * split_end)
    if not follows:
        during, start = 0.0, split_end
    else:
        start = 0.0
    compute_end = split_end + (pipeline_cpu - during) / free

    def closes(share: float) -> float:
        """When the pipeline has done `share` of its CPU time."""
        done = share * pipeline_cpu
        if done <= during:
            return start + (split_end - start) * done / during if during else split_end
        return split_end + (done - during) / free

    return SimpleNamespace(download=download, split_end=split_end, hidden=during / pipeline_cpu,
                           compute_end=compute_end, closes=closes, follows=follows)


def _uploads_end(ph: SimpleNamespace, seconds: float) -> float:
    """The build branch's uploads, `seconds` of the link in all, in a release's groups."""
    from release import RELEASE_GROUPS

    groups = sum(RELEASE_GROUPS.values())
    uploaded = 0.0
    for k in range(1, groups + 1):
        uploaded = max(uploaded, ph.closes(k / groups)) + seconds / groups
    return uploaded


def _main_uploads_end(ph: SimpleNamespace, seconds: float) -> float:
    """When the main set's pushes end, `_uploads_end`'s groups taken a set each in turn
    (RELEASE_GROUPS: 30 a set, the main set's first), a waiting main group pushed before
    a waiting scholarly one once the split has ended (push/turn.py, bench 17.4.0), in
    turn before. The link is busy as long as in `_uploads_end`, so every push ends with
    it."""
    from release import RELEASE_GROUPS

    groups = sum(RELEASE_GROUPS.values())
    ready = [(ph.closes(k / groups), k % 2 == 0) for k in range(1, groups + 1)]  # (when, scholarly)
    link, end, waiting, i = 0.0, 0.0, [], 0
    while i < len(ready) or waiting:
        if not waiting:
            link = max(link, ready[i][0])
        while i < len(ready) and ready[i][0] <= link:
            waiting.append(ready[i])
            i += 1
        waiting.sort(key=lambda g: (g[1] if link >= ph.split_end else 0, g[0]))
        _, scholarly = waiting.pop(0)
        link += seconds / groups
        if not scholarly:
            end = link
    return end


# finalise's tables in a release's order (main.finalise: claims first, the rest in
# config.Table's), claims_labels built after them
FINALISE_TABLES = ("claims", "labels", "descriptions", "aliases", "links", "entities")


def finalise_link(run: dict, rate: float, busy: float = 0.0) -> dict | None:
    """finalise's three runs at a release with their transfers (bench 17.2.0), modelled:
    its wall from its start, its group files' download and its published bytes, or None
    for a run without the emulated-link stage's bytes (link.py). Each run (main,
    scholarly, main again) a process after the one before, its tables in order:

    - **A table's download:** its group files from the build branch at
      link.DOWNLOAD_RATE, unless the commit keeps them by default (none at master:
      WIKIDATA_KEEP_GROUP_FILES unset keeps none), before its compaction.
    - **Its compute:** `finalise_eta`'s share: the claims' `finalise_claims_share` of it
      (both sets'), the rest the other tables', each set's and table's by its group files'
      bytes. The claims refs (`refs_eta`) once the main set's claims are done.
      claims_labels' build, link.CLAIMS_LABELS_BUILD, half in each set's last run.
    - **Its published upload** at `rate`, once it is written, as the commit orders it.
      At master (and with WIKIDATA_FINALISE_PENDING_UPLOAD_GB=0), commit_table and
      verify_table run after its write and before the next table, the CPU idle. A commit
      whose finalise has a background committer (config.FINALISE_PENDING_UPLOAD_BYTES
      above 0: one thread committing each table in order while the next is written, the
      next not started while more than that is written and not yet committed) has it
      modelled as such; the claims refs on the committer once the claims are committed;
      claims_labels' build once the set's claims and labels are; the run ends with its
      last commit.
    - **Its first run's commits behind `busy`** seconds of the link from its start (bench
      17.4.0): the scholarly set's last pushes, when the main set's first finalise starts
      before they end.

    Bytes: each set's and table's on the stage's sample (its group files on the build
    branch, and the published files finalise committed), scaled as the uploads are (the
    release's chunk files' bytes over the sample's)."""
    from link import CLAIMS_LABELS_BUILD, DOWNLOAD_RATE
    from release import RELEASE_SOURCE_BYTES

    lb = run.get("finalise_link_bytes")
    if not lb or not run.get("finalise_eta"):
        return None
    scale = {k: RELEASE_SOURCE_BYTES[k] / n for k, n in lb["sample"].items() if n}
    group = {k: {t: b * scale[k] for t, b in v.items()} for k, v in lb["group"].items() if k in scale}
    pub = {k: {t: b * scale[k] for t, b in v.items()} for k, v in lb["published"].items() if k in scale}
    fin, share = run["finalise_eta"], run.get("finalise_claims_share")
    claims = sum(g.get("claims", 0) for g in group.values())
    rest = sum(b for g in group.values() for t, b in g.items() if t != "claims")
    if share is None:  # (by bytes alone)
        share = claims / (claims + rest)

    def cpu(k: str, t: str) -> float:
        b = group[k].get(t, 0)
        return fin * (share * b / claims if t == "claims" else (1 - share) * b / rest)

    def one(k: str, tables: list[str], cl: bool, busy: float = 0.0) -> float:
        """One run's wall: the writer's clock and the committer's, in order; the link
        `busy` from its start."""
        t, free = 0.0, busy
        pending: list[tuple[float, float]] = []  # (committed at, bytes) of written tables
        done: dict[str, float] = {}
        for name in tables + (["claims_labels"] if cl else []):
            if name == "claims_labels":
                if lb["pending"]:  # (its build reads the set's sorted claims and labels)
                    t = max(t, *(done.get(n, 0.0) for n in ("claims", "labels")))
                t += CLAIMS_LABELS_BUILD / 2
            elif lb["pending"]:  # (no table started while more is unsent than the bound)
                while sum(b for c, b in pending if c > t) > lb["pending"]:
                    t = min(c for c, _ in pending if c > t)
            down = 0.0 if (lb["kept"] or name == "claims_labels") else group[k].get(name, 0) / DOWNLOAD_RATE
            t += down + (cpu(k, name) if name != "claims_labels" else 0.0)
            up = pub[k].get(name, 0) / rate
            refs = (run.get("refs_eta") or 0) if (k == "main" and name == "claims" and not cl) else 0
            if not lb["pending"]:
                t = max(t, busy) + up + refs
                done[name] = t
            else:
                free = max(free, t) + up
                done[name] = free
                free += refs
                pending.append((free, pub[k].get(name, 0)))
        return max(t, free)

    tables = {k: [t for t in FINALISE_TABLES if t in group[k]] for k in group}
    runs = [one("main", tables["main"], False, busy), one("scholar", tables["scholar"], True), one("main", [], True)]
    return {"wall": sum(runs), "download": 0.0 if lb["kept"] else sum(b for g in group.values() for b in g.values()) / DOWNLOAD_RATE,
            "group": sum(b for g in group.values() for b in g.values()),
            "published": sum(b for g in pub.values() for b in g.values())}


def release_wall(run: dict) -> dict:
    """A release's wall time as its recipes run it (bench 17.1.0), modelled from a run's
    stage ETAs and measured figures: the stages overlapped where the commit's release
    overlaps them, and the uploads, which the serial ETA (`eta`, `eta_refs`: the split,
    the pipeline, finalise and its claims refs, one after another, no transfer) leaves
    out. Every figure here is modelled, none measured; a commit without the figures
    (before bench 17.0.0, say) has none. In release seconds, from the release's start:

    - **The download** (`just download-split`): the dump's bz2 at RELEASE_DOWNLOAD_RATE,
      the split following it as it is written (#85). The split ends at the later of its
      ETA and the download's end (a split that waits for the dump ends with it).
    - **The split and the pipelines** (`just overlap-release`, for a commit whose runs
      follow the split: `overlap_wall` measured): the split at nice 0 keeps its ETA; both
      sets' runs at nice 10 take only the cores it leaves idle. The split's cores are its
      CPU time over its wall time on the sample (`split_cpu_total` / `split_wall_total`,
      of the tier's workers: the ETA's cores), the rest idle over its ETA, and every core
      while it waits for the download. The pipeline's CPU time at a release is its ETA
      times its cores on the sample (`cpu_total` / `wall`: its ETA is its wall time
      scaled by its CPU time, so at those cores). What it has left when the split ends
      runs on every core (bench 17.2.0: both sets' workers and closes; 17.1.0 ran it at
      the sample's pipeline's cores, about 1.9, so a split leaving fewer cores idle ended
      the compute later however little CPU time it took; its figures are kept, `_17_1`).
      A commit whose runs can't follow the split runs them after it (`just release`
      after `just download-split`): the pipeline's ETA after the split's end.
    - **The uploads:** the bytes `upload_eta_modelled` takes (a release's, at 7 MB/s, the
      one upload rate on record), in a release's groups (RELEASE_GROUPS: 30 a set) of
      equal bytes, each closing when the pipeline has done its share of the CPU time
      (the bytes going with it), uploaded one after another on one link at that rate: both
      sets' runs share it. The runs end once their last group is uploaded.
    - **Finalise** (`just release`, once `overlap-release` has ended: both sets' runs, so
      both sets' uploads, done, and the split): its runs (main, scholarly, main again for
      claims_labels) one after another, `finalise_eta` and its claims refs (`refs_eta`).
      finalise's own transfers (its downloads and commits) and claims_labels' build are
      in neither this nor the serial ETA.

    The critical path is the later of the compute's end (the split's or the pipeline's)
    and the uploads': `release_upload_backlog_modelled` is how much later the uploads end
    than the compute plus the last group's own upload (above zero, the link is the
    path, by that much; at or below, the compute). And, as a check on the CPU-share
    model, the split and the pipeline's ETAs summed times the measured at-once over
    one-after-another wall on the sample (`release_compute_ratio_modelled`; the
    sample's few parts, numbered late in a short split, overlap less than a release's
    thousands).

    **With finalise's transfers, at each upload rate** (bench 17.2.0,
    `release_wall_link_{7,30,60,120}_modelled`): the same, the build branch's uploads at
    that rate (their bytes `upload_eta_modelled`'s), then finalise as `finalise_link`
    models it, its group files' download and its published upload included, and
    claims_labels' build. For a commit whose `overlap-release` starts the main set's first
    finalise once the main set's run ends (config.FINALISE_EARLY, bench 17.4.0): the
    groups a set each in turn, a waiting main group pushed first (`_main_uploads_end`),
    finalise from the later of the compute's end and the main set's last push, its first
    run's commits after the scholarly set's last pushes (the link FIFO)."""
    need = ("split_eta", "pipeline_eta", "finalise_eta", "upload_eta_modelled", "split_cpu_total",
            "split_wall_total", "cpu_total", "wall", "workers", "overlap_seq_wall")
    if any(not run.get(k) for k in need):
        return {}
    uploads = run["upload_eta_modelled"]
    finalise = run["finalise_eta"] + (run.get("refs_eta") or 0)
    groups = sum(__import__("release").RELEASE_GROUPS.values())
    out = {}
    for fixed, suffix in ((False, "_17_1"), (True, "")):
        ph = _phase(run, fixed)
        uploaded = _uploads_end(ph, uploads)
        pipelines_end = max(ph.compute_end, uploaded)
        out |= {
            f"release_pipeline_hidden_modelled{suffix}": round(ph.hidden, 4),
            f"release_compute_end_modelled{suffix}": round(ph.compute_end),
            f"release_uploads_end_modelled{suffix}": round(uploaded),
            f"release_upload_backlog_modelled{suffix}": round(uploaded - ph.compute_end - uploads / groups),
            f"release_split_end_modelled{suffix}": round(ph.split_end),
            f"release_finalise_start_modelled{suffix}": round(max(ph.split_end, pipelines_end)),
            f"release_wall_modelled{suffix}": round(max(ph.split_end, pipelines_end) + finalise),
        }
    out["release_download_modelled"] = round(ph.download)
    if ph.follows:
        out["release_compute_ratio_modelled"] = round((run["split_eta"] + run["pipeline_eta"])
                                                      * run["overlap_wall"] / run["overlap_seq_wall"])
    # Finalise's transfers, at each upload rate (bench 17.2.0); the main set's first
    # finalise once its own pushes end, for a commit whose overlap-release runs it there
    # (config.FINALISE_EARLY, bench 17.4.0)
    early = (run.get("finalise_link_bytes") or {}).get("early")
    for rate in RELEASE_LINK_RATES:
        r = f"{rate / 1e6:.0f}"
        uploaded = _uploads_end(ph, uploads * 7e6 / rate)
        if early:
            start = max(ph.split_end, ph.compute_end, _main_uploads_end(ph, uploads * 7e6 / rate))
            f = finalise_link(run, rate, busy=max(0.0, uploaded - start))
        else:
            start = max(ph.split_end, ph.compute_end, uploaded)
            f = finalise_link(run, rate)
        if f is None:
            break
        out |= {f"release_uploads_end_link_{r}_modelled": round(uploaded),
                f"release_finalise_start_link_{r}_modelled": round(start),
                f"release_finalise_link_{r}_modelled": round(f["wall"]),
                f"release_wall_link_{r}_modelled": round(start + f["wall"])}
        out |= {"release_group_gb_modelled": round(f["group"] / 1e9, 1),
                "release_published_gb_modelled": round(f["published"] / 1e9, 1),
                "release_finalise_download_modelled": round(f["download"])}
    return out
