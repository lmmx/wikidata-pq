#!/usr/bin/env python3
"""Test that a run killed at any step of a chunk's partition or a group's close, then run
again, uploads the same files, byte for byte, and publishes the same sorted files, as a
run never interrupted (no Hub: a stand-in HfApi keeps its "build branch" on disk).

Each run is a process of its own, as `process-wikidata` is: main.run's loop (resume an
unfinished group, then pool.process_chunks over the chunks not yet partitioned, groups
closed in the background by push.close_group), on synthetic chunks of every table. A
chunk's "process" writes its processed tables, seeded by its index; its partition is
main.partition_chunk. Chunks 3-5 wait for the first group to be closed, so the groups are
always chunks 0-2 and 3-5. A kill is os._exit at a hook (the work is cut off where it is,
nothing flushed or cleaned up, as by SIGKILL), in the run's process or a chunk worker's.
The run is repeated until it completes. Then:

- the stand-in Hub's files (the build branch) have the uninterrupted run's digests;
- every chunk is complete, every group done, and no processed, partition or staged file
  is left;
- finalise's local stages (compaction, the sort) on those files give the uninterrupted
  run's published files, byte for byte.

The kill points marked "merged ahead" hold the first group at a step of its close (its
upload or its verify) until the second group has merged ahead of it (pool.MERGE_AHEAD),
so the kill finds both at once; the second group's upload checks the first is done (one
push at a time, in order). A run whose second group did not merge ahead within a minute
fails, as does one with no kill that is not as the uninterrupted run.

Usage: python scripts/test_group_restart.py [kill point ...]   (default: every point)
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

CHUNKS = 6
GROUPS = [(0, 2), (3, 5)]
LANGS = ["en", "de", "fr", "nl", "zh", "zh-hans", "mul", "ja", "sv"]

# Where a run is killed: (process, hook, when[, the first group's step held until the
# second has merged ahead]). Hooks are counted in their process; a hook of None is a run
# not killed
KILLS = {
    "partition: mid chunk, between its tables' sidecars": ("worker:1", "write_stats_sidecar", 2),
    "partition: all tables done, before the chunk's state": ("worker:4", "update_state:PARTITION", 1),
    "merge: mid table, after a sink": ("parent", "sink", 2),
    "merge: a table staged, before its marker": ("parent", "marker", 2),
    "merge: after a marker, mid deletion of processed files": ("parent", "unlink_processed", 2),
    "merge: in the claims glue's thread": ("parent", "glue", 1),
    "close: after `merged` recorded": ("parent", "stage:merged", 1),
    "push: mid upload, a table uploaded": ("parent", "upload_folder", 2),
    "close: after `pushed` recorded": ("parent", "stage:pushed", 1),
    "verify: the second group's": ("parent", "get_paths_info", 8),
    "close: after `verified` recorded": ("parent", "stage:verified", 1),
    "close: mid clean-up": ("parent", "unlink_staged", 2),
    "close: after `done` recorded, chunks still running": ("parent", "stage:done", 1),
    "push: the last group's upload": ("parent", "upload_folder", 9),
    "merged ahead: uninterrupted, the second merged as the first uploads": ("parent", None, 0, "upload_folder"),
    "merged ahead: the second's merge, mid table, the first `merged`": ("parent", "sink:next", 2, "upload_folder"),
    "merged ahead: the second's merge, mid table, the first `pushed` not verified": ("parent", "sink:next", 2, "get_paths_info"),
    "merged ahead: the second's claims glue, the first `pushed`": ("parent", "glue:next", 1, "get_paths_info"),
    "merged ahead: the first's upload, mid table, the second merged": ("parent", "upload_folder", 2, "upload_folder"),
    "merged ahead: the first's verify, the second merged": ("parent", "get_paths_info", 2, "upload_folder"),
    "merged ahead: the first's clean-up, the second merged": ("parent", "unlink_staged", 2, "get_paths_info"),
    "merged ahead: the second's `closed`, recorded at its turn": ("parent", "stage:closed", 2, "upload_folder"),
}
FIRST, NEXT = (f"upload_{g[0]}" for g in GROUPS)  # (the pool's upload threads of the groups)


def die_at(hook: str, n: int):
    """A function that kills this process on its n-th call."""
    calls = [0]

    def tick(*_):
        calls[0] += 1
        if calls[0] == n:
            print(f"[test] killed at {hook} ({n})", flush=True)
            os._exit(137)

    return tick


class FakeHub:
    """The HfApi calls close_group makes, against a directory: a repo's branch's files at
    root/{repo}/{revision}/{path}."""

    def __init__(self, root: Path, tick=None):
        self.root, self.tick = root, tick or {}

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
        if t := self.tick.get("hold:upload_folder"):
            t()
        folder = Path(folder_path)
        for p in sorted(folder.glob(allow_patterns)):
            dst = self._dir(repo_id, revision) / p.relative_to(folder)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, dst.with_suffix(".part"))
            dst.with_suffix(".part").replace(dst)
        if t := self.tick.get("upload_folder"):
            t()

    def get_paths_info(self, repo_id, paths, revision=None, **kw):
        if t := self.tick.get("hold:get_paths_info"):
            t()
        if t := self.tick.get("get_paths_info"):
            t()
        d = self._dir(repo_id, revision)
        return [
            SimpleNamespace(path=p, size=(d / p).stat().st_size,
                            lfs=SimpleNamespace(sha256=hashlib.sha256((d / p).read_bytes()).hexdigest()))
            for p in paths if (d / p).exists()
        ]


def tables(c: int):
    """Chunk c's processed tables, in each table's processed layout."""
    import polars as pl

    from wikidata.config import Table

    n = 40 + 7 * c
    ids = [f"Q{(r * 7919 + c * 104729) % 100000}" for r in range(n)]  # out of id order
    langs = lambda r, k: [LANGS[(r + c + j) % len(LANGS)] for j in range(k)]
    return {
        Table.LABEL: pl.DataFrame({"id": ids, "labels": [[{"key": l, "value": f"{i} {l}"} for l in langs(r, 1 + r % 5)] for r, i in enumerate(ids)]}),
        Table.DESC: pl.DataFrame({"id": ids, "descriptions": [[{"key": l, "value": f"about {i}"} for l in langs(r, r % 3)] for r, i in enumerate(ids)]}),
        Table.ALIAS: pl.DataFrame({"id": ids, "aliases": [[{"key": l, "value": [f"{i}a", f"{i}b"][: 1 + r % 2]} for l in langs(r, r % 2)] for r, i in enumerate(ids)]}),
        Table.LINKS: pl.DataFrame({"id": ids, "sitelinks": [[{"key": f"{l}wiki", "value": {"site": f"{l}wiki", "title": i}} for l in langs(r, r % 4)] for r, i in enumerate(ids)]}),
        Table.CLAIMS: pl.DataFrame({"id": [i for i in ids for _ in range(3)], "property": [f"P{k}" for _ in ids for k in range(3)],
                                    "value": [f"{i}/{k}" for i in ids for k in range(3)]}),
        Table.CLAIMS_LABELS: pl.DataFrame([("labels", f"Q{r % 9}", l, f"Q{r % 9} {l}") for r in range(30) for l in langs(r, 2)],
                                          schema=["field", "ref", "language", "label"], orient="row"),
        Table.ENTITIES: pl.DataFrame({"id": ids, "type": ["item"] * n, "lastrevid": list(range(n))}),
    }


def work(chunk: int, results: Path, state: Path, kill: str | None) -> None:
    """A chunk's process (its processed tables written, then its state) and partition."""
    from wikidata import main
    from wikidata.state import Step, update_state

    if kill:
        where, hook, n, *_ = KILLS[kill]
        if where == f"worker:{chunk}":
            tick = die_at(hook, n)
            if hook == "write_stats_sidecar":
                f = main.write_stats_sidecar
                main.write_stats_sidecar = lambda *a, **k: (f(*a, **k), tick())
            elif hook == "update_state:PARTITION":
                f = main.update_state
                main.update_state = lambda p, step, d: (tick() if step == Step.PARTITION else None, f(p, step, d))
    if chunk >= GROUPS[1][0]:  # the second group's chunks wait for the first to close
        ledger = state / "groups.jsonl"
        while not (ledger.exists() and '"first": 0, "last": 2, "stage": "closed"' in ledger.read_text()):
            time.sleep(0.05)
    for table, df in tables(chunk).items():
        dst = results / table / f"chunk_{chunk}.parquet"
        dst.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(dst)
    update_state(Path(f"chunk_{chunk}.parquet"), Step.PROCESS, state)
    main.partition_chunk(chunk, state, results)


def run(kill: str | None) -> None:
    """main.run's loop, the Hub a FakeHub."""
    import pathlib

    import polars as pl

    from wikidata import main
    from wikidata.config import OUTPUT_DIR, STATE_DIR, Table
    from wikidata.pool import process_chunks
    from wikidata.push import Group, close_group, open_chunks, record_partitioned, unfinished_group
    import wikidata.push.core as core
    from wikidata.state import Step, get_all_state, init_files, last_chunk

    ticks = {}
    state = STATE_DIR.resolve()
    if kill and len(KILLS[kill]) > 3:
        ticks[f"hold:{KILLS[kill][3]}"] = hold_first(state)
    if kill and KILLS[kill][0] == "parent" and KILLS[kill][1]:
        _, hook, n, *_ = KILLS[kill]
        tick = die_at(hook, n)
        if hook.endswith(":next"):  # (counted in the second group's close only: its thread, or its files)
            def tick(*a, tick=tick):
                if threading.current_thread().name == NEXT or f"chunks-{GROUPS[1][0]:05d}-" in str(a):
                    tick()
            hook = hook[:-5]
        if hook in ("upload_folder", "get_paths_info"):
            ticks[hook] = tick
        elif hook == "sink":
            f = core._sink_batch
            core._sink_batch = lambda *a: (f(*a), tick())
        elif hook == "glue":
            f = core.pq_glue.concat_many
            core.pq_glue = SimpleNamespace(concat_many=lambda *a: (r := f(*a), tick(*a), r)[2])
        elif hook == "marker":
            f = pathlib.Path.touch
            pathlib.Path.touch = lambda self, *a, **k: (tick() if ".merged" in self.parts else None, f(self, *a, **k))
        elif hook == "unlink_processed":
            f = pathlib.Path.unlink
            pathlib.Path.unlink = lambda self, *a, **k: (f(self, *a, **k), tick() if self.parent.parent.name == "results" else None)
        elif hook == "unlink_staged":
            f = pathlib.Path.unlink
            # the close's clean-up of its staged files, staging/{table}/{key}/{group}.parquet
            pathlib.Path.unlink = lambda self, *a, **k: (f(self, *a, **k), tick() if self.parent.parent.parent.name == "staging" else None)
        elif hook.startswith("stage:"):
            f = core.record_stage
            core.record_stage = lambda d, g, s: (f(d, g, s), tick() if s == hook[6:] else None)
    hub = FakeHub(Path("fakehub").resolve(), ticks)
    results = OUTPUT_DIR.resolve()
    if not state.exists():
        init_files([Path(f"chunk_{c}.parquet") for c in range(CHUNKS)], state)
    repos = {t: f"test/wikidata-{t}" for t in Table}

    def close(chunks: list[int]) -> None:
        close_group(Group(chunks[0], chunks[-1], last_chunk(state)), target_repos=repos, state_dir=state, api=hub)

    if pending := unfinished_group(state):
        group, stage = pending
        close_group(group, target_repos=repos, state_dir=state, stage=stage, api=hub)
    todo = get_all_state(state).filter(pl.col("step") < Step.PARTITION)["chunk"].unique().sort().to_list()
    process_chunks(
        todo, open_chunks(state), workers=2, chunks_per_worker=4, start=lambda c: None,
        work=partial(work, results=results, state=state, kill=kill),
        done=lambda c: record_partitioned(state, c, 1000 + c), ready=lambda g: len(g) >= 3, close=close,
    )
    main._remove_empty_dirs(results)


def hold_first(state: Path):
    """A hook that holds the first group's close (at its first call in that close's
    thread) until the second group has merged ahead of it, and checks that the second
    group's close (at each call in its thread) comes once the first is done; either
    failing exits the run (3, 4)."""
    import wikidata.push.core as core

    merged = threading.Event()
    merge_group = core.merge_group

    def merge(group, *a, **k):
        merge_group(group, *a, **k)
        if group.first == GROUPS[1][0]:
            merged.set()

    core.merge_group = merge

    def hold():
        name = threading.current_thread().name
        if name == FIRST and not merged.is_set():
            print("[test] the first group held until the second merges ahead", flush=True)
            if not merged.wait(60):
                print("[test] FAIL: the second group did not merge ahead", flush=True)
                os._exit(3)
        elif name == NEXT and f'"first": {GROUPS[0][0]}, "last": {GROUPS[0][1]}, "stage": "done"' not in (state / "groups.jsonl").read_text():
            print("[test] FAIL: the second group uploads before the first is done", flush=True)
            os._exit(4)

    return hold


def finalise(out: Path) -> None:
    """finalise's local stages (as bench/pipeline.py runs them) on the stand-in Hub's
    build branch, and the digests of the published files and of the build branch."""
    from wikidata import compact, sort_by_id
    from wikidata.config import COMPACT_DIR, HUB_COPY_DIR, HUB_REVISION, STATE_DIR, Table

    hub = FakeHub(Path("fakehub").resolve())
    branch = {t: hub._dir(f"test/wikidata-{t}", HUB_REVISION) for t in Table}
    uploaded = {f"{t}/{p.relative_to(d)}": digest(p) for t, d in branch.items() for p in sorted(d.rglob("*.parquet"))}
    state = STATE_DIR.resolve()
    done = [t for t in Table if t != Table.CLAIMS_LABELS]  # a release builds claims_labels in finalise
    for t in done:
        for p in branch[t].rglob("*.parquet"):
            dst = COMPACT_DIR / "src" / t / p.relative_to(branch[t])
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, dst)
    for t in sorted(done, key=lambda t: t != Table.CLAIMS):
        compact.record_stage(state, t, "downloaded")
        compact.compact_table(t, "unused", state)
    for t in sorted(done, key=lambda t: t != Table.CLAIMS):
        (HUB_COPY_DIR / t).mkdir(parents=True, exist_ok=True)
        sort_by_id.check_compacted_copy(t)
        sort_by_id.record_stage(state, t, "sourced")
        sort_by_id.remove_compact_src(t)
        sort_by_id.write_table(t)
        sort_by_id.record_stage(state, t, "written")
        sort_by_id.write_metadata(t)
        sort_by_id.replace_local_copy(t)
        sort_by_id.record_stage(state, t, "done")
    published = {str(p.relative_to(HUB_COPY_DIR)): digest(p) for p in sorted(HUB_COPY_DIR.rglob("*.parquet"))
                 if ".cache" not in p.parts}
    from wikidata.state import get_all_state

    from wikidata.push.groups import _read

    left = sorted(str(p) for d in ("releases/20261012/results", "releases/20261012/staging")
                  for p in Path(d).rglob("*") if p.is_file())
    out.write_text(json.dumps({
        "uploaded": uploaded, "published": published, "left": left,
        "steps": sorted(set(get_all_state(state, "chunk_*")["step"].to_list())),
        "groups": [(r["first"], r["last"], r["stage"]) for r in _read(state / "groups.jsonl") if r["stage"] == "done"],
    }))


def digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def child(*args: str, root: Path) -> int:
    env = {**os.environ, "WIKIDATA_RELEASE": "20261012", "HF_HUB_OFFLINE": "1", "WIKIDATA_FINALISE_WORKERS": "2"}
    log = root / "log.txt"
    with log.open("a") as f:
        return subprocess.run([sys.executable, __file__, *args], cwd=root, env=env, stdout=f, stderr=subprocess.STDOUT).returncode


def scenario(kill: str | None, tmp: Path) -> dict:
    root = tmp / (str(abs(hash(kill))) if kill else "reference")
    root.mkdir()
    code = child("run", kill or "", root=root)
    log = (root / "log.txt").read_text()
    if "[test] FAIL" in log:
        raise SystemExit(f"FAIL {kill}: {log[log.index('[test] FAIL'):].splitlines()[0]}")
    if kill and KILLS[kill][1] and code == 0:
        raise SystemExit(f"FAIL {kill}: the run was not killed")
    if kill and not KILLS[kill][1] and code != 0:
        raise SystemExit(f"FAIL {kill}: the run did not complete:\n{log[-3000:]}")
    for _ in range(3):
        if child("run", "", root=root) == 0:
            break
    else:
        raise SystemExit(f"FAIL {kill}: the rerun did not complete:\n{(root / 'log.txt').read_text()[-3000:]}")
    if child("finalise", root=root):
        raise SystemExit(f"FAIL {kill}: finalise failed:\n{(root / 'log.txt').read_text()[-3000:]}")
    return json.loads((root / "out.json").read_text())


def main() -> None:
    points = sys.argv[1:] or list(KILLS)
    with TemporaryDirectory() as tmp:
        ref = scenario(None, Path(tmp))
        ok = (ref["steps"] == [6] and ref["groups"] == [[0, 2, "done"], [3, 5, "done"]] and not ref["left"]
              and len(ref["uploaded"]) > 50 and ref["published"])
        print(f"{'ok  ' if ok else 'FAIL'} the uninterrupted run: {len(ref['uploaded'])} files uploaded in 2 groups, "
              f"{len(ref['published'])} published, nothing left", flush=True)
        failed = not ok
        for kill in points:
            got = scenario(kill, Path(tmp))
            same = got["uploaded"] == ref["uploaded"] and got["published"] == ref["published"]
            clean = got["steps"] == [6] and got["groups"] == ref["groups"] and not got["left"]
            print(f"{'ok  ' if same and clean else 'FAIL'} killed at {kill}: rerun to completion, "
                  f"the same {len(got['uploaded'])} uploaded and {len(got['published'])} published files"
                  + ("" if clean else f" (left: {got['left'][:3]}, steps {got['steps']}, groups {got['groups']})"), flush=True)
            failed |= not (same and clean)
    if failed:
        sys.exit(1)
    print("all passed")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        run(sys.argv[2] or None)
    elif len(sys.argv) > 1 and sys.argv[1] == "finalise":
        finalise(Path("out.json").resolve())
    else:
        main()
