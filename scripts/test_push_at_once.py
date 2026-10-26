#!/usr/bin/env python3
"""Test a group's tables pushed at once (push.core.push_group, PUSH_TABLES_AT_ONCE: every
table by default; WIKIDATA_PUSH_TABLES_AT_ONCE=1 one after another), killed and run again,
as scripts/test_group_restart.py does for the serial push (no Hub: a stand-in HfApi keeps
its "build branch" on disk, each upload slow enough that a group's tables are in flight
together).

Each run is a process of its own, main.run's loop (test_group_restart.run's, on its
synthetic chunks: groups of chunks 0-2 and 3-5). Checked:

- **The same files:** pushed one table after another and at once, the stand-in Hub's
  files (the build branch) and finalise's published files are the same, byte for byte.
- **At once:** in each group, a table's upload starts before another's has ended, every
  table's upload call made once per run, and the group's `pushed` recorded only once every
  table is up.
- **Killed and run again** (os._exit, as by SIGKILL, in the run's process): mid-way through
  a table's files while the group's other tables are uploading; once a table's upload has
  ended while others are still going; after every table is up, before `pushed` is
  recorded. A table's upload failing (an exception in its thread) stops the run with an
  error once the group's other tables have finished, the group not recorded `pushed`.
  Each run again to completion: the same files, byte for byte, every group done and
  nothing left.

Usage: python scripts/test_push_at_once.py [kill point ...]   (default: every point)
"""

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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_group_restart as tgr  # noqa: E402

# Where a run is killed (or fails): (hook, n), counted across the run's threads
KILLS = {
    "mid-way through a table's files, the others uploading": ("copy", 9),
    "a table's upload ended, the group's others still going": ("upload_end", 2),
    "the second group's tables in flight": ("copy", 40),
    "every table up, before `pushed`": ("stage:pushed", 1),
    "a table's upload failing in its thread": ("fail:labels", 1),
}
# Each file's copy slept, so a group's tables overlap
COPY_SLEEP = 0.02


class FakeHub(tgr.FakeHub):
    """test_group_restart's stand-in, its uploads' files copied one by one, slowly, each
    upload's start and end logged (`events.jsonl`), and a kill or failure at a hook."""

    def __init__(self, root: Path, kill: str | None):
        super().__init__(root)
        self.kill, self.lock, self.calls = kill, threading.Lock(), {}

    def _tick(self, hook: str) -> None:
        if not self.kill or KILLS[self.kill][0] != hook:
            return
        with self.lock:
            self.calls[hook] = self.calls.get(hook, 0) + 1
            if self.calls[hook] == KILLS[self.kill][1]:
                print(f"[test] killed at {hook} ({self.calls[hook]})", flush=True)
                os._exit(137)

    def _log(self, event: str, repo_id: str, group: str) -> None:
        with self.lock, open(self.root / "events.jsonl", "a") as f:
            f.write(json.dumps({"t": time.monotonic(), "event": event, "repo": repo_id, "group": group}) + "\n")

    def upload_folder(self, repo_id, folder_path, allow_patterns, revision=None, **kw):
        group = allow_patterns.split("/")[-1].removesuffix(".parquet")
        self._log("start", repo_id, group)
        if self.kill and KILLS[self.kill][0] == f"fail:{repo_id.split('wikidata-')[1]}":
            time.sleep(0.2)  # (the others under way)
            raise RuntimeError("the stand-in Hub refused the upload")
        folder = Path(folder_path)
        for p in sorted(folder.glob(allow_patterns)):
            dst = self._dir(repo_id, revision) / p.relative_to(folder)
            dst.parent.mkdir(parents=True, exist_ok=True)
            time.sleep(COPY_SLEEP)
            shutil.copyfile(p, dst.with_suffix(".part"))
            self._tick("copy")
            dst.with_suffix(".part").replace(dst)
        self._log("end", repo_id, group)
        self._tick("upload_end")


def run(kill: str | None) -> None:
    """test_group_restart.run's loop, the Hub this module's FakeHub."""
    import polars as pl

    from wikidata import main
    from wikidata.config import OUTPUT_DIR, STATE_DIR, Table
    from wikidata.pool import process_chunks
    from wikidata.push import Group, close_group, open_chunks, record_partitioned, unfinished_group
    import wikidata.push.core as core
    from wikidata.state import Step, get_all_state, init_files, last_chunk

    hub = FakeHub(Path("fakehub").resolve(), kill)
    if kill and KILLS[kill][0].startswith("stage:"):
        f, stage = core.record_stage, KILLS[kill][0][6:]
        core.record_stage = lambda d, g, s: (hub._tick(KILLS[kill][0]) if s == stage else None, f(d, g, s))
    state, results = STATE_DIR.resolve(), OUTPUT_DIR.resolve()
    if not state.exists():
        init_files([Path(f"chunk_{c}.parquet") for c in range(tgr.CHUNKS)], state)
    repos = {t: f"test/wikidata-{t}" for t in Table}

    def close(chunks: list[int]) -> None:
        close_group(Group(chunks[0], chunks[-1], last_chunk(state)), target_repos=repos, state_dir=state, api=hub)

    if pending := unfinished_group(state):
        group, stage = pending
        close_group(group, target_repos=repos, state_dir=state, stage=stage, api=hub)
    todo = get_all_state(state).filter(pl.col("step") < Step.PARTITION)["chunk"].unique().sort().to_list()
    process_chunks(
        todo, open_chunks(state), workers=2, chunks_per_worker=4, start=lambda c: None,
        work=partial(tgr.work, results=results, state=state, kill=None),
        done=lambda c: record_partitioned(state, c, 1000 + c), ready=lambda g: len(g) >= 3, close=close,
    )
    main._remove_empty_dirs(results)


def child(*args: str, root: Path, env: dict) -> int:
    env = {**os.environ, "WIKIDATA_RELEASE": "20261012", "HF_HUB_OFFLINE": "1", "WIKIDATA_FINALISE_WORKERS": "2",
           **env}
    with (root / "log.txt").open("a") as f:
        return subprocess.run([sys.executable, __file__, *args], cwd=root, env=env, stdout=f,
                              stderr=subprocess.STDOUT).returncode


def events(root: Path) -> list[dict]:
    path = root / "fakehub" / "events.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def at_once(ev: list[dict]) -> bool:
    """In each group, some table's upload started before another's ended."""
    spans: dict[str, dict[str, list[float]]] = {}
    for e in ev:
        spans.setdefault(e["group"], {}).setdefault(e["repo"], []).append(e["t"])
    return bool(spans) and all(
        any(a[0] < b[-1] and b[0] < a[-1] for r, a in g.items() for s, b in g.items() if r < s)
        for g in spans.values())


def stages(root: Path) -> list[tuple]:
    path = root / "releases" / "20261012" / "state" / "groups.jsonl"
    return [(r["first"], r["last"], r["stage"]) for r in map(json.loads, path.read_text().splitlines())]


def scenario(name: str, kill: str | None, env: dict, tmp: Path) -> tuple[dict, Path]:
    root = tmp / name
    root.mkdir()
    code = child("run", kill or "", root=root, env=env)
    if kill:
        if code == 0:
            raise SystemExit(f"FAIL {name}: the run was not stopped")
        ev = events(root)
        group = ev[-1]["group"]  # (the group in flight when stopped)
        pushed = [s for s in stages(root) if s[2] == "pushed"]
        if KILLS[kill][0] != "stage:pushed" and len(pushed) >= 1 + (group != ev[0]["group"]):
            raise SystemExit(f"FAIL {name}: `pushed` recorded though a table was not up ({pushed})")
        if KILLS[kill][0].startswith("fail:"):
            log = (root / "log.txt").read_text()
            started = [e for e in ev if e["event"] == "start" and e["group"] == group]
            ended = [e for e in ev if e["event"] == "end" and e["group"] == group]
            if "the stand-in Hub refused the upload" not in log or len(ended) != len(started) - 1:
                raise SystemExit(f"FAIL {name}: not an error once the group's other tables ended "
                                 f"({len(ended)} of {len(started)} ended)")
        for _ in range(3):
            if child("run", "", root=root, env=env) == 0:
                break
        else:
            raise SystemExit(f"FAIL {name}: the rerun did not complete:\n{(root / 'log.txt').read_text()[-3000:]}")
    elif code:
        raise SystemExit(f"FAIL {name}: the run failed:\n{(root / 'log.txt').read_text()[-3000:]}")
    if child("finalise", root=root, env=env):
        raise SystemExit(f"FAIL {name}: finalise failed:\n{(root / 'log.txt').read_text()[-3000:]}")
    return json.loads((root / "out.json").read_text()), root


def main() -> None:
    points = sys.argv[1:] or list(KILLS)
    failed = False

    def check(name: str, ok: bool) -> None:
        nonlocal failed
        print(f"{'ok  ' if ok else 'FAIL'} {name}", flush=True)
        failed |= not ok

    with TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        serial, sroot = scenario("serial", None, {"WIKIDATA_PUSH_TABLES_AT_ONCE": "1"}, tmp)
        check(f"one table after another: {len(serial['uploaded'])} files uploaded in 2 groups, "
              f"{len(serial['published'])} published, nothing left",
              serial["steps"] == [6] and serial["groups"] == [[0, 2, "done"], [3, 5, "done"]]
              and not serial["left"] and len(serial["uploaded"]) > 50 and serial["published"])
        check("one table after another: no two uploads at once", not at_once(events(sroot)))
        ref, root = scenario("at once", None, {}, tmp)
        ev = events(root)
        check("at once: the same uploaded and published files as one after another",
              ref["uploaded"] == serial["uploaded"] and ref["published"] == serial["published"])
        check("at once: in each group, a table's upload started before another's ended", at_once(ev))
        starts = [(e["group"], e["repo"]) for e in ev if e["event"] == "start"]
        want = sorted((e["group"], e["repo"]) for e in events(sroot) if e["event"] == "start")
        check(f"at once: each table's upload called once a group ({len(starts)} calls, as one after another)",
              sorted(starts) == want and len(set(starts)) == len(starts))
        st = stages(root)
        check("at once: every group done, nothing left", ref["groups"] == serial["groups"] and not ref["left"])
        check("at once: `pushed` once per group, after `merged`",
              [s for s in st if s[2] in ("merged", "pushed")] == [(0, 2, "merged"), (0, 2, "pushed"),
                                                                 (3, 5, "merged"), (3, 5, "pushed")])
        for kill in points:
            got, _ = scenario(kill, kill, {}, tmp)
            same = got["uploaded"] == serial["uploaded"] and got["published"] == serial["published"]
            clean = got["steps"] == [6] and got["groups"] == serial["groups"] and not got["left"]
            check(f"{kill}: rerun to completion, the same {len(got['uploaded'])} uploaded and "
                  f"{len(got['published'])} published files, nothing left", same and clean)
    if failed:
        sys.exit(1)
    print("all passed")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        run(sys.argv[2] or None)
    elif len(sys.argv) > 1 and sys.argv[1] == "finalise":
        tgr.finalise(Path("out.json").resolve())
    else:
        main()
