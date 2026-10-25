#!/usr/bin/env python3
"""Test the pipeline following the split (`just overlap-release`): the split and both
sets' runs (main.run with WIKIDATA_FOLLOW_SPLIT=1) at once, each a process of its own,
on a dump slice made of the chunk files given (bench/dumpslice.py), cut into chunks of
2,000 entities, with groups of a few chunks (WIKIDATA_GROUP_SOURCE_GB) and no Hub (a
stand-in HfApi keeps each repo's build branch on disk, test_group_restart.FakeHub).

- **Overlapped, uninterrupted:** each set's stand-in Hub gets the files, byte for byte,
  that a split followed by each set's run (the pipeline as before) uploads, in the same
  groups, and finalise's local stages (compaction, the sort) on them publish the same
  files, byte for byte.
- **Killed and run again:** each process killed at a kill point is run again until it
  completes, the others left running. Whether the split was killed partway through its
  parts, a run partway through a group, or both at once, the outcome is the same as
  above, byte for byte, and every chunk is complete and every group done.

A kill is os._exit at a hook, the work cut off where it is (as by SIGKILL).

Usage: python scripts/test_overlap_restart.py CHUNK.parquet ... [kill point ...]
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent.parent
RELEASE = "20261012"
sys.path.insert(0, str(ROOT / "bench"))
sys.path.insert(0, str(ROOT / "scripts"))

# Where a process is killed: {process: (hook, n)}, each hook counted in its process
KILLS = {
    "split: after numbering its first two chunks": {"split": ("numbered", 2)},
    "split: mid part, its claims file written": {"split": ("claims", 3)},
    "main run: mid group, after `merged`": {"main": ("stage:merged", 1)},
    "scholar run: mid upload": {"scholar": ("upload_folder", 2)},
    "both: the split mid part and the main run mid group": {"split": ("claims", 4), "main": ("stage:merged", 1)},
}

SPLIT = """
import os, sys
sys.path.insert(0, {src!r})
from wikidata import dump
dump.CHUNK_ENTITIES = 2000
hook, n = os.environ.get("KILL_HOOK"), int(os.environ.get("KILL_N") or 0)
calls = [0]
def tick():
    calls[0] += 1
    if calls[0] == n:
        os._exit(137)
if hook == "numbered":
    advance = dump._Numbering.advance
    def advance_then_die(self, done):
        advance(self, done)
        if self.next >= n:
            os._exit(137)
    dump._Numbering.advance = advance_then_die
if hook == "claims":
    from wikidata import claims_build
    write = claims_build.write_claims
    claims_build.write_claims = lambda *a: (write(*a), tick())[0]
dump.split()
"""

RUN = """
import os, sys
from functools import partial
from pathlib import Path
sys.path.insert(0, {src!r})
sys.path.insert(0, {scripts!r})
from test_group_restart import FakeHub
from wikidata import main
import wikidata.push.core as core
hook, n = os.environ.get("KILL_HOOK"), int(os.environ.get("KILL_N") or 0)
calls = [0]
def tick():
    calls[0] += 1
    if calls[0] == n:
        os._exit(137)
ticks = {{}}
if hook == "upload_folder":
    ticks[hook] = tick
elif hook and hook.startswith("stage:"):
    record = core.record_stage
    core.record_stage = lambda d, g, s: (record(d, g, s), tick() if s == hook[6:] else None)
hub = FakeHub(Path("fakehub").resolve(), ticks)
main.close_group = partial(main.close_group, api=hub)
main.run()
"""

FINALISE = """
import json, os, shutil, sys
from pathlib import Path
sys.path.insert(0, {src!r})
sys.path.insert(0, {scripts!r})
from test_group_restart import FakeHub, digest
from wikidata import compact, sort_by_id
from wikidata.config import COMPACT_DIR, HF_USER, HUB_COPY_DIR, HUB_REVISION, REPO_TARGET, STATE_DIR, WORK_DIR, Table
from wikidata.push.groups import _read
from wikidata.state import get_all_state
hub = FakeHub(Path("fakehub").resolve())
branch = {{t: hub._dir(REPO_TARGET.format(hf_user=HF_USER, tbl=t), HUB_REVISION) for t in Table}}
uploaded = {{f"{{t}}/{{p.relative_to(d)}}": digest(p) for t, d in branch.items() for p in sorted(d.rglob("*.parquet"))}}
state = STATE_DIR.resolve()
done = [t for t in Table if t != Table.CLAIMS_LABELS]  # (a release builds claims_labels in finalise)
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
published = {{str(p.relative_to(HUB_COPY_DIR)): digest(p) for p in sorted(HUB_COPY_DIR.rglob("*.parquet"))
             if ".cache" not in p.parts}}
left = sorted(str(p) for d in ("results", "staging") for p in (WORK_DIR / d).rglob("*") if p.is_file())
Path(sys.argv[1]).write_text(json.dumps({{
    "uploaded": uploaded, "published": published, "left": left,
    "steps": sorted(set(get_all_state(state, "chunk_*")["step"].to_list())),
    "groups": [(r["first"], r["last"], r["stage"]) for r in _read(state / "groups.jsonl") if r["stage"] == "done"],
}}))
"""

# Groups of a few chunks: a chunk of 2,000 entities' source is about 0.5-1.5 MB
GROUP_SOURCE_GB = "0.0008"


def env(name: str, kill: tuple[str, int] | None, follow: bool) -> dict:
    e = {**os.environ, "WIKIDATA_RELEASE": RELEASE, "HF_HUB_OFFLINE": "1", "WIKIDATA_WORKERS": "1",
         "WIKIDATA_GROUP_SOURCE_GB": GROUP_SOURCE_GB, "WIKIDATA_FINALISE_WORKERS": "2"}
    for k in ("WIKIDATA_SCHOLAR", "WIKIDATA_FOLLOW_SPLIT", "KILL_HOOK", "KILL_N"):
        e.pop(k, None)
    if name == "scholar":
        e["WIKIDATA_SCHOLAR"] = "1"
    if follow:
        e["WIKIDATA_FOLLOW_SPLIT"] = "1"
    if kill:
        e["KILL_HOOK"], e["KILL_N"] = kill[0], str(kill[1])
    return e


def start(name: str, root: Path, kill=None, follow=True) -> subprocess.Popen:
    code = (SPLIT if name == "split" else RUN).format(src=str(ROOT / "src"), scripts=str(ROOT / "scripts"))
    log = (root / f"{name}.log").open("a")
    return subprocess.Popen([sys.executable, "-c", code], cwd=root, env=env(name, kill, follow),
                            stdout=log, stderr=subprocess.STDOUT)


def wait(procs: dict[str, subprocess.Popen], root: Path, timeout: float = 900) -> dict[str, int]:
    """Each process's exit code once all have exited, or as soon as one fails (the others,
    still running, left out)."""
    t, codes = time.monotonic(), {}
    while len(codes) < len(procs):
        for name, p in procs.items():
            if name not in codes and p.poll() is not None:
                codes[name] = p.returncode
        if any(codes.values()):
            return codes
        if time.monotonic() - t > timeout:
            for p in procs.values():
                p.kill()
            logs = "\n".join(f"--- {n}\n{(root / f'{n}.log').read_text()[-2000:]}" for n in procs)
            raise SystemExit(f"FAIL: timed out\n{logs}")
        time.sleep(0.2)
    return codes


def outcome(root: Path) -> dict:
    out = {}
    for name in ("main", "scholar"):
        code = FINALISE.format(src=str(ROOT / "src"), scripts=str(ROOT / "scripts"))
        # (the cards' figures in the scenario's own dir, not the checkout's docs/releases)
        e = env(name, None, False) | {"WIKIDATA_CARD_FIGURES_DIR": str(root / "card-figures" / name)}
        p = subprocess.run([sys.executable, "-c", code, str(root / f"{name}.json")], cwd=root,
                           env=e, capture_output=True, text=True)
        if p.returncode:
            raise SystemExit(f"FAIL: {name}'s finalise failed:\n{p.stdout[-2000:]}{p.stderr[-2000:]}")
        out[name] = json.loads((root / f"{name}.json").read_text())
    return out


def prepare(root: Path, dump_slice: Path) -> None:
    d = root / "releases" / RELEASE / "dump"
    d.mkdir(parents=True)
    os.link(dump_slice, d / f"wikidata-{RELEASE}-all.json.bz2")


def sequential(tmp: Path, dump_slice: Path) -> dict:
    root = tmp / "sequential"
    prepare(root, dump_slice)
    if wait({"split": start("split", root)}, root)["split"]:
        raise SystemExit(f"FAIL: the split failed\n{(root / 'split.log').read_text()[-3000:]}")
    codes = wait({n: start(n, root, follow=False) for n in ("main", "scholar")}, root)
    if any(codes.values()):
        raise SystemExit(f"FAIL: a run failed: {codes}")
    return outcome(root)


def overlapped(tmp: Path, dump_slice: Path, kill: str | None) -> dict:
    root = tmp / (f"kill-{list(KILLS).index(kill)}" if kill else "overlapped")
    prepare(root, dump_slice)
    hooks = KILLS.get(kill, {})
    procs = {n: start(n, root, hooks.get(n)) for n in ("split", "main", "scholar")}
    for _ in range(8):  # each killed process run again (the others left running) until all complete
        codes = wait(procs, root)
        failed = {n: c for n, c in codes.items() if c}
        if not failed:
            return outcome(root)
        if not hooks or any(c != 137 for c in failed.values()):
            logs = "\n".join(f"--- {n}\n{(root / f'{n}.log').read_text()[-2000:]}" for n in failed)
            raise SystemExit(f"FAIL {kill}: {failed}\n{logs}")
        procs |= {n: start(n, root) for n in failed}
    raise SystemExit(f"FAIL {kill}: not complete after eight runs")


def main() -> None:
    from dumpslice import make

    files = [a for a in sys.argv[1:] if a.endswith(".parquet")]
    points = [a for a in sys.argv[1:] if not a.endswith(".parquet")] or list(KILLS)
    with TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        dump_slice = tmp / "slice.json.bz2"
        make(dump_slice, files)
        ref = sequential(tmp, dump_slice)
        ok = all(r["steps"] == [6] and not r["left"] and r["uploaded"] and r["published"] and len(r["groups"]) > 1
                 for r in ref.values())
        print(f"{'ok  ' if ok else 'FAIL'} split, then each set's run: "
              + ", ".join(f"{n} {len(r['uploaded'])} files uploaded in {len(r['groups'])} groups, "
                          f"{len(r['published'])} published" for n, r in ref.items()), flush=True)
        failed = not ok
        for kill in [None, *points]:
            got = overlapped(tmp, dump_slice, kill)
            same = all(got[n][k] == ref[n][k] for n in ref for k in ("uploaded", "published", "groups"))
            clean = all(r["steps"] == [6] and not r["left"] for r in got.values())
            label = f"killed at {kill}, run again" if kill else "overlapped, uninterrupted"
            print(f"{'ok  ' if same and clean else 'FAIL'} {label}: the same files uploaded, in the same groups, "
                  f"and published, byte for byte" + ("" if clean else f" (not clean: {got})"), flush=True)
            failed |= not (same and clean)
    if failed:
        sys.exit(1)
    print("all passed")


if __name__ == "__main__":
    main()
