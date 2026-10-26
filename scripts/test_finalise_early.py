#!/usr/bin/env python3
"""Test the main set's first finalise started before the scholarly set's uploads end
(WIKIDATA_FINALISE_EARLY, on by default; docs/journal/2026-10-26-finalise-before-uploads-end.md),
as `just overlap-release` then `just release` run it, on a dump slice made of the chunk
files given (bench/dumpslice.py), cut into chunks of 2,000 entities, with groups of a few
chunks and no Hub: bench/link.py's stand-in (each repo's branches on disk, serving
close_group's calls and finalise's), its uploads slept for their bytes on one link.

- **The reference:** WIKIDATA_FINALISE_EARLY=0 in every process: the split and both sets'
  runs at once, then finalise main, scholar, main (as before).
- **Early, uninterrupted:** the split and both runs at once; once the main run ends, the
  main set's first finalise, the scholarly run still going; then finalise main, scholar,
  main. The build branches end byte for byte as the reference's (every repo's files,
  published and not), both sets finalised, and:
  - **the turn:** no scholarly push starts while the main run holds its turn
    (push/turn.py), and at least one waits for it;
  - **the overlap:** the early finalise starts before the scholarly run ends.
- **The recipe:** `just overlap-release` itself, its commands stand-ins on PATH that log
  when they run: the early finalise runs once the main run has ended and before the
  scholarly run has, as the main set's (WIKIDATA_SCHOLAR empty), not following the split,
  at nice 10; with WIKIDATA_FINALISE_EARLY=0 it doesn't run; with the main run failing it
  doesn't run, the scholarly run is still waited for, and the recipe fails.
- **Killed and run again:** a process killed at a kill point is run again until it
  completes, the others left running, and the order kept: the early finalise once the
  main run has ended, the last finalises once every process has. The same outcome, byte
  for byte. A kill is os._exit at a hook, the work cut off where it is (as by SIGKILL).

Usage: python scripts/test_finalise_early.py CHUNK.parquet ... [kill point ...]
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent.parent
RELEASE = "20261013"
sys.path.insert(0, str(ROOT / "bench"))

# Where a process is killed: {process: (hook, n)}, each hook counted in its process
KILLS = {
    "main run: holding its turn, before a push": {"main": ("push", 2)},
    "scholar run: after waiting for its turn, before its push": {"scholar": ("waited", 1)},
    "early finalise: claims written, the scholarly run still pushing": {"early": ("sort:claims:written", 1)},
    "early finalise: its first commit landed, the next not": {"early": ("commit", 1)},
    "both: the scholarly run mid group and the early finalise mid compaction":
        {"scholar": ("stage:merged", 3), "early": ("compact:labels:downloaded", 1)},
}

SPLIT = """
import sys
sys.path.insert(0, {src!r})
from wikidata import dump
dump.CHUNK_ENTITIES = 2000
dump.split()
"""

COMMON = """
import json, os, sys, time
from pathlib import Path
sys.path.insert(0, {src!r})
sys.path.insert(0, {bench!r})
from link import install
hub = install(Path("fakehub").resolve())
hook, n = os.environ.get("KILL_HOOK"), int(os.environ.get("KILL_N") or 0)
calls = [0]
def tick():
    calls[0] += 1
    if calls[0] == n:
        os._exit(137)
def event(**kw):
    with open("events.jsonl", "a") as f:
        f.write(json.dumps({{"set": "scholar" if os.environ.get("WIKIDATA_SCHOLAR") else "main", **kw}}) + "\\n")
"""

RUN = COMMON + """
from wikidata import main
import wikidata.push.core as core
push, wait, hold = core.push_group, core.wait_turn, main.hold_turn
def push_group(*a, **kw):
    if hook == "push":
        tick()
    t = time.time()
    push(*a, **kw)
    event(what="push", start=t, end=time.time())
def wait_turn(name):
    w = wait(name)
    if w > 0:
        event(what="wait", waited=w)
    if hook == "waited" and w > 0:
        tick()
    return w
state = [False]
def hold_turn(on):
    hold(on)
    if on != state[0]:
        state[0] = on
        event(what="hold", on=on, t=time.time(), pid=os.getpid())
core.push_group, core.wait_turn, main.hold_turn = push_group, wait_turn, hold_turn
if hook and hook.startswith("stage:"):
    record = core.record_stage
    core.record_stage = lambda d, g, s: (record(d, g, s), tick() if s == hook[6:] else None)
main.run()
"""

FINALISE = COMMON + """
from wikidata import compact, main, sort_by_id
# (the cards' figures sample entities the slice doesn't have: as bench/link.py's stage)
main.update_stats, main.write_cards = (lambda: None), (lambda: {{}})
if hook and hook.split(":")[0] in ("sort", "compact"):
    mod = sort_by_id if hook.startswith("sort") else compact
    _, table, stage = hook.split(":")
    record = mod.record_stage
    mod.record_stage = lambda d, t, s: (record(d, t, s), tick() if (str(t), s) == (table, stage) else None)
if hook == "commit":
    commit = hub.create_commit
    def create_commit(*a, **kw):
        out = commit(*a, **kw)
        if not kw.get("run_as_future"):
            tick()
        return out
    hub.create_commit = create_commit
event(what="finalise", start=time.time())
main.finalise()
"""

GROUP_SOURCE_GB = "0.0008"  # (groups of a few chunks: a chunk of 2,000 entities is ~0.5-1.5 MB)
UP_RATE = 3e6  # (the stand-in link's upload rate, sample bytes a second)
# (the scholarly set's pushes slowed, so its last ones run on well after the main run ends,
# as at a release on a slow link, and the early finalise's kill points fall under them)
SCALE = {"main": 1.0, "scholar": 5.0}


def env(root: Path, name: str, early: bool, kill: tuple[str, int] | None) -> dict:
    e = {**os.environ, "WIKIDATA_RELEASE": RELEASE, "HF_HUB_OFFLINE": "1", "WIKIDATA_WORKERS": "1",
         "WIKIDATA_GROUP_SOURCE_GB": GROUP_SOURCE_GB, "WIKIDATA_FINALISE_WORKERS": "2",
         "WIKIDATA_FINALISE_EARLY": "1" if early else "0",
         "BENCH_LINK": json.dumps({"up": UP_RATE, "down": 1e12, "scale": SCALE,
                                   "time_scale": 1.0, "dir": str(root / "link")})}
    for k in ("WIKIDATA_SCHOLAR", "WIKIDATA_FOLLOW_SPLIT", "KILL_HOOK", "KILL_N", "WIKIDATA_KEEP_GROUP_FILES",
              "WIKIDATA_FINALISE_PENDING_UPLOAD_GB", "WIKIDATA_FINALISE_CLAIMS_LAST"):
        e.pop(k, None)
    set_ = "scholar" if "scholar" in name else "main"
    if set_ == "scholar":
        e["WIKIDATA_SCHOLAR"] = "1"
    if name in ("main", "scholar"):  # (the runs follow the split)
        e["WIKIDATA_FOLLOW_SPLIT"] = "1"
    e["WIKIDATA_CARD_FIGURES_DIR"] = str(root / "card-figures" / set_)
    if kill:
        e["KILL_HOOK"], e["KILL_N"] = kill[0], str(kill[1])
    return e


def start(root: Path, name: str, early: bool, kill=None) -> subprocess.Popen:
    """`name`: split, main, scholar (the runs), early (the main set's first finalise, in
    overlap-release), or finalise-main / finalise-scholar (`release`'s)."""
    code = {"split": SPLIT, "main": RUN, "scholar": RUN}.get(name, FINALISE)
    code = code.format(src=str(ROOT / "src"), bench=str(ROOT / "bench"))
    log = (root / f"{name}.log").open("a")
    return subprocess.Popen([sys.executable, "-c", code], cwd=root, env=env(root, name, early, kill),
                            stdout=log, stderr=subprocess.STDOUT)


def release(tmp: Path, dump_slice: Path, early: bool, kill: str | None) -> dict:
    """overlap-release then release, as the recipes run them (each process killed at its
    kill point run again), and the outcome."""
    root = tmp / (f"kill-{list(KILLS).index(kill)}" if kill else ("early" if early else "reference"))
    d = root / "releases" / RELEASE / "dump"
    d.mkdir(parents=True)
    os.link(dump_slice, d / f"wikidata-{RELEASE}-all.json.bz2")
    (root / "link").mkdir()
    hooks = KILLS.get(kill, {})
    procs = {n: start(root, n, early, hooks.get(n)) for n in ("split", "main", "scholar")}
    ended: dict[str, float] = {}
    killed: dict[str, float] = {}
    t0 = time.monotonic()
    # overlap-release: the early finalise once the main run has ended (its exit code 0)
    while len(ended) < len(procs) or (early and "early" not in procs):
        if time.monotonic() - t0 > 1200:
            for p in procs.values():
                p.kill()
            raise SystemExit(f"FAIL {kill}: timed out (a process waiting for ever?)\n" + "\n".join(
                f"--- {n}\n{(root / f'{n}.log').read_text()[-1500:]}" for n in procs))
        for n, p in list(procs.items()):
            if n in ended or p.poll() is None:
                continue
            if p.returncode == 137 and n in hooks:
                killed[n] = time.time()
                procs[n] = start(root, n, early)  # (run again, its kill point spent)
                hooks = {k: v for k, v in hooks.items() if k != n}
                continue
            if p.returncode:
                raise SystemExit(f"FAIL {kill}: {n} failed ({p.returncode})\n{(root / f'{n}.log').read_text()[-3000:]}")
            ended[n] = time.time()
            if n == "main" and early:
                procs["early"] = start(root, "early", early, hooks.get("early"))
        time.sleep(0.1)
    # release: its runs find their sets done; finalise main, scholar, main
    for n in ("finalise-main", "finalise-scholar", "finalise-main"):
        for _ in range(3):
            p = start(root, n, early)
            if p.wait() == 0:
                break
        else:
            raise SystemExit(f"FAIL {kill}: {n} failed\n{(root / f'{n}.log').read_text()[-3000:]}")
    from link import _digest

    events = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
    done = all((root / "releases" / s / "state" / "finalise.done").exists() for s in (RELEASE, f"{RELEASE}-scholar"))
    return {"digest": _digest(root / "fakehub", f"build-{RELEASE}"), "done": done, "events": events,
            "ended": ended, "files": sum(1 for _ in (root / "fakehub").rglob("*.parquet")),
            "unreached": sorted(hooks), "killed": killed}


STUB = """#!/usr/bin/env bash
echo "$(basename "$0") start $(date +%s.%N) scholar=${{WIKIDATA_SCHOLAR:-}} follow=${{WIKIDATA_FOLLOW_SPLIT:-}} \
workers=${{WIKIDATA_WORKERS:-}} nice=$(nice) args=$*" >> {log}
case "$(basename "$0")" in
  process-wikidata) sleep $([ -n "${{WIKIDATA_SCHOLAR:-}}" ] && echo 3 || echo 1)
                    [ -z "${{WIKIDATA_SCHOLAR:-}}" ] && [ -n "${{MAIN_FAILS:-}}" ] && exit 1 ;;
  finalise-wikidata) sleep 1 ;;
esac
echo "$(basename "$0") end $(date +%s.%N) scholar=${{WIKIDATA_SCHOLAR:-}}" >> {log}
"""


def recipe(tmp: Path) -> bool:
    """`just overlap-release` with its commands (process-wikidata, finalise-wikidata, and
    download-split's download-dump and split-dump) stand-ins that log their start and end."""
    import shutil

    real_just = shutil.which("just")
    if real_just is None:
        print("skip the recipe: no `just`", flush=True)
        return True
    ok = True
    for case, extra in (("default", {}), ("WIKIDATA_FINALISE_EARLY=0", {"WIKIDATA_FINALISE_EARLY": "0"}),
                        ("the main run failing", {"MAIN_FAILS": "1"})):
        d = tmp / f"recipe-{len(extra)}{''.join(extra)}"
        (d / "bin").mkdir(parents=True)
        log = d / "calls.log"
        shutil.copyfile(ROOT / "Justfile", d / "Justfile")  # (`just download-split` reads it from here)
        for name in ("process-wikidata", "finalise-wikidata", "download-dump", "split-dump"):
            (d / "bin" / name).write_text(STUB.format(log=log))
            (d / "bin" / name).chmod(0o755)
        e = {k: v for k, v in os.environ.items() if not k.startswith("WIKIDATA_")}
        e |= {"PATH": f"{d / 'bin'}:{e['PATH']}", **extra}
        code = subprocess.run([real_just, "overlap-release", RELEASE], cwd=d, env=e, capture_output=True,
                              text=True).returncode
        calls = [line.split() for line in log.read_text().splitlines()]

        def when(name, what, scholar):
            return [float(c[2]) for c in calls if c[0] == name and c[1] == what and c[3] == f"scholar={scholar}"]

        fin = [c for c in calls if c[0] == "finalise-wikidata" and c[1] == "start"]
        main_end, scholar_end = when("process-wikidata", "end", ""), when("process-wikidata", "end", "1")
        if case == "default":
            good = (code == 0 and len(fin) == 1 and main_end and scholar_end
                    and main_end[0] <= float(fin[0][2]) < scholar_end[0]
                    and fin[0][3:7] == ["scholar=", "follow=", "workers=", "nice=10"])
        elif case == "WIKIDATA_FINALISE_EARLY=0":
            good = code == 0 and not fin and scholar_end
        else:
            good = code != 0 and not fin and scholar_end  # (the scholarly run waited for)
        print(f"{'ok  ' if good else 'FAIL'} the recipe, {case}: " + ("; ".join(" ".join(c[:2] + c[3:7]) for c in calls)
                                                                     if not good else "the early finalise as expected"),
              flush=True)
        ok &= bool(good)
    return ok


def turn_violations(events: list[dict]) -> tuple[int, int]:
    """Scholarly pushes started while the main run held its turn, of all scholarly pushes."""
    spans, on = [], {}
    for e in sorted((e for e in events if e["what"] == "hold" and e["set"] == "main"), key=lambda e: e["t"]):
        if e["on"]:
            on[e["pid"]] = e["t"]
        elif e["pid"] in on:
            spans.append((on.pop(e["pid"]), e["t"]))
    pushes = [e for e in events if e["what"] == "push" and e["set"] == "scholar"]
    bad = [p for p in pushes if any(a + 0.05 < p["start"] < b - 0.05 for a, b in spans)]
    return len(bad), len(pushes)


def held_at(events: list[dict], t: float) -> bool:
    """Whether a main run held its turn at time t (its last turn event before t on)."""
    last: dict[int, bool] = {}
    for e in sorted((e for e in events if e["what"] == "hold" and e["set"] == "main" and e["t"] < t),
                    key=lambda e: e["t"]):
        last[e["pid"]] = e["on"]
    return any(last.values())


def main_pushes_overlapped(events: list[dict]) -> int:
    """Scholarly pushes started during a main push (the contention the turn removes)."""
    mains = [(e["start"], e["end"]) for e in events if e["what"] == "push" and e["set"] == "main"]
    return sum(any(a < e["start"] < b for a, b in mains) for e in events if e["what"] == "push" and e["set"] == "scholar")


def main() -> None:
    from dumpslice import make

    files = [a for a in sys.argv[1:] if a.endswith(".parquet")]
    points = [a for a in sys.argv[1:] if not a.endswith(".parquet")] or list(KILLS)
    failed = False
    with TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        failed |= not recipe(tmp)
        dump_slice = tmp / "slice.json.bz2"
        make(dump_slice, files)
        ref = release(tmp, dump_slice, False, None)
        ok = ref["done"] and ref["files"] > 0
        print(f"{'ok  ' if ok else 'FAIL'} reference (WIKIDATA_FINALISE_EARLY=0): {ref['files']} files on the "
              f"build branches, both sets finalised; {main_pushes_overlapped(ref['events'])} scholarly pushes "
              "started during a main push", flush=True)
        failed |= not ok
        for kill in [None, *points]:
            got = release(tmp, dump_slice, True, kill)
            same = got["digest"] == ref["digest"] and got["done"] and not got["unreached"]
            label = f"killed at {kill}, run again" if kill else "early, uninterrupted"
            extra = f" (not killed: {got['unreached']}'s kill point not reached)" if got["unreached"] else ""
            if kill is not None and not got["unreached"] and "main" in got["killed"]:
                held = held_at(got["events"], got["killed"]["main"])
                extra += f"; the main run {'held' if held else 'did not hold'} its turn when killed"
                same &= held
            if kill is not None and not got["unreached"]:
                ends = got["ended"]["scholar"]
                extra += "; " + ", ".join(f"{n} killed {'while' if t < ends else 'after'} the scholarly run "
                                          f"{'was going' if t < ends else 'ended'} ({ends - t:+.1f} s)"
                                          for n, t in got["killed"].items())
            if kill is None:
                bad, n = turn_violations(got["events"])
                waits = [e["waited"] for e in got["events"] if e["what"] == "wait"]
                fin = next(e["start"] for e in got["events"] if e["what"] == "finalise")
                lead = got["ended"]["scholar"] - fin
                extra = (f"; {bad} of {n} scholarly pushes started while the main run held its turn, "
                         f"{len(waits)} waited for it ({sum(waits):.1f} s); the early finalise started "
                         f"{lead:.1f} s before the scholarly run ended")
                same &= bad == 0 and len(waits) > 0 and lead > 0
            print(f"{'ok  ' if same else 'FAIL'} {label}: the same build branches, byte for byte, both sets "
                  f"finalised{extra}", flush=True)
            failed |= not same
    if failed:
        sys.exit(1)
    print("all passed")


if __name__ == "__main__":
    main()
