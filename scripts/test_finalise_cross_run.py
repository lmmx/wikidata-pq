#!/usr/bin/env python3
"""Test a release's finalise runs overlapped across their processes
(on by default, WIKIDATA_FINALISE_CROSS_RUN=0 the serial runs; src/wikidata/cross_run.py,
docs/journal/2026-10-26-cross-run-overlap.md), on a dump slice made of the chunk files
given (bench/dumpslice.py), cut into chunks of 2,000 entities, with groups of a few chunks
and no Hub: bench/link.py's stand-in (each repo's branches on disk), its uploads slept for
their bytes on one link, so a run's last commits are still going when the next run starts.

The split and both sets' runs run once (as `just overlap-release` runs them, without the
early finalise). Each case below finalises a copy of that output:

- **The reference:** finalise main, scholar, main, one process after another
  (WIKIDATA_FINALISE_CROSS_RUN=0: the serial runs).
- **Cross-run, uninterrupted:** cross_run.run_release, as `finalise-release-runs` runs
  it (the flag on in each run). The build branches end byte for byte as the reference's
  (every repo's files, published and not), both sets finalised, the same local copies
  (the sorted files, by sha256) and the same cards' metadata, and:
  - **the overlap:** each next run starts before the run before has ended, and once it
    has written its last table (its `finalise.written`), so one run writes at a time;
  - **claims_labels** is built for both sets, each once the other set's labels are done;
  - **the main set's second run** starts its work only once its first has ended (their
    lock).
- **The gate at 0** (WIKIDATA_FINALISE_CROSS_RUN_GB=0): the scholarly run's first table
  is started only once the main run has nothing written and not yet committed. The same
  outcome.
- **Killed and run again:** a run killed (os._exit at a hook, as by SIGKILL) at a kill
  point; the orchestrator goes on as it does (a run killed before its mark stops the next
  from starting; one killed after it leaves the others going), and then runs again, as
  `just release` would be, until it ends with 0. The same outcome, byte for byte.
- **The recipes:** `just overlap-release` and `just release` with stand-ins for their
  commands on PATH: by default (the flag unset) and with it =1, the first runs `finalise-release-runs --after FILE`
  once the main run has ended, at nice 10, and writes the scholarly run's exit code to
  FILE once it ends (0, or 1 and the recipe failing); the second runs
  `finalise-release-runs` in place of its three finalise-wikidata calls. With
  WIKIDATA_FINALISE_CROSS_RUN=0 both are as before.

Usage: python scripts/test_finalise_cross_run.py CHUNK.parquet ... [kill point ...]
"""

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent.parent
RELEASE = "20261014"
sys.path.insert(0, str(ROOT / "bench"))
sys.path.insert(0, str(ROOT / "scripts"))

from test_finalise_early import RUN, SPLIT  # noqa: E402

# Where a run is killed: {set: (hook, n)}, each hook counted in its process (the first
# process of that set the hook is reached in; a run after a kill has none)
KILLS = {
    "main's first run, before its mark: labels written": {"main": ("sort:labels:written", 1)},
    "main's first run, after its mark: a commit, the scholarly run writing": {"main": ("commit_after_mark", 1)},
    "the scholarly run, claims written, main's commits still going": {"scholar": ("sort:claims:written", 1)},
    "the scholarly run, its claims_labels written, before its mark": {"scholar": ("sort:claims_labels:written", 1)},
    "main's second run, its claims_labels written, the scholarly commits going":
        {"main": ("sort:claims_labels:written", 1)},
    "both: main after its mark mid-commit, the scholarly run mid-compaction":
        {"main": ("commit_after_mark", 2), "scholar": ("compact:labels:downloaded", 1)},
}

COMMON = """
import json, os, sys, time
from pathlib import Path
sys.path.insert(0, {src!r})
sys.path.insert(0, {bench!r})
from link import install
hub = install(Path("fakehub").resolve())
set_ = "scholar" if os.environ.get("WIKIDATA_SCHOLAR") else "main"
hook, n = os.environ.get("KILL_HOOK"), int(os.environ.get("KILL_N") or 0)
spent = Path("kills") / set_
if hook and spent.exists():  # (a kill point is spent once reached)
    hook = None
calls = [0]
def tick():
    calls[0] += 1
    if calls[0] == n:
        spent.parent.mkdir(exist_ok=True)
        spent.touch()
        os._exit(137)
def event(**kw):
    with open("events.jsonl", "a") as f:
        f.write(json.dumps({{"set": set_, "pid": os.getpid(), "t": time.time(), **kw}}) + "\\n")
"""

FINALISE = COMMON + """
from wikidata import compact, cross_run, main, sort_by_id
from wikidata.config import STATE_DIR
# (the cards' figures sample entities the slice doesn't have: as bench/link.py's stage)
main.update_stats, main.write_cards = (lambda: None), (lambda: {{}})
if hook and hook.split(":")[0] in ("sort", "compact"):
    mod = sort_by_id if hook.startswith("sort") else compact
    _, table, stage = hook.split(":")
    record = mod.record_stage
    mod.record_stage = lambda d, t, s: (record(d, t, s), tick() if (str(t), s) == (table, stage) else None)
commit = hub.create_commit
def create_commit(*a, **kw):
    if kw.get("run_as_future"):
        return commit(*a, **kw)
    t = time.time()
    out = commit(*a, **kw)
    event(what="commit", start=t, end=time.time())
    if hook == "commit_after_mark" and (STATE_DIR / cross_run.WRITTEN).exists():
        tick()
    return out
hub.create_commit = create_commit
compact_table = main.compact_table
def compact_table_(tbl, *a, **kw):
    event(what="compact", table=str(tbl))
    return compact_table(tbl, *a, **kw)
main.compact_table = compact_table_
mark = cross_run.mark_written
cross_run.mark_written = lambda d: (mark(d), event(what="written"))
build = main.build_claims_labels
def build_claims_labels(*a, **kw):
    event(what="claims_labels")
    return build(*a, **kw)
main.build_claims_labels = build_claims_labels
hold = cross_run.hold
cross_run.hold = lambda d: (hold(d), event(what="locked"))
event(what="start")
main.finalise()
event(what="end")
"""

# The orchestrator, as `finalise-release-runs` runs it, each run FINALISE
ORCH = """
import json, os, subprocess, sys
from pathlib import Path
sys.path.insert(0, {src!r})
from wikidata import cross_run
envs, code = json.loads(Path("envs.json").read_text()), Path("finalise.py").read_text()
def start(i, set_, scholar):
    log = open(f"finalise-{{i + 1}}.log", "a")
    return subprocess.Popen([sys.executable, "-c", code], env=envs[set_] | {{"WIKIDATA_FINALISE_CROSS_RUN": "1"}},
                            stdout=log, stderr=subprocess.STDOUT)
sys.exit(cross_run.run_release(start, poll=0.1))
"""

GROUP_SOURCE_GB = "0.0008"  # (groups of a few chunks: a chunk of 2,000 entities is ~0.5-1.5 MB)
UP_RATE = 0.4e6  # (the stand-in link's upload rate in finalise, sample bytes a second)


def env(root: Path, set_: str, extra: dict | None = None) -> dict:
    e = {**os.environ, "WIKIDATA_RELEASE": RELEASE, "HF_HUB_OFFLINE": "1", "WIKIDATA_WORKERS": "1",
         "WIKIDATA_GROUP_SOURCE_GB": GROUP_SOURCE_GB, "WIKIDATA_FINALISE_WORKERS": "2",
         "WIKIDATA_FINALISE_EARLY": "0", "WIKIDATA_FINALISE_CROSS_RUN": "0"}  # (the serial reference's)
    for k in ("WIKIDATA_SCHOLAR", "WIKIDATA_FOLLOW_SPLIT", "KILL_HOOK", "KILL_N", "WIKIDATA_KEEP_GROUP_FILES",
              "WIKIDATA_FINALISE_PENDING_UPLOAD_GB", "WIKIDATA_FINALISE_CLAIMS_LAST",
              "WIKIDATA_FINALISE_CROSS_RUN_GB", "BENCH_LINK"):
        e.pop(k, None)
    if set_ == "scholar":
        e["WIKIDATA_SCHOLAR"] = "1"
    e["WIKIDATA_CARD_FIGURES_DIR"] = str(root / "card-figures" / set_)
    return e | (extra or {})


def overlap(root: Path, dump_slice: Path) -> None:
    """The split and both sets' runs at once, no link (the finalises' input)."""
    d = root / "releases" / RELEASE / "dump"
    d.mkdir(parents=True)
    os.link(dump_slice, d / f"wikidata-{RELEASE}-all.json.bz2")
    procs = []
    for name in ("split", "main", "scholar"):
        e = env(root, "scholar" if name == "scholar" else "main")
        if name != "split":
            e["WIKIDATA_FOLLOW_SPLIT"] = "1"
        code = (SPLIT if name == "split" else RUN).format(src=str(ROOT / "src"), bench=str(ROOT / "bench"))
        procs.append(subprocess.Popen([sys.executable, "-c", code], cwd=root, env=e,
                                      stdout=(root / f"{name}.log").open("a"), stderr=subprocess.STDOUT))
    for (name, p) in zip(("split", "main", "scholar"), procs):
        if p.wait():
            raise SystemExit(f"FAIL the overlap's {name} failed\n{(root / f'{name}.log').read_text()[-3000:]}")


def link_env(root: Path) -> dict:
    return {"BENCH_LINK": json.dumps({"up": UP_RATE, "down": 1e12, "scale": {"main": 1.0, "scholar": 1.0},
                                      "time_scale": 1.0, "dir": str(root / "link")})}


def prepare(base: Path, root: Path) -> None:
    from link import _copy_tree

    _copy_tree(base, root)
    (root / "link").mkdir(exist_ok=True)
    for f in ("events.jsonl",):
        (root / f).unlink(missing_ok=True)


def serial(root: Path) -> None:
    for i, set_ in enumerate(("main", "scholar", "main")):
        code = FINALISE.format(src=str(ROOT / "src"), bench=str(ROOT / "bench"))
        p = subprocess.run([sys.executable, "-c", code], cwd=root, env=env(root, set_, link_env(root)),
                           stdout=(root / f"serial-{i + 1}.log").open("a"), stderr=subprocess.STDOUT)
        if p.returncode:
            raise SystemExit(f"FAIL the reference's run {i + 1} failed\n{(root / f'serial-{i + 1}.log').read_text()[-3000:]}")


def cross(root: Path, kills: dict, extra: dict | None = None) -> list[int]:
    """The orchestrator run until it ends with 0 (at most 4 times): its exit codes."""
    codes = []
    for attempt in range(4):
        envs = {}
        for set_ in ("main", "scholar"):
            e = env(root, set_, link_env(root) | (extra or {}))
            if attempt == 0 and set_ in kills:
                e["KILL_HOOK"], e["KILL_N"] = kills[set_][0], str(kills[set_][1])
            envs[set_] = e
        (root / "envs.json").write_text(json.dumps(envs))
        (root / "finalise.py").write_text(FINALISE.format(src=str(ROOT / "src"), bench=str(ROOT / "bench")))
        # (in a session of its own, killed once it ends: a killed run's spawned workers are
        # orphaned, as by SIGKILL, and would otherwise outlive the test)
        p = subprocess.Popen([sys.executable, "-c", ORCH.format(src=str(ROOT / "src"))], cwd=root,
                             env=env(root, "main"), stdout=(root / "orchestrator.log").open("a"),
                             stderr=subprocess.STDOUT, start_new_session=True)
        try:
            p.wait(timeout=900)
        finally:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        codes.append(p.returncode)
        if p.returncode == 0:
            return codes
    logs = "\n".join(f"--- {f.name}\n{f.read_text()[-2000:]}" for f in sorted(root.glob("*.log")))
    raise SystemExit(f"FAIL the orchestrator never ended with 0: {codes}\n{logs}")


def sha(p: Path) -> str:
    with open(p, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def outcome(root: Path) -> dict:
    from link import _digest

    sets = (RELEASE, f"{RELEASE}-scholar")
    local = {str(p.relative_to(root)): sha(p) for s in sets for p in sorted((root / "releases" / s / "hub").rglob("*.parquet"))}
    meta = {s: json.loads(p.read_text()) for s in ("main", "scholar")
            if (p := root / "card-figures" / s / "dataset_cards_metadata.json").exists()}
    return {"digest": _digest(root / "fakehub", f"build-{RELEASE}"),
            "done": all((root / "releases" / s / "state" / "finalise.done").exists() for s in sets),
            "local": local, "meta": meta,
            "files": sum(1 for p in (root / "fakehub").rglob("*.parquet") if f"build-{RELEASE}" in p.parts)}


def events(root: Path) -> list[dict]:
    return [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]


def runs_of(ev: list[dict]) -> list[dict]:
    """Each finalise process (by pid, in start order): its set, start, end, lock and mark."""
    out: dict[int, dict] = {}
    for e in ev:
        r = out.setdefault(e["pid"], {"set": e["set"], "pid": e["pid"]})
        r.setdefault(e["what"], e["t"])
        if e["what"] in ("commit", "compact", "claims_labels"):
            r.setdefault(e["what"] + "s", []).append(e)
    return sorted(out.values(), key=lambda r: r["start"])


def same(got: dict, ref: dict) -> list[str]:
    bad = [k for k in ("digest", "local", "meta") if got[k] != ref[k]]
    return bad + ([] if got["done"] else ["not finalised"])


def recipe(tmp: Path) -> bool:
    """`just overlap-release` and `just release`, their commands stand-ins that log."""
    real_just = shutil.which("just")
    if real_just is None:
        print("skip the recipes: no `just`", flush=True)
        return True
    stub = """#!/usr/bin/env bash
echo "$(basename "$0") start $(date +%s.%N) scholar=${{WIKIDATA_SCHOLAR:-}} nice=$(nice) args=$*" >> {log}
case "$(basename "$0")" in
  process-wikidata) sleep $([ -n "${{WIKIDATA_SCHOLAR:-}}" ] && echo 3 || echo 1)
                    [ -n "${{WIKIDATA_SCHOLAR:-}}" ] && [ -n "${{SCHOLAR_FAILS:-}}" ] && exit 1 ;;
  finalise-release-runs) if [ "${{1:-}}" = --after ]; then
                           for _ in $(seq 100); do [ -e "$2" ] && break; sleep 0.1; done
                           echo "finalise-release-runs after $(date +%s.%N) $(cat "$2" 2>/dev/null)" >> {log}
                         fi ;;
esac
echo "$(basename "$0") end $(date +%s.%N) scholar=${{WIKIDATA_SCHOLAR:-}}" >> {log}
"""
    ok = True
    cases = (("overlap-release, default", ["overlap-release", RELEASE], {}),
             ("overlap-release, cross-run", ["overlap-release", RELEASE], {"WIKIDATA_FINALISE_CROSS_RUN": "1"}),
             ("overlap-release, cross-run, the scholarly run failing", ["overlap-release", RELEASE],
              {"WIKIDATA_FINALISE_CROSS_RUN": "1", "SCHOLAR_FAILS": "1"}),
             ("overlap-release, without", ["overlap-release", RELEASE], {"WIKIDATA_FINALISE_CROSS_RUN": "0"}),
             ("release, cross-run", ["release", RELEASE, "20261001"], {"WIKIDATA_FINALISE_CROSS_RUN": "1"}),
             ("release, default", ["release", RELEASE, "20261001"], {}),
             ("release, without", ["release", RELEASE, "20261001"], {"WIKIDATA_FINALISE_CROSS_RUN": "0"}))
    for n, (case, args, extra) in enumerate(cases):
        d = tmp / f"recipe-{n}"
        (d / "bin").mkdir(parents=True)
        log = d / "calls.log"
        shutil.copyfile(ROOT / "Justfile", d / "Justfile")
        for name in ("process-wikidata", "finalise-wikidata", "finalise-release-runs", "download-dump", "split-dump",
                     "promote-release"):
            (d / "bin" / name).write_text(stub.format(log=log))
            (d / "bin" / name).chmod(0o755)
        e = {k: v for k, v in os.environ.items() if not k.startswith("WIKIDATA_")}
        e |= {"PATH": f"{d / 'bin'}:{e['PATH']}", **extra}
        code = subprocess.run([real_just, *args], cwd=d, env=e, capture_output=True, text=True).returncode
        calls = [line.split() for line in log.read_text().splitlines()]
        runs = [c for c in calls if c[0] == "finalise-release-runs" and c[1] == "start"]
        fins = [c for c in calls if c[0] == "finalise-wikidata" and c[1] == "start"]
        after = [c for c in calls if c[:2] == ["finalise-release-runs", "after"]]
        ends = {c[3]: float(c[2]) for c in calls if c[0] == "process-wikidata" and c[1] == "end"}
        if case in ("overlap-release, cross-run", "overlap-release, default"):
            good = (code == 0 and len(runs) == 1 and not fins and runs[0][4] == "nice=10"
                    and runs[0][5:] == ["args=--after", runs[0][6]] and ends.get("scholar=")
                    and ends["scholar="] <= float(runs[0][2]) < ends["scholar=1"]
                    and after and after[0][3] == "0" and float(after[0][2]) >= ends["scholar=1"])
        elif case.endswith("failing"):
            good = code != 0 and len(runs) == 1 and after and after[0][3] == "1"
        elif case == "overlap-release, without":
            good = code == 0 and not runs and len(fins) == 1
        elif case in ("release, cross-run", "release, default"):
            good = code == 0 and len(runs) == 1 and runs[0][5:] == ["args="] and not fins
        else:
            good = code == 0 and not runs and len(fins) == 3
        print(f"{'ok  ' if good else 'FAIL'} the recipe: {case}" + ("" if good else f": {code} {calls}"), flush=True)
        ok &= bool(good)
    return ok


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
        base = tmp / "base"
        overlap(base, dump_slice)
        ref_root = tmp / "reference"
        prepare(base, ref_root)
        t = time.monotonic()
        serial(ref_root)
        ref_wall = time.monotonic() - t
        ref = outcome(ref_root)
        ok = ref["done"] and ref["files"] > 0
        print(f"{'ok  ' if ok else 'FAIL'} reference (one run after another): {ref['files']} files on the build "
              f"branches, both sets finalised, {ref_wall:.1f} s", flush=True)
        failed |= not ok

        root = tmp / "cross"
        prepare(base, root)
        t = time.monotonic()
        codes = cross(root, {})
        wall = time.monotonic() - t
        got, runs = outcome(root), runs_of(events(root))
        bad = same(got, ref)
        sets = [r["set"] for r in runs]
        overlaps = [(runs[i + 1]["start"] - runs[i]["end"], runs[i + 1]["start"] - runs[i].get("written", 1e18))
                    for i in range(len(runs) - 1)]
        built = {r["set"] for r in runs if r.get("claims_labels")}
        # (the main set's second run past its lock only once its first has ended)
        second = runs[2]["locked"] >= runs[0]["end"] if len(runs) == 3 else False
        good = (not bad and codes == [0] and sets == ["main", "scholar", "main"] and built == {"main", "scholar"}
                and all(a < 0 <= b for a, b in overlaps) and second)
        print(f"{'ok  ' if good else 'FAIL'} cross-run, uninterrupted: the same build branches, local copies and "
              f"cards' metadata, both sets finalised ({', '.join(bad) or 'all equal'}); {wall:.1f} s against "
              f"{ref_wall:.1f} s; each next run started " + ", ".join(f"{-a:.1f} s before the one before ended and "
                                                                   f"{b:.1f} s after its mark" for a, b in overlaps)
              + f"; claims_labels built for {sorted(built)}; main's second run locked "
              f"{runs[2]['locked'] - runs[0]['end']:+.1f} s after its first ended", flush=True)
        failed |= not good

        root = tmp / "gate0"
        prepare(base, root)
        codes = cross(root, {}, {"WIKIDATA_FINALISE_CROSS_RUN_GB": "0"})
        got, runs = outcome(root), runs_of(events(root))
        bad = same(got, ref)
        main_commits = [c["end"] for c in runs[0].get("commits", [])]
        first = min(c["t"] for c in runs[1].get("compacts", [{"t": 0}]))
        good = not bad and codes == [0] and main_commits and first >= max(main_commits) - 0.05
        print(f"{'ok  ' if good else 'FAIL'} cross-run, gated at 0 GB: the same outcome ({', '.join(bad) or 'all equal'}); "
              f"the scholarly run's first table started {first - max(main_commits or [first]):+.1f} s after the main "
              "run's last commit", flush=True)
        failed |= not good

        for i, kill in enumerate(points):
            root = tmp / f"kill-{i}"
            prepare(base, root)
            codes = cross(root, KILLS[kill])
            got = outcome(root)
            bad = same(got, ref)
            reached = sorted(p.name for p in (root / "kills").glob("*")) if (root / "kills").exists() else []
            want = sorted(KILLS[kill])
            good = not bad and reached == want and codes[0] != 0 and codes[-1] == 0
            print(f"{'ok  ' if good else 'FAIL'} killed at {kill}: {', '.join(bad) or 'the same outcome, byte for byte'}; "
                  f"killed {reached or 'none'} (of {want}); the orchestrator's exit codes {codes}", flush=True)
            failed |= not good
    if failed:
        sys.exit(1)
    print("all passed")


if __name__ == "__main__":
    main()
