#!/usr/bin/env python3
"""Test a release's claims schema growing as it runs (src/wikidata/release_schema.py),
end to end: the split, both sets' runs (main.run) and their finalises' local stages, each
a process of its own, on a dump slice made of the chunk files given (a main set's and a
scholarly set's, bench/dumpslice.py's lines), cut into chunks of 1,000 entities, in
groups of about two chunks, the Hub a stand-in (test_group_restart.FakeHub: each repo's
build branch on disk). The slice has new fields injected (INJECT): a mainsnak's string
field in two chunks of the main set, a reference snak's datavalue integer field in
another, and a statement's record field in a scholarly entity, none of them in the
stored claims schema.

- **fits:** the slice with nothing injected. No schema version is added, no claims file
  is tagged, and (with BASELINE_SRC, a checkout of master's src/) the same files are
  uploaded and published, byte for byte, as master's code makes: no regression when the
  dump fits.
- **native:** the injected slice with the release schema given its final version before
  the split (the log of `evolved`): every part's claims are built by the split's fast
  path (none left as JSON, the new fields among them), and every file is in that version.
- **evolved:** the injected slice, the fields found as the runs process their chunks
  (auto mode): the main set's first group (chunks 0-1) is uploaded and verified before
  its chunk 2 is processed (its claims wait), so a field is added once groups are on the
  build branch. The log's final schema is native's, in the same order; the build branch's
  claims files are each exactly the schema of the version they are tagged (the first
  group's untagged); finalise publishes, byte for byte, native's files, each of exactly
  the frozen schema; and each injected value is in the published claims, as injected.
- **overlapped, killed:** the split and both runs at once (the runs following the split),
  each process killed at a kill point and run again until it completes (finalise too),
  publish native's files, byte for byte, with native's schema.
- **strict:** WIKIDATA_SCHEMA_EVOLUTION=strict: a run halts at a new field, naming it
  and `just accept-schema`; its proposal is pending and no version added; accepted
  (python -m wikidata.release_schema accept), the run resumes, until it completes; the
  published files are native's, byte for byte.
- **incompatible:** a statement's rank an integer: the run halts with the path, both
  types and the chunk, no version is added, and the chunk's claims file is not written.
- **concurrent:** two processes proposing at once, released by one barrier, in either
  order (different fields and the same field): one final schema, the same every round,
  in the canonical order (stored fields first, the rest sorted by name).
- **promote:** `promote` of evolved's release writes claims.promoted.json and
  claims.avsc, which make the stored schema the release's final one.

A kill is os._exit at a hook installed in every process of the run (sitecustomize, so in
spawned workers too), the work cut off where it is.

Usage: python scripts/test_schema_evolution.py MAIN.parquet SCHOLAR.parquet [scenario ...]
       (e.g. bench-data's r50.parquet and s30.parquet; BASELINE_SRC=<master>/src optional)
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

CHUNK = 1000
N_MAIN, N_SCHOLAR = 4000, 2000  # rows of each file, in blocks of 500 alternately
GROUP_SOURCE_GB = "0.0007"

# What is injected, by the line it goes in (the first line at or after it that has the
# place for it, in the set given): (name, set, line)
INJECT = [("zz_flag", "main", 2100), ("newkey", "main", 3100), ("aa_note", "scholar", 3600), ("zz_flag", "main", 4100)]
MARKERS = {"zz_flag": '"zz_flag":"injected"', "newkey": '"newkey":4242', "aa_note": '"aa_note":{"n":7,"text":"note"}'}

KILLS = {
    "run: a version added, before its chunk's claims file is written": {"main": ("propose", 1)},
    "run: mid merge, an earlier version's file written in the group's": {"main": ("upgrade_file", 1)},
    "split: a part built in a later version": {"split": ("built_later", 1)},
    "finalise: the sort reading a file of an earlier version": {"finalise": ("conform", 1)},
}

HOOKS = r'''
import os, sys, time
hook, n = os.environ.get("KILL_HOOK"), int(os.environ.get("KILL_N") or 0)
calls = [0]
def tick(why=""):
    calls[0] += 1
    if calls[0] == n:
        print(f"[test] killed at {hook} {why}", flush=True)
        os._exit(137)
if hook or os.environ.get("WAIT_FIRST_GROUP"):
    from wikidata import release_schema, schema_evolution
if hook == "propose":
    f = release_schema.ReleaseSchema.propose
    def propose(self, *a, **k):
        before = self.latest()
        v = f(self, *a, **k)
        if v > before:
            tick()
        return v
    release_schema.ReleaseSchema.propose = propose
if hook == "upgrade_file":
    f = release_schema.ReleaseSchema.upgrade_file
    release_schema.ReleaseSchema.upgrade_file = lambda self, *a, **k: (f(self, *a, **k), tick())[0]
if hook == "conform":
    f = schema_evolution.conform_table
    def conform(t, schema):
        out = f(t, schema)
        if not t.schema.remove_metadata().equals(schema):
            tick()
        return out
    schema_evolution.conform_table = conform
if hook == "built_later":
    from wikidata import claims_build
    for name in ("write_claims", "write_built"):
        def wrap(*a, write=getattr(claims_build, name), **k):
            # A part from the split's chunk 3 on waits for the runs to find a field (they process
            # the chunks numbered before it), so the split has parts left to build in a later version
            import re
            dst = next(str(x) for x in a if "chunk_" in str(x))
            if int(re.search(r"chunk_(\d+)", dst).group(1)) >= 3:
                t = time.monotonic()
                while release_schema.CLAIMS.latest() == 0 and time.monotonic() - t < 300:
                    time.sleep(0.2)
            r = write(*a, **k)
            if claims_build.current()[0] > 0:
                tick()
            return r
        setattr(claims_build, name, wrap)
if os.environ.get("WAIT_FIRST_GROUP") and os.environ.get("WIKIDATA_SCHOLAR") != "1":
    # A main chunk from 2 on waits for the first group to be verified (on the stand-in Hub)
    from wikidata import process
    f = process.normalise_claims_direct
    def wait(input_path, *a, **k):
        import re
        c = int(re.search(r"chunk_(\d+)", input_path.name).group(1))
        ledger = process.Path("releases") / os.environ["WIKIDATA_RELEASE"] / "state" / "groups.jsonl"
        while c >= 2 and not (ledger.exists() and '"stage": "verified"' in ledger.read_text()):
            time.sleep(0.05)
        return f(input_path, *a, **k)
    process.normalise_claims_direct = wait
'''

SPLIT = """
import sys
sys.path.insert(0, {src!r})
from wikidata import dump
dump.CHUNK_ENTITIES = {chunk}
dump.split()
"""

RUN = """
import sys
from functools import partial
from pathlib import Path
sys.path.insert(0, {src!r})
sys.path.insert(0, {scripts!r})
from test_group_restart import FakeHub
from wikidata import main
main.close_group = partial(main.close_group, api=FakeHub(Path("fakehub").resolve()))
main.run()
"""

FINALISE = """
import json, shutil, sys
from pathlib import Path
sys.path.insert(0, {src!r})
sys.path.insert(0, {scripts!r})
import pyarrow.parquet as pq
import polars as pl
from test_group_restart import FakeHub, digest
from wikidata import compact, sort_by_id
from wikidata.config import COMPACT_DIR, HF_USER, HUB_COPY_DIR, HUB_REVISION, REPO_TARGET, STATE_DIR, WORK_DIR, Table
from wikidata.push.groups import _read
from wikidata.state import get_all_state
try:
    from wikidata import release_schema as rs_mod
except ImportError:  # (master's code)
    rs_mod = None
# The claims sorted through buckets (about 1.5 MB of their rows each), the others in memory
plan_rows = sort_by_id.plan_rows
sort_by_id.plan_rows = lambda t, r, d, m=None, f=None: plan_rows(t, r, d, 0 if t == Table.CLAIMS else m, f)
sort_by_id.SORT_BUCKET_BYTES = 1_500_000
hub = FakeHub(Path("fakehub").resolve())
branch = {{t: hub._dir(REPO_TARGET.format(hf_user=HF_USER, tbl=t), HUB_REVISION) for t in Table}}
uploaded = {{f"{{t}}/{{p.relative_to(d)}}": digest(p) for t, d in branch.items() for p in sorted(d.rglob("*.parquet"))}}
versions = {{}}
if rs_mod is not None:
    rs = rs_mod.CLAIMS
    for p in sorted(branch[Table.CLAIMS].rglob("*.parquet")):
        versions[p.name] = rs.check_file(p, rs.latest())  # (raises if not exactly its version's)
state = STATE_DIR.resolve()
done = [t for t in Table if t != Table.CLAIMS_LABELS]
for t in sorted(done, key=lambda t: t != Table.CLAIMS):
    if compact.last_stage(state, t) == "done":
        continue
    for p in branch[t].rglob("*.parquet"):
        dst = COMPACT_DIR / "src" / t / p.relative_to(branch[t])
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dst)
    compact.record_stage(state, t, "downloaded")
    compact.compact_table(t, "unused", state)
for t in sorted(done, key=lambda t: t != Table.CLAIMS):
    stage = sort_by_id.last_stage(state, t)
    (HUB_COPY_DIR / t).mkdir(parents=True, exist_ok=True)
    if stage is None:
        sort_by_id.check_compacted_copy(t)
        sort_by_id.record_stage(state, t, "sourced")
        stage = "sourced"
    if stage == "sourced":
        sort_by_id.remove_compact_src(t)
        sort_by_id.write_table(t)
        sort_by_id.record_stage(state, t, "written")
        stage = "written"
    if stage == "written":
        sort_by_id.write_metadata(t)
        sort_by_id.replace_local_copy(t)
        sort_by_id.record_stage(state, t, "done")
published = {{str(p.relative_to(HUB_COPY_DIR)): digest(p) for p in sorted(HUB_COPY_DIR.rglob("*.parquet"))
             if ".cache" not in p.parts}}
claims = sorted(p for p in (HUB_COPY_DIR / "claims").rglob("*.parquet") if ".cache" not in p.parts)
out = {{"uploaded": uploaded, "published": published, "versions": versions}}
if rs_mod is not None and rs_mod.CLAIMS.latest():
    rs = rs_mod.CLAIMS
    frozen = rs.frozen()
    want = rs.file_schema(frozen)
    out["frozen"] = frozen
    out["schema"] = str(rs.schema(frozen))
    # every published claims file exactly the frozen schema, untagged
    out["exact"] = all(pq.read_schema(p).equals(want, check_metadata=True) for p in claims)
    rows = pl.concat([pl.read_parquet(p) for p in claims])
    out["claims_rows"] = rows.height
    out["text"] = {{i: rows.filter(pl.col("id") == i).write_ndjson() for i in json.loads({ids!r})}}
left = sorted(str(p) for d in ("results", "staging") for p in (WORK_DIR / d).rglob("*") if p.is_file())
out |= {{"left": left, "steps": sorted(set(get_all_state(state, "chunk_*")["step"].to_list())),
        "groups": [(r["first"], r["last"], r["stage"]) for r in _read(state / "groups.jsonl") if r["stage"] == "done"]}}
Path(sys.argv[1]).write_text(json.dumps(out))
"""


def slice_lines(main: str, scholar: str, inject: bool, incompatible: bool = False) -> tuple[list[bytes], dict]:
    """The slice's dump lines (the files' rows in blocks of 500 alternately), and the ids
    of the entities given each injected field."""
    import orjson
    import pyarrow.parquet as pq
    from dumpslice import dump_escaped, orjson_line

    rows = {"main": pq.read_table(main).slice(0, N_MAIN).to_pylist(),
            "scholar": pq.read_table(scholar).slice(0, N_SCHOLAR).to_pylist()}
    order = []
    i = {"main": 0, "scholar": 0}
    while i["main"] < N_MAIN or i["scholar"] < N_SCHOLAR:
        for s in ("main", "scholar"):
            take = rows[s][i[s]: i[s] + 500]
            order += [(s, r) for r in take]
            i[s] += len(take)
    lines = [orjson_line(r) for _, r in order]
    ids: dict[str, list[str]] = {}
    plan = INJECT if inject else []
    if incompatible:
        plan = [("rank", "main", 3100)]
    for name, s, at in plan:
        for k in range(at, len(lines)):
            if order[k][0] != s:
                continue
            e = orjson.loads(lines[k])
            if not _inject(e, name):
                continue
            lines[k] = orjson.dumps(e)
            ids.setdefault(name, []).append(e["id"])
            break
    return [dump_escaped(line) for line in lines], ids


def _inject(e: dict, name: str) -> bool:
    statements = [s for ss in e.get("claims", {}).values() for s in ss]
    if not statements:
        return False
    if name == "zz_flag":
        statements[0]["mainsnak"]["zz_flag"] = "injected"
        return True
    if name == "aa_note":
        statements[0]["aa_note"] = {"text": "note", "n": 7}
        return True
    if name == "rank":
        for s in statements:
            s["rank"] = 7
        return True
    for s in statements:  # newkey: a reference snak's datavalue holding an object
        for r in s.get("references", []):
            for sn in (x for v in r["snaks"].values() for x in v):
                if isinstance(sn.get("datavalue", {}).get("value"), dict):
                    sn["datavalue"]["value"]["newkey"] = 4242
                    return True
    return False


def write_slice(dst: Path, lines: list[bytes]) -> None:
    import bz2

    with bz2.open(dst, "wb") as f:
        f.write(b"[\n" + b",\n".join(lines) + b"\n]\n")


def env(name: str, root: Path, kill: tuple[str, int] | None = None, extra: dict | None = None) -> dict:
    e = {**os.environ, "WIKIDATA_RELEASE": RELEASE, "HF_HUB_OFFLINE": "1", "WIKIDATA_WORKERS": "2",
         "WIKIDATA_GROUP_SOURCE_GB": GROUP_SOURCE_GB, "WIKIDATA_FINALISE_WORKERS": "2",
         "PYTHONPATH": str(root.parent / "hooks"),
         # (finalise's local stages run here one set at a time, as each run's own; the cross-run
         # orchestration, off by default today, is set off explicitly)
         "WIKIDATA_FINALISE_CROSS_RUN": "0"}
    for k in ("WIKIDATA_SCHOLAR", "WIKIDATA_FOLLOW_SPLIT", "KILL_HOOK", "KILL_N", "WIKIDATA_SCHEMA_EVOLUTION",
              "WAIT_FIRST_GROUP"):
        e.pop(k, None)
    if name in ("scholar", "finalise-scholar"):
        e["WIKIDATA_SCHOLAR"] = "1"
    if kill:
        e["KILL_HOOK"], e["KILL_N"] = kill[0], str(kill[1])
    return e | (extra or {})


class Scenario:
    def __init__(self, tmp: Path, name: str, lines: list[bytes], src: Path = ROOT / "src", extra: dict | None = None):
        self.root, self.src, self.extra = tmp / name, src, extra or {}
        d = self.root / "releases" / RELEASE / "dump"
        d.mkdir(parents=True)
        write_slice(d / f"wikidata-{RELEASE}-all.json.bz2", lines)

    def code(self, which: str, ids: dict | None = None) -> str:
        return {"split": SPLIT, "run": RUN, "finalise": FINALISE}[which].format(
            src=str(self.src), scripts=str(ROOT / "scripts"), chunk=CHUNK,
            ids=json.dumps(sorted({i for v in (ids or {}).values() for i in v})))

    def start(self, name: str, kill=None, follow: bool = False, extra: dict | None = None) -> subprocess.Popen:
        e = env(name, self.root, kill, self.extra | (extra or {}))
        if follow:
            e["WIKIDATA_FOLLOW_SPLIT"] = "1"
        code = self.code("split" if name == "split" else "run")
        log = (self.root / f"{name}.log").open("a")
        return subprocess.Popen([sys.executable, "-c", code], cwd=self.root, env=e, stdout=log, stderr=subprocess.STDOUT)

    def log(self, name: str, n: int = 3000) -> str:
        p = self.root / f"{name}.log"
        return p.read_text()[-n:] if p.exists() else ""

    def wait(self, procs: dict, timeout: float = 900, early: bool = False) -> dict[str, int]:
        """Each process's exit code once all have exited, or (`early`) as soon as one
        fails (the others, still running, left out)."""
        t, codes = time.monotonic(), {}
        while len(codes) < len(procs):
            for name, p in procs.items():
                if name not in codes and p.poll() is not None:
                    codes[name] = p.returncode
            if early and any(codes.values()):
                return codes
            if time.monotonic() - t > timeout:
                for p in procs.values():
                    p.kill()
                raise SystemExit(f"FAIL {self.root.name}: timed out\n" + "\n".join(self.log(n) for n in procs))
            time.sleep(0.2)
        return codes

    def finalise(self, ids: dict, kill=None) -> dict:
        out = {}
        for name in ("main", "scholar"):
            for attempt in range(4):
                k = kill if attempt == 0 else None
                e = env(f"finalise-{name}" if name == "scholar" else "finalise", self.root, k, self.extra)
                e["WIKIDATA_CARD_FIGURES_DIR"] = str(self.root / "card-figures" / name)
                p = subprocess.run([sys.executable, "-c", self.code("finalise", ids), str(self.root / f"{name}.json")],
                                   cwd=self.root, env=e, capture_output=True, text=True)
                (self.root / f"finalise-{name}.log").open("a").write(p.stdout + p.stderr)
                if p.returncode == 0:
                    break
                if not k:
                    raise SystemExit(f"FAIL {self.root.name}: {name}'s finalise failed:\n{(p.stdout + p.stderr)[-3000:]}")
            out[name] = json.loads((self.root / f"{name}.json").read_text())
        return out

    def log_json(self) -> dict | None:
        p = self.root / "releases" / RELEASE / "schema" / "claims.json"
        return json.loads(p.read_text()) if p.exists() else None

    def sequential(self, extra_runs: dict | None = None) -> dict[str, int]:
        if self.wait({"split": self.start("split")})["split"]:
            raise SystemExit(f"FAIL {self.root.name}: the split failed\n{self.log('split')}")
        return self.wait({n: self.start(n, extra=extra_runs) for n in ("main", "scholar")})


def check(ok: bool, what: str) -> bool:
    print(f"{'ok  ' if ok else 'FAIL'} {what}", flush=True)
    return ok


def clean(out: dict) -> bool:
    return all(r["steps"] == [6] and not r["left"] and len(r["groups"]) > 1 for r in out.values())


def same_published(a: dict, b: dict) -> bool:
    return all(a[n]["published"] == b[n]["published"] for n in ("main", "scholar"))


def injected_kept(out: dict, ids: dict) -> bool:
    """Each injected value is in its entity's published claims, as injected."""
    for name, entities in ids.items():
        for i in entities:
            text = "".join(r.get("text", {}).get(i, "") for r in out.values())
            if MARKERS[name] not in text.replace(" ", ""):
                print(f"     {name} not found for {i}: {text[:300]}", flush=True)
                return False
    return True


def concurrent(tmp: Path) -> bool:
    """Two processes proposing at once, in either order: one final schema."""
    code = r'''
import json, os, sys, time
sys.path.insert(0, {src!r})
import polars as pl
from wikidata import release_schema as r
from wikidata import schema_evolution as se
base = r.CLAIMS.base
fields = json.loads(sys.argv[1])
dtype = pl.Struct(base)
for path, spec in fields:
    dtype = se._with(dtype, path[:-1], lambda n, name=path[-1], t=se.dtype_of(spec): pl.Struct({{**{{f.name: f.dtype for f in n.fields}}, name: t}}))
barrier = sys.argv[2]
while not os.path.exists(barrier):
    time.sleep(0.001)
time.sleep(float(sys.argv[3]))
print(r.CLAIMS.propose(pl.Schema(dtype), sys.argv[4]))
'''.format(src=str(ROOT / "src"))
    mainsnak = ["claims", "[]", "value", "[]", "mainsnak"]
    a = [[mainsnak + ["qq"], "String"], [["claims", "[]", "value", "[]", "bb"], "Int64"]]
    b = [[mainsnak + ["qq"], "String"], [mainsnak + ["aa"], {"struct": [["z", "String"], ["y", "Int64"]]}]]
    finals, ok = set(), True
    for k, (da, db) in enumerate([(0, 0.02), (0.02, 0), (0, 0), (0.01, 0), (0, 0.01), (0, 0)]):
        root = tmp / f"concurrent-{k}"
        root.mkdir()
        barrier = root / "go"
        e = env("main", root)
        procs = [subprocess.Popen([sys.executable, "-c", code, json.dumps(f), str(barrier), str(d), name], cwd=root,
                                  env=e, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                 for f, d, name in ((a, da, "chunk_a"), (b, db, "chunk_b"))]
        time.sleep(1.5)  # (each imported and waiting)
        barrier.touch()
        res = [p.communicate() for p in procs]
        if any(p.returncode for p in procs):
            print(res)
            return False
        log = json.loads((root / "releases" / RELEASE / "schema" / "claims.json").read_text())
        changes = [c for v in log["versions"] for c in v["changes"]]
        ok &= len(log["versions"]) in (1, 2) and len([c for c in changes if c["path"][-1] == "qq"]) == 1
        from wikidata.schema_evolution import apply
        from wikidata.release_schema import CLAIMS
        import polars as pl

        final = apply(pl.Struct(CLAIMS.base), changes)
        finals.add(str(final))
        snak = final.fields[0].dtype.inner.fields[1].dtype.inner.fields[0].dtype
        names = [f.name for f in snak.fields]
        ok &= names[-2:] == ["aa", "qq"] and [f.name for f in [g for g in snak.fields if g.name == "aa"][0].dtype.fields] == ["y", "z"]
    return check(ok and len(finals) == 1, "concurrent discovery: two workers proposing at once, different and the same "
                 "fields, in either order, give one final schema, in the canonical order, every round")


def main() -> None:
    files = [a for a in sys.argv[1:] if a.endswith(".parquet")]
    wanted = [a for a in sys.argv[1:] if not a.endswith(".parquet")]
    run = lambda s: not wanted or s in wanted  # noqa: E731
    os.environ.setdefault("WIKIDATA_RELEASE", RELEASE)
    failed = False
    # (TEST_TMP: where its work goes; KEEP=1 keeps it, to look at the runs' logs)
    with TemporaryDirectory(dir=os.environ.get("TEST_TMP"), delete=not os.environ.get("KEEP")) as tmp:
        tmp = Path(tmp)
        (tmp / "hooks").mkdir()
        (tmp / "hooks" / "sitecustomize.py").write_text(HOOKS)
        plain, _ = slice_lines(files[0], files[1], inject=False)
        lines, ids = slice_lines(files[0], files[1], inject=True)
        print(f"injected: {ids}", flush=True)

        if run("concurrent"):
            failed |= not concurrent(tmp)

        if run("fits"):
            s = Scenario(tmp, "fits", plain)
            codes = s.sequential()
            out = s.finalise({})
            log = s.log_json()
            ok = not any(codes.values()) and clean(out) and (log is None or not log["versions"])
            ok &= all(v == 0 for r in out.values() for v in r["versions"].values())
            failed |= not check(ok, "fits: no version added, no claims file tagged, every chunk complete")
            if base := os.environ.get("BASELINE_SRC"):
                m = Scenario(tmp, "fits-master", plain, src=Path(base))
                codes = m.sequential()
                ref = m.finalise({})
                # (the claims group files are without their derived leaves, claims_derived:
                # every other file uploaded, and every file published, master's byte for byte)
                same = all(ref[n][k] == out[n][k] for n in ref for k in ("published", "groups"))
                differ = {f for n in ref for f in set(ref[n]["uploaded"]) | set(out[n]["uploaded"])
                          if ref[n]["uploaded"].get(f) != out[n]["uploaded"].get(f)}
                same &= all(set(ref[n]["uploaded"]) == set(out[n]["uploaded"]) for n in ref)
                same &= bool(differ) and all(f.startswith("claims/") for f in differ)
                failed |= not check(same and not any(codes.values()), "fits: the same files published, byte for byte, "
                                    f"as master's code ({base}), and uploaded but the claims group files")

        evolved = native = None
        if run("evolved") or run("native") or run("overlapped") or run("strict") or run("promote"):
            s = Scenario(tmp, "evolved", lines)
            codes = s.sequential(extra_runs={"WAIT_FIRST_GROUP": "1"})
            if any(codes.values()):
                raise SystemExit(f"FAIL evolved: {codes}\n{s.log('main')}\n{s.log('scholar')}")
            evolved = s.finalise(ids)
            log = s.log_json()
            m = evolved["main"]
            first = [p for p in m["versions"] if p.startswith("chunks-00000-")]
            ok = bool(clean(evolved) and log["versions"] and first and all(m["versions"][p] == 0 for p in first))
            ok &= any(v > 0 for v in m["versions"].values())
            failed |= not check(bool(ok), f"evolved: {len(log['versions'])} versions added as chunks showed new fields, "
                                f"the first group uploaded before any (its claims untagged), every build-branch claims "
                                f"file exactly its version's schema ({sorted(set(m['versions'].values()))})")
            failed |= not check(all(r.get("exact") for r in evolved.values()), "evolved: every published claims file "
                                "exactly the frozen schema, untagged")
            failed |= not check(injected_kept(evolved, ids), "evolved: each injected value published, as injected "
                                "(none dropped)")
            evolved_root, evolved_log = s.root, log

            # native: the final version before the split
            n = Scenario(tmp, "native", lines)
            sd = n.root / "releases" / RELEASE / "schema"
            sd.mkdir(parents=True)
            seeded = dict(evolved_log, frozen={}, pending=[])
            (sd / "claims.json").write_text(json.dumps(seeded))
            codes = n.sequential()
            native = n.finalise(ids)
            final = len(evolved_log["versions"])
            ok = not any(codes.values()) and clean(native)
            ok &= all(v == final for r in native.values() for v in r["versions"].values())
            failed |= not check(ok, "native: every claims file in the final version")
            failed |= not check(_none_as_json(n.root), "native: every part's claims built by the split's fast path, "
                                "the new fields among them (none left as JSON)")
            failed |= not check(same_published(evolved, native) and evolved["main"]["schema"] == native["main"]["schema"],
                                "evolved: the published files native's, byte for byte, in its schema")

        if run("overlapped"):
            for kill in [None, *KILLS]:
                hooks = KILLS.get(kill, {})
                s = Scenario(tmp, f"overlapped-{list(KILLS).index(kill) if kill else 'x'}", lines)
                procs = {name: s.start(name, hooks.get(name), follow=name != "split") for name in ("split", "main", "scholar")}
                for _ in range(8):
                    codes = s.wait(procs, early=True)
                    bad = {k: c for k, c in codes.items() if c}
                    if not bad:
                        break
                    if not hooks:
                        raise SystemExit(f"FAIL overlapped: {bad}\n" + "\n".join(s.log(k) for k in bad))
                    # (each failed process run again, those still running left running)
                    procs = {k: s.start(k, follow=k != "split") if k in bad else v
                             for k, v in procs.items() if k in bad or k not in codes}
                else:
                    raise SystemExit(f"FAIL overlapped {kill}: not complete after eight runs")
                got = s.finalise(ids, kill=hooks.get("finalise"))
                killed = "[test] killed" in "".join(s.log(k, 10**7) for k in ("split", "main", "scholar")) or \
                    "[test] killed" in "".join((s.root / f"finalise-{k}.log").read_text() for k in ("main", "scholar"))
                ok = clean(got) and same_published(got, native) and got["main"]["schema"] == native["main"]["schema"]
                ok &= all(r.get("exact") for r in got.values()) and (killed or not kill)
                label = f"killed at {kill}, run again" if kill else "uninterrupted"
                failed |= not check(ok, f"overlapped, {label}: native's published files, byte for byte, in its "
                                    "schema" + ("" if killed or not kill else " (NOT KILLED)"))

        if run("strict"):
            s = Scenario(tmp, "strict", lines, extra={"WIKIDATA_SCHEMA_EVOLUTION": "strict"})
            halts = 0
            if s.wait({"split": s.start("split")})["split"]:
                raise SystemExit("FAIL strict: split")
            ok = True
            for _ in range(10):
                codes = s.wait({n: s.start(n) for n in ("main", "scholar")})
                if not any(codes.values()):
                    break
                halts += 1
                text = s.log("main", 10**6) + s.log("scholar", 10**6)
                log = s.log_json()
                ok &= "accept-schema" in text and bool(log["pending"]) and \
                    len(log["versions"]) == sum(1 for v in log["versions"] if v.get("accepted"))
                p = subprocess.run([sys.executable, "-m", "wikidata.release_schema", "accept"], cwd=s.root,
                                   env=env("main", s.root), capture_output=True, text=True)
                ok &= p.returncode == 0 and "accepted as version" in p.stdout
            else:
                raise SystemExit("FAIL strict: not complete after ten runs")
            got = s.finalise(ids)
            ok &= halts > 0 and clean(got) and same_published(got, native) and got["main"]["schema"] == native["main"]["schema"]
            failed |= not check(ok, f"strict: halted {halts} times with the fields proposed and pending, each accepted "
                                "(accept-schema) and resumed; native's published files, byte for byte")

        if run("incompatible"):
            bad_lines, bad_ids = slice_lines(files[0], files[1], inject=False, incompatible=True)
            s = Scenario(tmp, "incompatible", bad_lines)
            codes = s.sequential()
            text = s.log("main", 10**6)
            log = s.log_json()
            msg = [line for line in text.splitlines() if "claims[].value[].rank" in line]
            ok = codes["main"] != 0 and bool(msg) and "not additive" in text and (log is None or not log["versions"])
            ok &= "String in the release schema" in msg[0] and "chunk_" in msg[0] if msg else False
            failed |= not check(ok, "incompatible: a rank as an integer halts the run, naming the path, both types and "
                                f"the chunk, nothing added ({msg[0].strip() if msg else text[-500:]})")

        if run("promote") and evolved is not None:
            dst = tmp / "promoted"
            code = f"""
import sys; sys.path.insert(0, {str(ROOT / 'src')!r})
from pathlib import Path
from wikidata import release_schema
release_schema.promote(Path({str(dst)!r}))
"""
            p = subprocess.run([sys.executable, "-c", code], cwd=evolved_root, env=env("main", evolved_root),
                               capture_output=True, text=True)
            ok = p.returncode == 0 and (dst / "claims.promoted.json").exists()
            if ok:
                import polars as pl

                from wikidata.claims_schema import promoted_schema, stored_claims_schema
                from wikidata.schema_evolution import apply
                from wikidata.release_schema import CLAIMS

                promoted = json.loads((dst / "claims.promoted.json").read_text())
                changes = [c for v in evolved_log["versions"] for c in v["changes"]]
                final = apply(pl.Struct(CLAIMS.base), changes)
                ok &= pl.Struct(promoted_schema(stored_claims_schema(), promoted)) == final
                avsc = (dst / "claims.avsc").read_text()
                ok &= all(f in avsc for f in ("zz_flag", "newkey", "aa_note"))
            failed |= not check(ok, "promote: claims.promoted.json and claims.avsc make the stored schema the "
                                "release's final one" + ("" if ok else f"\n{p.stdout}{p.stderr}"))
    if failed:
        sys.exit(1)
    print("all passed")


def _none_as_json(root: Path) -> bool:
    """No part of the split left its claims as JSON (each set's manifest's entries)."""
    found = False
    for name in (RELEASE, f"{RELEASE}-scholar"):
        for m in (root / "releases" / name / "data").rglob("*manifest*.jsonl"):
            for line in m.read_text().splitlines():
                e = json.loads(line)
                found = True
                if e.get("claims_json"):
                    print(f"     {m.name}: {e.get('file')} claims as JSON: {e['claims_json']}", flush=True)
                    return False
    return found


if __name__ == "__main__":
    main()
