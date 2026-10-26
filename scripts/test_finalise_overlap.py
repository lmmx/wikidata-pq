#!/usr/bin/env python3
"""Test finalise's commits overlapped with its writes (sort_by_id.Committer), on small
synthetic tables of a release and a stand-in HfApi (no Hub): each repo's build branch
kept on disk, each commit logged with its operations' sha256, and its upload slept for
its bytes at an emulated rate.

main.finalise runs for real (compaction, the sort, the commits, verification and the
`done` stage), with the steps that need a whole release (the chunks' and groups' checks,
claims_labels' refs and build, the cards) replaced by stand-ins that record when they run.

- **The same published state:** serial (WIKIDATA_FINALISE_PENDING_UPLOAD_GB=0, no
  committer thread), overlapped (the default: two tables committed at once,
  WIKIDATA_FINALISE_COMMITTERS), overlapped with one table committed at a time
  (WIKIDATA_FINALISE_COMMITTERS=1), with claims sorted last
  (WIKIDATA_FINALISE_CLAIMS_LAST=1) and with a bound of one byte: the Hub's files, each
  repo's commits (their messages and operations, by sha256, in order), the local copies
  and the cards' metadata are the same.
- **Ordering:** each table's commits start only once its `written` stage is recorded,
  with each file's sha256 as the manifest lists it; claims' refs are collected once its
  sorted copy is local; claims_labels is built once the set's labels and claims are done;
  the cards come after every table's verification; serially, each table is committed
  before the next is started, and overlapped, a table is written while an earlier one
  commits; with two committers, two tables' commits are in flight at once (one at a time
  with one), and the tables are done (`done`) in the order they were handed over.
- **The bound:** no table is started while more than the bound is written and not yet
  committed.
- **Restart:** killed (os._exit, as by SIGKILL) mid-write of a table while the one before
  it commits; mid-commit (after a batch, before the next lands); after a commit lands,
  before `committed` is recorded; between commit and verify; after verify, before
  `verified` is recorded; mid-way through the `done` stage; with two tables' commits in
  flight at once; once a table handed over later is verified while an earlier one still
  commits. Each run again to the same published state and commits as above, byte for
  byte.
- **Errors:** a commit failing in the committer stops finalise with an error (exit code
  not 0, "COMMIT FAILED"), no table started after its next handover, and the rerun
  completes to the same state; a failure in the writer, likewise.

Usage: python scripts/test_finalise_overlap.py
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
from types import SimpleNamespace

RELEASE = "finaliseoverlap"
TABLES = ["labels", "descriptions", "aliases", "links", "claims", "claims_labels", "entities"]
KEYS = {"labels": ["en", "de", "fr", "zh"], "descriptions": ["en", "de", "fr"], "aliases": ["en", "de"],
        "links": ["enwiki", "dewiki"], "claims": ["all"], "entities": ["all"], "claims_labels": ["en", "de"]}
ROWS = {"claims": 120_000, "entities": 6_000}
GROUPS = [(0, 9), (10, 19), (20, 29)]


def check(name: str, ok: bool) -> None:
    print(f"{'ok  ' if ok else 'FAIL'} {name}", flush=True)
    if not ok:
        sys.exit(1)


def sha(p: Path) -> str:
    with open(p, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


# -- The child: one finalise run in a work directory -------------------------------------


def log(hub: Path, event: str, table: str = "", **kw) -> None:
    with open(hub / "events.jsonl", "a") as f:
        f.write(json.dumps({"t": time.monotonic(), "event": event, "table": table, **kw}) + "\n")


class FakeHub:
    """The HfApi calls the sort's commit and verify make, against a directory, each
    commit's upload slept for its bytes at FAKE_RATE bytes a second."""

    def __init__(self, root: Path, kill: str, rate: float, ledger: Path):
        self.root, self.kill, self.rate, self.ledger = root, kill, rate, ledger
        self.commits = 0
        self.inflight: set[str] = set()  # (the tables whose commit is in flight)

    def _dir(self, repo_id, revision):
        return self.root / repo_id.replace("/", "__") / (revision or "main")

    def list_repo_tree(self, repo_id, repo_type=None, recursive=False, revision=None, **kw):
        d = self._dir(repo_id, revision)
        return [SimpleNamespace(path=str(p.relative_to(d)), size=p.stat().st_size,
                                lfs=SimpleNamespace(sha256=sha(p)))
                for p in sorted(d.rglob("*")) if p.is_file()]

    def create_commit(self, repo_id, repo_type=None, revision=None, operations=(), commit_message="", **kw):
        from huggingface_hub import CommitOperationAdd

        table = repo_id.split("wikidata-", 1)[1]
        stages = [json.loads(line) for line in self.ledger.read_text().splitlines() if line]
        stage = next((r["stage"] for r in reversed(stages) if r["table"] == table), None)
        manifest = {f"{e['key']}/{f['name']}": f["sha256"]
                    for line in (Path(SORT_OUT) / table / "manifest.jsonl").read_text().splitlines()
                    for e in [json.loads(line)] for f in e["files"]}
        adds = [op for op in operations if isinstance(op, CommitOperationAdd)]
        bad = [op.path_in_repo for op in adds if sha(Path(op.path_or_fileobj)) != manifest.get(op.path_in_repo)]
        if stage != "written" or bad:
            with open(self.root / "violations.jsonl", "a") as f:
                f.write(json.dumps({"table": table, "stage": stage, "unchecked": bad}) + "\n")
        log(self.root, "commit_start", table)
        self.commits += 1
        if self.kill == "two_at_once" and self.inflight - {table}:
            log(self.root, "killed", table, inflight=sorted(self.inflight))
            os._exit(9)  # (another table's commit in flight, this one's started)
        self.inflight.add(table)
        if self.kill == f"commit_fails:{table}":
            raise RuntimeError("the stand-in Hub refused the commit")
        if self.kill == f"mid_commit:{table}" and self.commits_of(table) == 1:
            os._exit(9)  # (the first batch landed, the second never does)
        if self.kill == "write_during_commit" and table == "claims":
            (self.root / "claims-committing").touch()
            time.sleep(60)  # (the writer is killed meanwhile)
        time.sleep(sum(Path(op.path_or_fileobj).stat().st_size for op in adds) / self.rate)
        d = self._dir(repo_id, revision)
        ops = []
        for op in operations:
            if isinstance(op, CommitOperationAdd):
                dst = d / op.path_in_repo
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(op.path_or_fileobj, dst)
                ops.append(["add", op.path_in_repo, sha(dst)])
            else:
                (d / op.path_in_repo).unlink()
                ops.append(["delete", op.path_in_repo])
        with open(self.root / "commits.jsonl", "a") as f:
            f.write(json.dumps({"repo": repo_id, "message": commit_message, "ops": ops}) + "\n")
        self.inflight.discard(table)
        log(self.root, "commit_end", table)

    def commits_of(self, table: str) -> int:
        path = self.root / "commits.jsonl"
        lines = path.read_text().splitlines() if path.exists() else []
        return sum(json.loads(line)["repo"].endswith(f"wikidata-{table}") for line in lines)


SORT_OUT = ""


def frame(table: str, key: str, n: int, seed: int):
    import polars as pl

    col = "ref" if table == "claims_labels" else "id"
    ids = [f"Q{(i * 7919 + seed) % (n // 2 + 1)}" for i in range(n)]
    return pl.DataFrame({col: ids, "key": key, "value": [f"{table} {key} {i} {'x' * (i % 13)}" for i in range(n)]})


def child(work: Path, kill: str) -> None:
    os.chdir(work)
    import polars as pl  # noqa: F401

    from wikidata import compact, main
    from wikidata import sort_by_id as sbi
    from wikidata.config import COMPACT_DIR, HUB_COPY_DIR, SORT_DIR, STATE_DIR, Table
    from wikidata.main import FINALISE_DONE

    global SORT_OUT
    SORT_OUT = str(SORT_DIR / "out")
    hub = Path("fakehub").resolve()
    rate = float(os.environ.get("FAKE_RATE") or 2e6)
    sample = Path(os.environ["FINALISE_SAMPLE"]) if os.environ.get("FINALISE_SAMPLE") else None
    if sample and not STATE_DIR.exists():  # the bench pipeline's kept group files (measure.py)
        STATE_DIR.mkdir(parents=True)
        files = sorted((sample / "staging").glob("*/*/*/*.parquet"))  # group/table/key/file
        last = max(int(f.stem.split("-")[-1]) for f in files)
        (STATE_DIR / f"chunk_{last}.jsonl").write_text("")
        for f in files:
            t, key = f.parent.parent.name, f.parent.name
            if t == "claims_labels":
                continue
            for dst in (COMPACT_DIR / "src" / t / key / f.name,
                        hub / f"test__wikidata-{t}" / f"build-{RELEASE}" / key / f.name):
                dst.parent.mkdir(parents=True, exist_ok=True)
                os.link(f, dst)
        for t in {f.parent.parent.name for f in files} - {"claims_labels"}:
            compact.record_stage(STATE_DIR, Table(t), "downloaded")
    if not STATE_DIR.exists():  # the release's group files: compaction's source and the build branch
        STATE_DIR.mkdir(parents=True)
        (STATE_DIR / f"chunk_{GROUPS[-1][1]}.jsonl").write_text("")
        for t in TABLES:
            if t == "claims_labels":
                continue  # (built by finalise, below)
            for k, key in enumerate(KEYS[t]):
                df = frame(t, key, ROWS.get(t, 2_000 * (k + 1)), k)
                for (a, b), part in zip(GROUPS, df.iter_slices(-(-df.height // len(GROUPS)))):
                    name = f"chunks-{a:02d}-{b:02d}.parquet"
                    p = COMPACT_DIR / "src" / t / key / name
                    p.parent.mkdir(parents=True, exist_ok=True)
                    part.write_parquet(p)
                    remote = hub / f"test__wikidata-{t}" / f"build-{RELEASE}" / key / name
                    remote.parent.mkdir(parents=True, exist_ok=True)
                    remote.write_bytes(p.read_bytes())
            compact.record_stage(STATE_DIR, Table(t), "downloaded")
    if not sample:  # claims (120k rows) through buckets, the rest in memory; labels' keys two a commit
        sbi.SORT_IN_MEMORY_BYTES = 1_500_000
        sbi.COMPACT_COMMIT_MAX_ADDS = 2
    api = FakeHub(hub, kill, rate, sbi._ledger(STATE_DIR))
    sbi.HfApi = main.HfApi = lambda: api
    main.get_next_chunk = lambda state_dir: None
    main.unfinished_group = lambda state_dir: None
    main.OTHER_WORK_DIR = Path("other")
    main.sort_stage = lambda state_dir, table: "done"  # (the other set's labels)

    def collect_refs_stage(state_dir):
        parts = sorted((HUB_COPY_DIR / "claims" / "all").glob("part-*.parquet"))
        log(hub, "refs", "claims", sorted_copy=bool(parts) and sbi.last_stage(STATE_DIR, Table.CLAIMS) == "done")

    def build_claims_labels(repo_id, state_dir, last, api_):
        log(hub, "build", "claims_labels", labels=sbi.last_stage(STATE_DIR, Table.LABEL),
            claims=sbi.last_stage(STATE_DIR, Table.CLAIMS))
        kept = sorted((sample / "staging").glob("*/claims_labels/*/*.parquet")) if sample else []
        for f in kept:  # (the pipeline's group files of it, as if built)
            dst = COMPACT_DIR / "src" / "claims_labels" / f.parent.name / f.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, dst)
        for k, key in enumerate([] if kept else KEYS["claims_labels"]):
            p = COMPACT_DIR / "src" / "claims_labels" / key / "chunks-00-29.parquet"
            p.parent.mkdir(parents=True, exist_ok=True)
            frame("claims_labels", key, 3_000 * (k + 1), k).write_parquet(p)

    main.collect_refs_stage = collect_refs_stage
    main.build_claims_labels = build_claims_labels
    main.update_stats = lambda: log(hub, "stats")
    main.write_cards = lambda: (log(hub, "cards", stages={t: sbi.last_stage(STATE_DIR, Table(t)) for t in TABLES}), {})[1]

    # Kill points and logged stages
    compact_table, record_stage = main.compact_table, sbi.record_stage
    verify_table, append = sbi.verify_table, sbi._append_jsonl
    finish_table, move = sbi.finish_table, sbi.shutil.move
    moved, finishing = [], []

    def compact_table_(table, repo_id, state_dir):
        log(hub, "start", str(table), pending=main_committer_pending())
        if kill == f"writer_fails:{table}":
            raise RuntimeError("the writer failed")
        compact_table(table, repo_id, state_dir)

    def append_(path, entry):
        append(path, entry)
        if kill == "write_during_commit" and "/labels/" in str(path) and path.name == "manifest.jsonl":
            while not (hub / "claims-committing").exists():
                time.sleep(0.05)
            os._exit(9)  # (claims' commit in flight, labels half written)

    def record_stage_(state_dir, table, stage):
        if kill == f"before_ledger:{stage}:{table}":
            os._exit(9)
        record_stage(state_dir, table, stage)
        log(hub, stage, str(table))
        if kill == "verified_ahead" and stage == "verified" and api.inflight - {str(table)}:
            log(hub, "killed", str(table), inflight=sorted(api.inflight))
            os._exit(9)  # (a table verified while one handed over before it still commits)

    def verify_table_(table, repo_id, api_):
        if kill == f"before_verify:{table}":
            os._exit(9)
        verify_table(table, repo_id, api_)

    def finish_table_(table, state_dir):
        finishing.append(str(table))
        finish_table(table, state_dir)
        finishing.pop()

    def move_(src, dst):
        if kill == "mid_done:labels" and finishing == ["labels"]:
            moved.append(src)
            if len(moved) == 2:
                os._exit(9)
        return move(src, dst)

    committers = []
    Committer = sbi.Committer

    def committer_(*a, **kw):
        committers.append(Committer(*a, **kw))
        return committers[-1]

    def main_committer_pending():
        return committers[-1].pending() if committers else 0

    main.compact_table, sbi._append_jsonl, sbi.record_stage = compact_table_, append_, record_stage_
    sbi.verify_table, sbi.shutil.move, main.Committer = verify_table_, move_, committer_
    sbi.finish_table = finish_table_
    main.finalise(STATE_DIR, hf_user="test")
    import threading

    import resource

    uploaded = sum(p.stat().st_size for p in hub.glob("test__*/**/part-*.parquet"))
    log(hub, "end", threads=sorted(t.name for t in threading.enumerate()), done=(STATE_DIR / FINALISE_DONE).exists(),
        rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024, uploaded_mb=uploaded / 1e6)


# -- The parent: scenarios, each in a work directory of its own ---------------------------


def run(work: Path, env: dict, kill: str = "") -> subprocess.CompletedProcess:
    env = {**os.environ, "WIKIDATA_RELEASE": RELEASE, "WIKIDATA_FINALISE_WORKERS": "2", "HF_HUB_OFFLINE": "1",
           "WIKIDATA_FINALISE_CROSS_RUN": "0", **env}  # (L1a within a run: test_finalise_cross_run.py across)
    work.mkdir(parents=True, exist_ok=True)
    # In a session of its own, its output to files: a kill leaves the sort's spawned
    # workers orphaned (as a SIGKILL of finalise would), and they are killed with it
    with open(work / "stdout.log", "w") as out, open(work / "stderr.log", "w") as err:
        p = subprocess.Popen([sys.executable, __file__, "--child", str(work), kill], env=env,
                             stdout=out, stderr=err, start_new_session=True)
        try:
            code = p.wait(timeout=600)
        finally:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    return SimpleNamespace(returncode=code, stdout=(work / "stdout.log").read_text(),
                           stderr=(work / "stderr.log").read_text())


def outcome(work: Path) -> dict:
    hub = work / "fakehub"
    files = {str(p.relative_to(hub)): sha(p) for p in sorted(hub.glob("test__*/**/*.parquet"))}
    # Each repo's commits: each key's operations, the commit holding them (a key's in one
    # commit), and how many. Which keys share a commit follows the manifest's order, the
    # order the sort's jobs finished in, which varies from run to run with or without the
    # committer; the batches' limits are the same
    commits: dict[str, dict] = {}
    for i, line in enumerate((hub / "commits.jsonl").read_text().splitlines()):
        c = json.loads(line)
        repo = commits.setdefault(c["repo"], {"keys": {}, "n": 0, "split": False})
        repo["n"] += 1
        keys = {}
        for op in c["ops"]:
            keys.setdefault(op[1].split("/")[0], []).append(op)
        for key, ops in keys.items():
            repo["split"] |= key in repo["keys"]
            repo["keys"][key] = sorted(ops)
    rel = work / "releases" / RELEASE
    local = {str(p.relative_to(rel)): sha(p) for p in sorted((rel / "hub").rglob("*.parquet"))}
    meta = next(rel.rglob("dataset_cards_metadata.json"), None)
    return {"files": files, "commits": commits, "local": local, "meta": meta.read_text() if meta else None}


def same(got: dict, want: dict) -> bool:
    """got == want, else the parts that differ printed."""
    for part in want:
        if got[part] != want[part]:
            g, w = got[part], want[part]
            diff = sorted(k for k in set(g) | set(w) if g.get(k) != w.get(k)) if isinstance(w, dict) else "(text)"
            print(f"     {part} differ: {diff[:6] if isinstance(diff, list) else diff}")
    return got == want


def events(work: Path) -> list[dict]:
    return [json.loads(line) for line in (work / "fakehub" / "events.jsonl").read_text().splitlines()]


def ordered(ev: list[dict], serial: bool) -> bool:
    """Ordering checks on one uninterrupted run's events."""
    def at(e, t=None):
        return [x["t"] for x in ev if x["event"] == e and (t is None or x["table"] == t)]

    refs = next(x for x in ev if x["event"] == "refs")
    build = next(x for x in ev if x["event"] == "build")
    cards = next(x for x in ev if x["event"] == "cards")
    end = next(x for x in ev if x["event"] == "end")
    parts = {"refs from the sorted copy": refs["sorted_copy"],
             "claims_labels built once labels and claims are done": build["labels"] == build["claims"] == "done",
             "the cards once every table is done": all(s == "done" for s in cards["stages"].values()),
             "the cards after every verify": max(at("verified")) < cards["t"],
             "done, the committer joined": end["done"] and "finalise-committer" not in end["threads"]}
    for name, good in parts.items():
        if not good:
            print(f"     not so: {name}")
    ok = all(parts.values())
    starts = [x for x in ev if x["event"] == "start"]
    if serial:  # each table committed and verified before the next is started
        ok &= all(max(at("verified", a["table"])) < b["t"] for a, b in zip(starts, starts[1:]))
    return bool(ok)


def overlapped(ev: list[dict]) -> bool:
    """A table started while an earlier table's commit was in flight."""
    spans = {}
    for x in ev:
        if x["event"] in ("commit_start", "commit_end"):
            spans.setdefault(x["table"], []).append(x["t"])
    return any(any(s[0] < x["t"] < s[-1] for s in spans.values()) for x in ev if x["event"] == "start")


def commits_at_once(ev: list[dict]) -> int:
    """The most tables with a commit in flight at once."""
    inflight, most = set(), 0
    for x in ev:
        if x["event"] == "commit_start":
            inflight.add(x["table"])
            most = max(most, len(inflight))
        elif x["event"] == "commit_end":
            inflight.discard(x["table"])
    return most


def done_in_order(ev: list[dict]) -> bool:
    """The tables done (`done` recorded) in the order they were started (handed over)."""
    started = [x["table"] for x in ev if x["event"] == "start"]
    done = [x["table"] for x in ev if x["event"] == "done"]
    return done == [t for t in started if t in done] and len(done) == len(started)


def main() -> None:
    with TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        def scenario(name: str, env: dict, kill: str = "") -> Path:
            work = tmp / name
            r = run(work, env, kill)
            if kill:
                check(f"{name}: the first run stopped", r.returncode != 0)
                (work / "fakehub" / "claims-committing").unlink(missing_ok=True)
                r2 = run(work, env)
                check(f"{name}: the rerun completed", r2.returncode == 0 or print(r2.stdout[-3000:], r2.stderr[-3000:]))
                r = SimpleNamespace(first=r, rerun=r2)
            else:
                check(f"{name}: completed", r.returncode == 0 or print(r.stdout[-3000:], r.stderr[-3000:]))
            check(f"{name}: every commit after its table's `written`, of its checked files",
                  not (work / "fakehub" / "violations.jsonl").exists())
            return work

        serial = scenario("serial", {"WIKIDATA_FINALISE_PENDING_UPLOAD_GB": "0"})
        want = outcome(serial)
        check("serial: labels in two commits (two keys a commit), no key split",
              want["commits"]["test/wikidata-labels"]["n"] == 2 and not any(r["split"] for r in want["commits"].values()))
        check("serial: each table committed before the next started, refs and cards in place",
              ordered(events(serial), serial=True))
        for name, env in (("overlapped", {}), ("one committer", {"WIKIDATA_FINALISE_COMMITTERS": "1"}),
                          ("claims last", {"WIKIDATA_FINALISE_CLAIMS_LAST": "1"}),
                          ("bound of 1 byte", {"WIKIDATA_FINALISE_PENDING_UPLOAD_GB": "1e-9"})):
            work = scenario(name, {"FAKE_RATE": "1e6", **env})
            got, ev = outcome(work), events(work)
            check(f"{name}: the same Hub files, commits, local copies and metadata as serial", same(got, want))
            check(f"{name}: refs, claims_labels' build and the cards in place, the committer joined",
                  ordered(ev, serial=False))
            starts = [x for x in ev if x["event"] == "start"]
            if name == "bound of 1 byte":
                check(f"{name}: no table started while a byte was uncommitted", all(x["pending"] <= 1 for x in starts))
            else:
                check(f"{name}: a table written while an earlier one committed", overlapped(ev))
            if name == "claims last":
                order = [x["table"] for x in starts]
                check(f"{name}: claims sorted after the other tables", order.index("claims") == len(order) - 2)
            if name in ("overlapped", "one committer"):
                n = commits_at_once(ev)
                check(f"{name}: at most {'two' if name == 'overlapped' else 'one'} table's commits in flight at once, "
                      f"and as many ({n})", n == (2 if name == "overlapped" else 1))
            check(f"{name}: the tables done in the order handed over", done_in_order(ev))

        kills = {
            "killed mid-write of labels while claims commits": "write_during_commit",
            "killed mid-commit (labels' second batch)": "mid_commit:labels",
            "killed after labels' commit, before `committed`": "before_ledger:committed:labels",
            "killed between labels' commit and verify": "before_verify:labels",
            "killed after labels' verify, before `verified`": "before_ledger:verified:labels",
            "killed mid-way through labels' `done`": "mid_done:labels",
            "killed after claims' commit, before `committed`": "before_ledger:committed:claims",
            "killed with two tables' commits in flight": "two_at_once",
            "killed once a table is verified while one before it commits": "verified_ahead",
        }
        for name, kill in kills.items():
            work = scenario(name, {"FAKE_RATE": "1e6"}, kill)
            if kill in ("two_at_once", "verified_ahead"):
                killed = [x for x in events(work) if x["event"] == "killed"]
                check(f"{name}: killed as meant ({killed[0]['table'] if killed else '-'} with "
                      f"{killed[0]['inflight'] if killed else '-'} in flight)", bool(killed))
            check(f"{name}: rerun to the same Hub files, commits, local copies and metadata", same(outcome(work), want))

        work = tmp / "commit fails"
        r = run(work, {"FAKE_RATE": "1e6"}, "commit_fails:descriptions")
        ev = events(work)
        failed = next(x["t"] for x in ev if x["event"] == "commit_start" and x["table"] == "descriptions")
        started = [x["table"] for x in ev if x["event"] == "start" and x["t"] > failed]
        check("a commit failing: finalise stops with an error, loudly",
              r.returncode != 0 and "COMMIT FAILED" in r.stdout and "CommitterFailed" in r.stderr)
        check("a commit failing: at most one table started after it (the one being written)", len(started) <= 1)
        check("a commit failing: finalise not marked done", not any(x["event"] in ("cards", "end") for x in ev))
        stages = [json.loads(line) for line in (work / "releases" / RELEASE / "state" / "sort.jsonl").read_text().splitlines()]
        check("a commit failing: the table left at `written`",
              [s["stage"] for s in stages if s["table"] == "descriptions"][-1] == "written")
        check("a commit failing: the rerun completes", run(work, {"FAKE_RATE": "1e6"}).returncode == 0)
        check("a commit failing: the same Hub files, commits, local copies and metadata", same(outcome(work), want))

        work = scenario("the writer failing (aliases)", {"FAKE_RATE": "1e6"}, "writer_fails:aliases")
        check("the writer failing: rerun to the same state", same(outcome(work), want))
    print("all passed")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--child"]:
        child(Path(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else "")
    else:
        main()
