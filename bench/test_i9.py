"""bench/i9.sh (`just bench-i9`) in a scratch repository, with stand-ins for the harness
(ab.py, tiers.py, combine.py, dumpslice.py) and for uv: it runs the workflow's six round
jobs in order with the workflow's arguments, the extras after a job's core, combines them,
writes meta.txt and outcome.txt, bundles one new directory per attempt, reuses the arms'
checkouts, and refuses uncommitted or unpushed work. Nothing is measured.

    python3 bench/test_i9.py
"""

import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ab.py's stand-in: each call logged (its argv, ROUND, ORDER, the interpreter, the slice's
# path); FAKE_AB names a job (r<round>-<order>) whose core fails parity or crashes
FAKE_AB = r"""import json, os, sys
from pathlib import Path
args = sys.argv[1:]
out = Path(args[args.index("--out") + 1])
job = f"r{os.environ['ROUND']}-{os.environ['ORDER']}"
with open(os.environ["FAKE_LOG"], "a") as f:
    f.write(json.dumps({"job": job, "args": args, "python": sys.executable,
                        "slice": os.environ.get("BENCH_DUMP_SLICE"), "rounds": os.environ.get("ROUNDS")}) + "\n")
mode, _, which = os.environ.get("FAKE_AB", "").partition(":")
if "--resume" in args:
    r = json.loads((out / "results.json").read_text())
    r["resumed"] = True
    (out / "results.json").write_text(json.dumps(r))
    sys.exit(0)
if which == job and mode == "crash":
    sys.exit(2)
out.mkdir(parents=True, exist_ok=True)
ok = not (which == job and mode == "parity")
(out / "results.json").write_text(json.dumps({"job": job, "parity_ok": ok, "bz2native_parity_ok": True}))
sys.exit(0 if ok else 1)
"""

FAKE_TIERS = r"""import json, os, sys
args = sys.argv[1:]
assert args[2:6] == ["--suite", "pr", "--labels-json", "[]"], args
assert open(args[args.index("--body-file") + 1]).read() == ""
deferred = [s for s in os.environ.get("FAKE_DEFERRED", "").split(",") if s]
open(args[args.index("--out") + 1], "w").write(json.dumps({"suite": "pr", "inline": ["bz2native"], "deferred": deferred}))
"""

FAKE_COMBINE = r"""import sys
from pathlib import Path
assert sys.argv[1] == "3", sys.argv
Path(sys.argv[2]).write_text("combined " + " ".join(Path(p).parent.name for p in sys.argv[3:]) + "\n")
"""

# uv's stand-in: `uv sync --frozen` makes .venv/bin/python, this Python
FAKE_UV = f"""#!/usr/bin/env bash
[ "$1 $2" = "sync --frozen" ] || exit 9
mkdir -p .venv/bin && ln -sf {sys.executable} .venv/bin/python
"""


def git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


class BenchI9(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        remote = self.tmp / "origin.git"
        git(self.tmp, "init", "-q", "--bare", "-b", "master", str(remote))
        self.work = w = self.tmp / "work"
        git(self.tmp, "init", "-q", "-b", "master", str(w))
        git(w, "config", "user.name", "Test")
        git(w, "config", "user.email", "test@example.com")
        git(w, "remote", "add", "origin", str(remote))
        # the bench data, on an orphan branch
        git(w, "checkout", "-q", "--orphan", "bench-data")
        (w / "c0.parquet").write_text("rows")
        (w / "SHA256SUMS").write_text(subprocess.run(["sha256sum", "c0.parquet"], cwd=w, capture_output=True,
                                                     text=True, check=True).stdout)
        git(w, "add", "-A")
        git(w, "commit", "-qm", "data")
        self.data = git(w, "rev-parse", "HEAD")
        git(w, "push", "-q", "origin", "bench-data")
        git(w, "rm", "-rq", ".")
        # master: the harness's stand-ins, and the recipe's own scripts
        git(w, "checkout", "-q", "--orphan", "master")
        (w / ".github" / "workflows").mkdir(parents=True)
        (w / ".github" / "workflows" / "bench.yml").write_text(f"env:\n  DATA_COMMIT: {self.data} # v1\n")
        (w / "bench").mkdir()
        for name in ("i9.sh", "bundle.sh"):
            shutil.copy(HERE / name, w / "bench" / name)
        (w / "bench" / "ab.py").write_text(FAKE_AB)
        (w / "bench" / "tiers.py").write_text(FAKE_TIERS)
        (w / "bench" / "combine.py").write_text(FAKE_COMBINE)
        (w / "bench" / "dumpslice.py").write_text("# v1\n")
        (w / ".gitignore").write_text("target/\n")
        git(w, "add", "-A")
        git(w, "commit", "-qm", "base")
        self.base = git(w, "rev-parse", "HEAD")
        git(w, "push", "-q", "origin", "master")
        # the pull request's head, pushed
        git(w, "checkout", "-q", "-b", "feature")
        (w / "src.txt").write_text("change\n")
        git(w, "add", "-A")
        git(w, "commit", "-qm", "change")
        self.head = git(w, "rev-parse", "HEAD")
        git(w, "push", "-q", "origin", "feature")
        # uv and the decoders, stood in for
        bin_ = self.tmp / "bin"
        bin_.mkdir()
        (bin_ / "uv").write_text(FAKE_UV)
        for tool in ("lbzip2", "igzip", "pigz"):
            (bin_ / tool).write_text("#!/bin/sh\n")
        for f in bin_.iterdir():
            f.chmod(0o755)
        self.log = self.tmp / "ab.jsonl"
        self.env = {**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}", "FAKE_LOG": str(self.log),
                    "GH": str(self.tmp / "no-gh")}

    def run_i9(self, *args, **env):
        return subprocess.run(["bash", "bench/i9.sh", *args], cwd=self.work, env={**self.env, **env},
                              capture_output=True, text=True)

    def calls(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []

    def attempts(self):
        top = self.work / "target" / "bench-i9"
        return sorted(p for p in top.iterdir() if p.name[:2] == "20" and p.is_dir()) if top.exists() else []

    def member(self, attempt, name):
        with tarfile.open(attempt.parent / f"{attempt.name}.tar.gz") as t:
            return t.extractfile(f"{attempt.name}/{name}").read().decode()

    def test_the_six_jobs_in_the_workflows_order_then_combined(self):
        r = self.run_i9("219")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        [d] = self.attempts()
        jobs = [f"r{n}-{o}" for n in (1, 2, 3) for o in ("forward", "reverse")]
        calls = self.calls()
        self.assertEqual([c["job"] for c in calls], jobs)
        arms = self.work / "target" / "bench-i9" / "arms"
        for c, job in zip(calls, jobs):
            n, order = job[1], job.split("-")[1]
            a = c["args"]
            self.assertEqual(a[a.index("--arm") + 1], f"before={arms / self.base}")
            self.assertEqual(a[a.index("--arm", a.index("--arm") + 1) + 1], f"after={arms / self.head}")
            for flag, value in (("--rounds", "1"), ("--seed", n), ("--order", order), ("--tier", "quick"),
                                ("--out", str(d / job)), ("--suite", str(d / "suite.json")),
                                ("--data", str(self.work / "target" / "bench-i9" / "data"))):
                self.assertEqual(a[a.index(flag) + 1], value, flag)
            self.assertEqual(Path(c["python"]).parent, arms / self.head / ".venv" / "bin")
            self.assertEqual(c["slice"], str(self.work / "target" / "bench-i9" / "dump-slice.json.bz2"))
            self.assertEqual(c["rounds"], "3")
        self.assertEqual((d / "summary.md").read_text(), "combined " + " ".join(jobs) + "\n")
        self.assertEqual((d / "outcome.txt").read_text(), "COMPLETE\n")
        meta = (d / "meta.txt").read_text().splitlines()
        self.assertEqual(meta[:4], ["pr 219", f"before {self.base}", f"after {self.head}", f"data {self.data}"])
        self.assertIn("lscpu:", meta)
        self.assertTrue(any(line.startswith("nproc ") for line in meta))
        # the bundle: what `just submit`'s glob (target/*/*.tar.gz) finds, and only it
        bundles = sorted((self.work / "target").glob("*/*.tar.gz"))
        self.assertEqual(bundles, [d.parent / f"{d.name}.tar.gz"])
        self.assertEqual(self.member(d, "outcome.txt"), "COMPLETE\n")
        self.assertIn("bench-i9: COMPLETE", self.member(d, "console.log"))

    def test_the_extras_after_each_jobs_core(self):
        r = self.run_i9("219", FAKE_DEFERRED="misfit,evolve")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        calls = self.calls()
        self.assertEqual(len(calls), 12)
        for core, extras in zip(calls[::2], calls[1::2]):
            self.assertEqual(core["job"], extras["job"])
            self.assertNotIn("--resume", core["args"])
            self.assertIn("--resume", extras["args"])

    def test_a_parity_failure_runs_every_job_and_says_so(self):
        r = self.run_i9("219", FAKE_AB="parity:r2-reverse", FAKE_DEFERRED="misfit")
        self.assertNotEqual(r.returncode, 0)
        [d] = self.attempts()
        self.assertEqual((d / "outcome.txt").read_text(), "PARITY FAILED\n")
        # (every job ran; the failing job's extras did not)
        self.assertEqual(len(self.calls()), 11)
        self.assertTrue((d.parent / f"{d.name}.tar.gz").exists())

    def test_a_crashed_job_is_incomplete(self):
        r = self.run_i9("219", FAKE_AB="crash:r1-reverse")
        self.assertNotEqual(r.returncode, 0)
        [d] = self.attempts()
        self.assertEqual((d / "outcome.txt").read_text(), "INCOMPLETE: round r1-reverse\n")
        self.assertEqual((d / "summary.md").read_text().split()[1:], ["r1-forward", "r2-forward", "r2-reverse",
                                                                      "r3-forward", "r3-reverse"])

    def test_a_failed_setup_step_is_named_and_still_bundled(self):
        (self.tmp / "bin" / "uv").write_text("#!/bin/sh\nexit 1\n")
        r = self.run_i9("219")
        self.assertNotEqual(r.returncode, 0)
        [d] = self.attempts()
        self.assertEqual((d / "outcome.txt").read_text(), "INCOMPLETE: environments\n")
        self.assertTrue((d.parent / f"{d.name}.tar.gz").exists())

    def test_a_second_attempt_is_a_new_directory_and_reuses_the_arms(self):
        self.assertEqual(self.run_i9("219").returncode, 0)
        marker = self.work / "target" / "bench-i9" / "arms" / self.head / ".venv" / "kept"
        marker.write_text("")
        subprocess.run(["sleep", "1.1"])
        r = self.run_i9("219", self.base)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(len(self.attempts()), 2)
        self.assertTrue(marker.exists())

    def test_uncommitted_changes_are_refused(self):
        (self.work / "src.txt").write_text("edited\n")
        r = self.run_i9("219")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("uncommitted changes", r.stderr)
        self.assertEqual(self.attempts(), [])

    def test_an_unpushed_head_is_refused(self):
        (self.work / "src.txt").write_text("more\n")
        git(self.work, "commit", "-qam", "local only")
        r = self.run_i9("219")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("is not on origin", r.stderr)
        self.assertEqual(self.attempts(), [])
        self.assertEqual(self.calls(), [])

    def test_a_pr_that_is_not_a_number_is_refused(self):
        r = self.run_i9("abc")
        self.assertNotEqual(r.returncode, 0)
        self.assertEqual(self.attempts(), [])


if __name__ == "__main__":
    sys.exit(unittest.main())
