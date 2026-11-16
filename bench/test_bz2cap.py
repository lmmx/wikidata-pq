"""bench/bz2cap.sh (`just bench-bz2-cap`) in a scratch repository, with stand-ins for uv,
bench/full/run.py and bench/bz2threads.py: it refuses a dirty tree and a mibz2 other than
the one it wants before making anything, rebuilds the extension, fetches the dump, runs
bz2threads.py's release-shaped comparison with the arms and rounds asked into one new
directory under target/bench-bz2-cap/, records the plan each arm gets, and bundles the
directory whatever the outcome. Nothing is measured.

    python3 bench/test_bz2cap.py
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

# uv's stand-in: each call logged; `uv run --frozen python ARGS` runs this Python on ARGS
FAKE_UV = f"""#!/usr/bin/env bash
echo "$*" >> "$FAKE_LOG.uv"
if [ "$1 $2 $3" = "run --frozen python" ]; then shift 3; exec {sys.executable} "$@"; fi
[ "$1 $2" = "sync --frozen" ] || exit 9
"""

# bz2threads.py's stand-in: its argv logged, its files written; FAKE_RUNS=parity makes one
# run's digest differ, crash makes it fail before any summary
FAKE_BZ2THREADS = r"""import json, os, sys
from pathlib import Path
args = sys.argv[1:]
open(os.environ["FAKE_LOG"], "a").write(json.dumps({"args": args}) + "\n")
mode = os.environ.get("FAKE_RUNS", "")
if mode == "crash":
    sys.exit(2)
out = Path(args[args.index("--out") + 1])
recs = [{"round": 0, "arm": a, "kind": "full", "ok": True, "same_as_first": not (mode == "parity" and a == "8")}
        for a in args[args.index("--threads") + 1].split(",")]
(out / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
(out / "summary.md").write_text("# summary\n")
sys.exit(1 if mode == "parity" else 0)
"""

FAKE_RUN = r"""import json, os, sys
open(os.environ["FAKE_LOG"] + ".fetch", "a").write(json.dumps(sys.argv[1:]) + "\n")
"""

LOCK = '[[package]]\nname = "memchr"\nversion = "2.7.4"\n\n[[package]]\nname = "mibz2"\nversion = "{}"\n'


def git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


class BenchBz2Cap(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.work = w = self.tmp / "work"
        git(self.tmp, "init", "-q", "-b", "master", str(w))
        git(w, "config", "user.name", "Test")
        git(w, "config", "user.email", "test@example.com")
        (w / "bench" / "full").mkdir(parents=True)
        (w / "reshape").mkdir()
        (w / "reshape" / "Cargo.lock").write_text(LOCK.format("0.5.1"))
        for name in ("bz2cap.sh", "bundle.sh"):
            shutil.copy(HERE / name, w / "bench" / name)
        (w / "bench" / "bz2threads.py").write_text(FAKE_BZ2THREADS)
        (w / "bench" / "full" / "run.py").write_text(FAKE_RUN)
        # dump.bz2_threads's stand-in: auto is 4
        (w / "src" / "wikidata").mkdir(parents=True)
        (w / "src" / "wikidata" / "__init__.py").write_text("")
        (w / "src" / "wikidata" / "dump.py").write_text("def bz2_threads():\n    return 4\n")
        (w / ".gitignore").write_text("target/\n")
        git(w, "add", "-A")
        git(w, "commit", "-qm", "base")
        self.head = git(w, "rev-parse", "HEAD")
        bin_ = self.tmp / "bin"
        bin_.mkdir()
        (bin_ / "uv").write_text(FAKE_UV)
        (bin_ / "uv").chmod(0o755)
        self.log = self.tmp / "log.jsonl"
        self.env = {**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}", "FAKE_LOG": str(self.log),
                    "PYTHONPATH": str(w / "src"),
                    "PYTHONDONTWRITEBYTECODE": "1"}
        self.env.pop("BZ2_CAP_MIBZ2", None)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_cap(self, *args, **env):
        return subprocess.run(["bash", "bench/bz2cap.sh", *args], cwd=self.work, env={**self.env, **env},
                              capture_output=True, text=True)

    def calls(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []

    def attempts(self):
        top = self.work / "target" / "bench-bz2-cap"
        return sorted(p for p in top.iterdir() if p.is_dir()) if top.exists() else []

    def member(self, attempt, name):
        with tarfile.open(attempt.parent / f"{attempt.name}.tar.gz") as t:
            return t.extractfile(f"{attempt.name}/{name}").read().decode()

    def test_the_default_arms_rounds_and_workload_in_one_new_directory(self):
        r = self.run_cap()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        [d] = self.attempts()
        [call] = self.calls()
        a = call["args"]
        for flag, value in (("--full", "--release 20261019 --windows 24 --window-mb 25"),
                            ("--threads", "auto,6,8,9,10,20"), ("--rounds", "8"), ("--out", str(d))):
            self.assertEqual(a[a.index(flag) + 1], value, flag)
        uv = (self.tmp / "log.jsonl.uv").read_text().splitlines()
        self.assertEqual(uv[0], "sync --frozen --reinstall-package wikidata-reshape")
        fetch = json.loads((self.tmp / "log.jsonl.fetch").read_text())
        self.assertEqual(fetch, ["fetch", "--release", "20261019", "--windows", "24", "--window-mb", "25"])
        self.assertEqual((d / "outcome.txt").read_text(), "COMPLETE\n")
        plans = (d / "plans.txt").read_text().splitlines()
        self.assertTrue(plans[0].startswith("# mibz2 0.5.1's Plan::for_threads"))
        self.assertEqual([p.split()[:2] for p in plans[1:]], [["auto", "4"], ["6", "6"], ["8", "8"], ["9", "9"], ["10", "10"], ["20", "20"]])
        self.assertEqual(plans[1].split(maxsplit=2)[2], "Workers(4)")  # (fewer than 6 cores' worth)
        meta = (d / "meta.txt").read_text().splitlines()
        self.assertEqual(meta[:2], [f"harness {self.head}", "mibz2_locked 0.5.1"])
        self.assertTrue(meta[2].startswith("note mibz2 0.5.1; arms auto,6,8,9,10,20, 8 rounds; plans (derived): auto=4 "))
        self.assertFalse(any(line.startswith("pr ") for line in meta))  # (posted on the queue issue)
        self.assertIn("lscpu:", meta)
        summary = (d / "summary.md").read_text()
        self.assertIn("## The plan each arm gets (derived, not logged)", summary)
        self.assertIn("- `auto` (4 threads): Workers(4)", summary)
        # the bundle: what `just submit`'s glob (target/*/*.tar.gz) finds, and only it
        self.assertEqual(sorted((self.work / "target").glob("*/*.tar.gz")), [d.parent / f"{d.name}.tar.gz"])
        self.assertEqual(self.member(d, "outcome.txt"), "COMPLETE\n")
        self.assertIn("bench-bz2-cap: COMPLETE", self.member(d, "console.log"))
        self.assertEqual(git(self.work, "status", "--porcelain"), "")  # (nothing in the tree)

    def test_arguments_given(self):
        r = self.run_cap("20261101", "auto,8", "4")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        a = self.calls()[0]["args"]
        self.assertEqual(a[a.index("--threads") + 1], "auto,8")
        self.assertEqual(a[a.index("--rounds") + 1], "4")
        self.assertEqual(a[a.index("--full") + 1], "--release 20261101 --windows 24 --window-mb 25")

    def test_a_new_directory_each_attempt(self):
        self.assertEqual(self.run_cap().returncode, 0)
        subprocess.run(["sleep", "1.1"])
        self.assertEqual(self.run_cap().returncode, 0)
        self.assertEqual(len(self.attempts()), 2)

    def test_refuses_another_mibz2_unless_overridden(self):
        (self.work / "reshape" / "Cargo.lock").write_text(LOCK.format("0.3.3"))
        git(self.work, "commit", "-qam", "0.3.3")
        r = self.run_cap()
        self.assertEqual(r.returncode, 1)
        self.assertIn("locks mibz2 0.3.3, not 0.5.1", r.stderr)
        self.assertEqual(self.attempts(), [])
        self.assertEqual(self.calls(), [])
        self.assertEqual(self.run_cap(BZ2_CAP_MIBZ2="any").returncode, 0)
        subprocess.run(["sleep", "1.1"])  # (a new stamp: an attempt's directory is never reused)
        self.assertEqual(self.run_cap(BZ2_CAP_MIBZ2="0.3.3").returncode, 0)

    def test_refuses_uncommitted_changes(self):
        (self.work / "reshape" / "Cargo.lock").write_text(LOCK.format("0.3.5"))
        r = self.run_cap()
        self.assertEqual(r.returncode, 1)
        self.assertIn("uncommitted changes", r.stderr)
        self.assertEqual(self.attempts(), [])

    def test_refuses_bad_arguments(self):
        for args in (("2026",), ("20261019", "auto,x"), ("20261019", "auto,8", "0")):
            r = self.run_cap(*args)
            self.assertEqual(r.returncode, 1, args)
        self.assertEqual(self.attempts(), [])

    def test_a_parity_failure_is_bundled_as_such(self):
        r = self.run_cap(FAKE_RUNS="parity")
        self.assertEqual(r.returncode, 1)
        [d] = self.attempts()
        self.assertEqual(self.member(d, "outcome.txt"), "PARITY FAILED\n")

    def test_a_crash_is_bundled_incomplete(self):
        r = self.run_cap(FAKE_RUNS="crash")
        self.assertEqual(r.returncode, 1)
        [d] = self.attempts()
        self.assertEqual(self.member(d, "outcome.txt"), "INCOMPLETE: runs\n")


if __name__ == "__main__":
    unittest.main()
