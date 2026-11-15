"""bench/i9-ladder.sh (`just bench-i9-ladder`) in a scratch repository, with stand-ins for
bench/i9.sh, bench/submit.sh and gh: it runs bench/i9.sh once a step, in the steps file's
order, with each step's pull request (or `-`), before and AFTER; submits each step's bundle
as it is made; records a failed, refused or unsubmitted step and runs the next; checks each
step's mibz2 versions before running it; and ends with the table of outcomes, bundled and
submitted itself. Also checks bench/i9-ladder.tsv: the pilot, then the ladder, each step's
before the last step's after. Nothing is measured.

    python3 bench/test_i9_ladder.py
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
LOCK = '[[package]]\nname = "mibz2"\nversion = "{}"\n'

# i9.sh's stand-in: each call logged (its arguments and I9_NOTE); makes one bundle in
# target/bench-i9/ whose outcome.txt is COMPLETE, or the outcome FAKE_I9_<after> names
# (`refuse`: no bundle, as i9.sh's refusals make none)
FAKE_I9 = r"""#!/usr/bin/env python3
import json, os, sys, tarfile
from pathlib import Path
pr, before, after = sys.argv[1:4]
with open(os.environ["FAKE_LOG"], "a") as f:
    f.write(json.dumps({"call": "i9", "args": sys.argv[1:], "note": os.environ.get("I9_NOTE")}) + "\n")
mode = os.environ.get(f"FAKE_I9_{after}", "COMPLETE")
if mode == "refuse":
    print("bench-i9: after (x) is not on origin", file=sys.stderr)
    sys.exit(1)
top = Path("target/bench-i9")
top.mkdir(parents=True, exist_ok=True)
stamp = f"2026111{len(list(top.glob('*.tar.gz')))}T000000Z"
d = top / stamp
d.mkdir()
(d / "outcome.txt").write_text(mode + "\n")
(d / "meta.txt").write_text(("" if pr == "-" else f"pr {pr}\n") + f"before {before}\nafter {after}\n")
with tarfile.open(top / f"{stamp}.tar.gz", "w:gz") as t:
    t.add(d, arcname=stamp)
print(f"bench-i9: {mode}")
sys.exit(0 if mode == "COMPLETE" else 1)
"""

# submit.sh's stand-in: logs the bundle; fails for a bundle whose after is FAKE_SUBMIT_FAIL
FAKE_SUBMIT = r"""#!/usr/bin/env python3
import json, os, sys, tarfile
from pathlib import Path
tarball = Path(sys.argv[1])
stamp = tarball.name.removesuffix(".tar.gz")
with tarfile.open(tarball) as t:
    meta = t.extractfile(f"{stamp}/meta.txt").read().decode()
    summary = t.extractfile(f"{stamp}/summary.md").read().decode() if f"{stamp}/summary.md" in t.getnames() else None
fields = dict(line.split(" ", 1) for line in meta.splitlines() if " " in line)
with open(os.environ["FAKE_LOG"], "a") as f:
    f.write(json.dumps({"call": "submit", "bundle": str(tarball), "summary": summary}) + "\n")
if fields.get("after") and fields["after"] == os.environ.get("FAKE_SUBMIT_FAIL"):
    print("submit: couldn't push to evidence", file=sys.stderr)
    sys.exit(1)
issue = fields.get("pr", "97")
print(f"submit: archived {tarball.parent.name}/{tarball.name} at {'c' * 40}")
print(f"submit: posted https://github.com/lmmx/wikidata-pq/issues/{issue}#issuecomment-1")
"""


def git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def steps(path=HERE / "i9-ladder.tsv"):
    rows = [line.split() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]
    return [dict(zip(("name", "pr", "before", "after", "vb", "va"), row)) for row in rows]


class TheSteps(unittest.TestCase):
    """bench/i9-ladder.tsv: what the ladder runs, and in what order."""

    def test_the_pilot_then_each_bump_against_the_last(self):
        s = steps()
        self.assertEqual([r["name"] for r in s], ["pilot", "0.2.1", "0.2.2", "0.2.5", "0.2.8", "0.2.9", "0.3.0", "0.3.1"])
        for r in s:
            self.assertEqual(len(r), 6, r)
            self.assertRegex(r["before"], r"^[0-9a-f]{40}$")
            self.assertRegex(r["after"], r"^[0-9a-f]{40}$")
        pilot, *ladder = s
        self.assertEqual((pilot["pr"], pilot["before"], pilot["after"], pilot["vb"], pilot["va"]),
                         ("227", "cd305f33c29fae794a9cfd81eee48b5409fa430a",
                          "a3047bfeccdf9a4e49cfd8578bc88b3cf716eb39", "0.3.1", "0.3.3"))
        # the ladder: on master (no pull request), from a1f2a958 (the cap-4 native decoder,
        # mibz2 0.1.4), each step's before the last step's after, its versions likewise
        self.assertEqual(ladder[0]["before"], "a1f2a958d2948d61c00b6bb5d8ad8593eafa20c4")
        self.assertEqual(ladder[0]["vb"], "0.1.4")
        for prev, r in zip(ladder, ladder[1:]):
            self.assertEqual(r["before"], prev["after"])
            self.assertEqual(r["vb"], prev["va"])
        for r in ladder:
            self.assertEqual(r["pr"], "-")
            self.assertEqual(r["va"], r["name"])
        self.assertEqual(ladder[-1]["after"], pilot["before"])

    def test_the_commits_lock_the_versions_the_file_says(self):
        """Where this clone has the commits: each locks its mibz2, and before precedes after."""
        root = HERE.parent
        for r in steps():
            for sha, version in ((r["before"], r["vb"]), (r["after"], r["va"])):
                if subprocess.run(["git", "cat-file", "-e", f"{sha}^{{commit}}"], cwd=root,
                                  capture_output=True).returncode:
                    self.skipTest(f"{sha} is not in this clone")
                lock = git(root, "show", f"{sha}:reshape/Cargo.lock")
                self.assertIn(f'name = "mibz2"\nversion = "{version}"', lock, (r["name"], sha))
            self.assertEqual(subprocess.run(["git", "merge-base", "--is-ancestor", r["before"], r["after"]],
                                            cwd=root).returncode, 0, r["name"])


class Ladder(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        remote = self.tmp / "origin.git"
        git(self.tmp, "init", "-q", "--bare", "-b", "master", str(remote))
        self.work = w = self.tmp / "work"
        git(self.tmp, "init", "-q", "-b", "master", str(w))
        git(w, "config", "user.name", "Test")
        git(w, "config", "user.email", "test@example.com")
        git(w, "remote", "add", "origin", str(remote))
        (w / "bench").mkdir()
        (w / "reshape").mkdir()
        for name in ("i9-ladder.sh", "bundle.sh"):
            shutil.copy(HERE / name, w / "bench" / name)
        (w / ".gitignore").write_text("target/\n")
        # four commits, each a mibz2 bump
        self.shas = []
        for version in ("0.1.4", "0.2.1", "0.2.2", "0.2.5"):
            (w / "reshape" / "Cargo.lock").write_text(LOCK.format(version))
            git(w, "add", "-A")
            git(w, "commit", "-qm", f"mibz2 {version}")
            self.shas.append(git(w, "rev-parse", "HEAD"))
        git(w, "push", "-q", "origin", "master")
        a, b, c, d = self.shas
        self.steps = self.tmp / "steps.tsv"
        self.steps.write_text(
            "# name pr before after vb va\n"
            f"pilot 227 {c} {d} 0.2.2 0.2.5\n"
            f"0.2.1 - {a} {b} 0.1.4 0.2.1\n"
            f"0.2.2 - {b} {c} 0.2.1 0.2.2\n"
            f"0.2.5 - {c} {d} 0.2.2 0.2.5\n"
        )
        bin_ = self.tmp / "bin"
        bin_.mkdir()
        for name, text in (("i9", FAKE_I9), ("submit", FAKE_SUBMIT), ("gh", "#!/bin/sh\nexit 0\n")):
            (bin_ / name).write_text(text)
            (bin_ / name).chmod(0o755)
        self.log = self.tmp / "log.jsonl"
        # (the stand-ins are scripts; the ladder runs them with bash, so each is wrapped)
        for name in ("i9", "submit"):
            (bin_ / f"{name}.sh").write_text(f'exec "{bin_ / name}" "$@"\n')
        self.env = {**os.environ, "FAKE_LOG": str(self.log), "LADDER_STEPS": str(self.steps),
                    "I9": str(bin_ / "i9.sh"), "SUBMIT": str(bin_ / "submit.sh"), "GH": str(bin_ / "gh")}

    def ladder(self, *args, **env):
        return subprocess.run(["bash", "bench/i9-ladder.sh", *args], cwd=self.work, env={**self.env, **env},
                              capture_output=True, text=True)

    def calls(self, kind=None):
        calls = [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []
        return [c for c in calls if kind is None or c["call"] == kind]

    def attempt(self):
        [d] = [p for p in (self.work / "target" / "bench-i9-ladder").iterdir() if p.is_dir()]
        return d

    def table(self, d):
        rows = [line.split("\t") for line in (d / "steps.tsv").read_text().splitlines()]
        return rows[1:]

    def test_every_step_in_order_each_submitted_then_the_summary(self):
        r = self.ladder()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        a, b, c, d = self.shas
        i9 = self.calls("i9")
        self.assertEqual([x["args"] for x in i9], [["227", c, d], ["-", a, b], ["-", b, c], ["-", c, d]])
        self.assertIn("step 1 of 4 (pilot)", i9[0]["note"])
        self.assertIn("step 4 of 4 (0.2.5)", i9[3]["note"])
        # each step's bundle submitted straight after its run, then the ladder's own
        kinds = [x["call"] for x in self.calls()]
        self.assertEqual(kinds, ["i9", "submit"] * 4 + ["submit"])
        submits = self.calls("submit")
        self.assertTrue(all("/target/bench-i9/" in s["bundle"] for s in submits[:4]))
        self.assertIn("/target/bench-i9-ladder/", submits[4]["bundle"])
        dd = self.attempt()
        rows = self.table(dd)
        self.assertEqual([row[0] for row in rows], ["pilot", "0.2.1", "0.2.2", "0.2.5"])
        self.assertEqual([row[1] for row in rows], ["0.2.2 → 0.2.5", "0.1.4 → 0.2.1", "0.2.1 → 0.2.2", "0.2.2 → 0.2.5"])
        self.assertEqual({row[4] for row in rows}, {"COMPLETE"})
        self.assertEqual(rows[0][5], "yes: https://github.com/lmmx/wikidata-pq/issues/227#issuecomment-1")
        self.assertEqual(rows[1][5], "yes: https://github.com/lmmx/wikidata-pq/issues/97#issuecomment-1")
        self.assertEqual((dd / "outcome.txt").read_text(), "COMPLETE\n")
        summary = (dd / "summary.md").read_text()
        self.assertIn("| Step | mibz2 | Before | After | Outcome | Submitted |", summary)
        self.assertIn(f"| 0.2.1 | 0.1.4 → 0.2.1 | `{a[:8]}` | `{b[:8]}` | COMPLETE | yes: ", summary)
        self.assertEqual(submits[4]["summary"], summary)
        self.assertIn(summary, r.stdout)
        # the attempt's bundle: its console log, each step's log, and no pull request named
        with tarfile.open(dd.parent / f"{dd.name}.tar.gz") as t:
            names = t.getnames()
            meta = t.extractfile(f"{dd.name}/meta.txt").read().decode()
            console = t.extractfile(f"{dd.name}/console.log").read().decode()
        self.assertIn(f"{dd.name}/steps/0.2.2.log", names)
        self.assertNotIn("\npr ", "\n" + meta)
        self.assertIn("bench-i9-ladder: COMPLETE", console)

    def test_a_failed_step_is_recorded_and_the_next_runs(self):
        a, b, c, d = self.shas
        r = self.ladder(**{f"FAKE_I9_{b}": "PARITY FAILED", f"FAKE_I9_{c}": "refuse"}, FAKE_SUBMIT_FAIL=d)
        self.assertNotEqual(r.returncode, 0)
        # every step ran, in order
        self.assertEqual([x["args"][2] for x in self.calls("i9")], [d, b, c, d])
        dd = self.attempt()
        rows = {row[0]: row for row in self.table(dd)}
        self.assertEqual(rows["0.2.1"][4:], ["PARITY FAILED", "yes: https://github.com/lmmx/wikidata-pq/issues/97#issuecomment-1"])
        self.assertTrue(rows["0.2.2"][4].startswith("REFUSED (exit 1, 0 bundles): bench-i9: after (x) is not on origin"))
        self.assertEqual(rows["0.2.2"][5], "-")
        for name in ("pilot", "0.2.5"):
            self.assertEqual(rows[name][4], "COMPLETE")
            self.assertTrue(rows[name][5].startswith("NO: retry with just submit-bundle "), rows[name])
        self.assertEqual((dd / "outcome.txt").read_text(), "INCOMPLETE: pilot 0.2.1 0.2.2 0.2.5\n")
        # the summary is still made and submitted
        self.assertEqual(self.calls()[-1]["call"], "submit")
        self.assertIn("| 0.2.1 | 0.1.4 → 0.2.1 |", self.calls()[-1]["summary"])

    def test_a_step_whose_commits_lock_other_versions_is_refused_and_the_rest_run(self):
        a, b, c, d = self.shas
        self.steps.write_text(self.steps.read_text().replace(f"{a} {b} 0.1.4 0.2.1", f"{a} {b} 0.1.4 0.2.0"))
        r = self.ladder()
        self.assertNotEqual(r.returncode, 0)
        self.assertEqual([x["args"][2] for x in self.calls("i9")], [d, c, d])
        rows = {row[0]: row for row in self.table(self.attempt())}
        self.assertEqual(rows["0.2.1"][4], "REFUSED: the commits lock mibz2 0.1.4 → 0.2.1")

    def test_steps_by_name(self):
        a, b, c, d = self.shas
        r = self.ladder("0.2.5", "0.2.1")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        # (in the file's order)
        self.assertEqual([x["args"] for x in self.calls("i9")], [["-", a, b], ["-", c, d]])
        self.assertIn("step 2 of 2 (0.2.5)", self.calls("i9")[1]["note"])
        r = self.ladder("0.9.9")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("no step 0.9.9", r.stderr)

    def test_refused_before_anything_runs(self):
        (self.tmp / "bin" / "gh").write_text("#!/bin/sh\nexit 1\n")
        r = self.ladder()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("gh isn't signed in", r.stderr)
        (self.tmp / "bin" / "gh").write_text("#!/bin/sh\nexit 0\n")
        (self.work / "reshape" / "Cargo.lock").write_text("edited\n")
        r = self.ladder()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("uncommitted changes", r.stderr)
        self.assertEqual(self.calls(), [])
        self.assertFalse((self.work / "target" / "bench-i9-ladder").exists())


if __name__ == "__main__":
    sys.exit(unittest.main())
