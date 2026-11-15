"""bench/submit.sh against a scratch repository and a stand-in for gh: it archives a bundle
once, posts once, on the pull request the bundle's meta.txt names (else on the queue issue,
#97), refuses a different bundle under the same run, and keeps the bundle when a push
fails, so the same command retries it.

    python3 bench/test_submit.py
"""

import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

SUBMIT = Path(__file__).resolve().parent / "submit.sh"

# gh's stand-in: `auth status` passes; `api` lists or appends to comments.json, each
# comment kept with the issue (or pull request) number it was posted on
FAKE_GH = r"""#!/usr/bin/env python3
import json, os, re, sys
store = os.environ["FAKE_COMMENTS"]
args = sys.argv[1:]
if args[:2] == ["auth", "status"]:
    sys.exit(0)
assert args[0] == "api", args
comments = json.load(open(store)) if os.path.exists(store) else []
endpoint = next(a for a in args[1:] if a.startswith("repos/"))
number = int(re.search(r"/issues/(\d+)/comments", endpoint).group(1))
if "--paginate" in args:
    jq = args[args.index("--jq") + 1]
    marker = re.search(r'contains\("(.*)"\)', jq).group(1)
    for c in comments:
        if c["issue"] == number and marker in c["body"]:
            print(c["html_url"])
    sys.exit(0)
body = open(args[args.index("-F") + 1].split("=@", 1)[1]).read()
url = f"https://github.com/lmmx/wikidata-pq/pull/{number}#issuecomment-{len(comments) + 1}"
comments.append({"issue": number, "body": body, "html_url": url})
json.dump(comments, open(store, "w"))
print(url)
"""

BEFORE, AFTER = "b" * 40, "a" * 40


def git(cwd, *args):
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


class Submit(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.remote = self.tmp / "lmmx" / "wikidata-pq.git"
        self.remote.parent.mkdir()
        git(self.tmp, "init", "-q", "--bare", str(self.remote))
        self.work = self.tmp / "work"
        self.work.mkdir()
        git(self.work, "init", "-q")
        git(self.work, "config", "user.name", "Test")
        git(self.work, "config", "user.email", "test@example.com")
        git(self.work, "remote", "add", "origin", str(self.remote))
        gh = self.tmp / "gh"
        gh.write_text(FAKE_GH)
        gh.chmod(0o755)
        self.env = {**os.environ, "GH": str(gh), "FAKE_COMMENTS": str(self.tmp / "comments.json")}

    def bundle(self, outcome, stamp="20261107T120000Z", pr="pr 219"):
        run = self.work / "target" / "bench-i9" / stamp
        run.mkdir(parents=True, exist_ok=True)
        (run / "outcome.txt").write_text(outcome + "\n")
        meta = f"before {BEFORE}\nafter {AFTER}\nnproc 20\nlscpu:\nModel name:   Intel(R) Core(TM) i9-10900K CPU @ 3.70GHz\n"
        (run / "meta.txt").write_text((pr + "\n" if pr else "") + meta)
        (run / "summary.md").write_text("**Speed:** 1.00× (0.99–1.01)\n")
        out = run.parent / f"{stamp}.tar.gz"
        with tarfile.open(out, "w:gz") as t:
            t.add(run, arcname=stamp)
        return out

    def submit(self, tarball):
        return subprocess.run(
            ["bash", str(SUBMIT), str(tarball)],
            cwd=self.work,
            env=self.env,
            capture_output=True,
            text=True,
        )

    def comments(self):
        return json.loads(Path(self.env["FAKE_COMMENTS"]).read_text())

    def test_archives_and_posts_once_on_the_pull_request(self):
        tarball = self.bundle("COMPLETE")
        first = self.submit(tarball)
        self.assertEqual(first.returncode, 0, first.stderr)
        tree = git(self.remote, "ls-tree", "-r", "--name-only", "evidence")
        self.assertEqual(
            tree.split(),
            [
                "bench-i9/20261107T120000Z.tar.gz",
                "bench-i9/20261107T120000Z.tar.gz.sha256",
            ],
        )
        stored = subprocess.run(
            ["git", "show", "evidence:bench-i9/20261107T120000Z.tar.gz"],
            cwd=self.remote,
            capture_output=True,
            check=True,
        ).stdout
        self.assertEqual(stored, tarball.read_bytes())
        # the evidence branch is an orphan: its one commit has no parent
        self.assertEqual(len(git(self.remote, "rev-list", "--parents", "-1", "evidence").split()), 1)
        [c] = self.comments()
        self.assertEqual(c["issue"], 219)
        for line in (
            "FROM: louis via workstation",
            "TO: coordinator",
            f"RE: bench-i9/20261107T120000Z @ {AFTER}",
            "TYPE: LOG",
            "**Outcome:** COMPLETE",
            f"before `{BEFORE}`, after `{AFTER}`",
            "pr 219",
            "cpu Intel(R) Core(TM) i9-10900K CPU @ 3.70GHz",
            "**Speed:** 1.00×",
        ):
            self.assertIn(line, c["body"])
        self.assertTrue(re.search(r"<!-- evidence: bench-i9/20261107T120000Z sha256:[0-9a-f]{64} -->", c["body"]))
        # again: nothing new committed or posted
        second = self.submit(tarball)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertIn("already archived", second.stdout)
        self.assertIn("already posted", second.stdout)
        self.assertEqual(len(git(self.remote, "rev-list", "evidence").split()), 1)
        self.assertEqual(len(self.comments()), 1)

    def test_a_bundle_without_a_pull_request_goes_to_the_queue_issue(self):
        for i, pr in enumerate(["", "pr none", "pr 0"]):
            r = self.submit(self.bundle("COMPLETE", f"20261107T12000{i}Z", pr=pr))
            self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual([c["issue"] for c in self.comments()], [97, 97, 97])
        self.assertIn("names no pull request", self.comments()[0]["body"])

    def test_a_second_run_is_added_beside_the_first(self):
        self.assertEqual(self.submit(self.bundle("COMPLETE")).returncode, 0)
        self.assertEqual(self.submit(self.bundle("PARITY FAILED", "20261107T130000Z")).returncode, 0)
        self.assertEqual(
            len(git(self.remote, "ls-tree", "-r", "--name-only", "evidence").split()), 4
        )
        self.assertEqual(len(git(self.remote, "rev-list", "evidence").split()), 2)
        self.assertEqual(len(self.comments()), 2)

    def test_a_different_bundle_under_the_same_run_is_refused(self):
        self.assertEqual(self.submit(self.bundle("COMPLETE")).returncode, 0)
        other = self.submit(self.bundle("INCOMPLETE: round r2-reverse"))
        self.assertNotEqual(other.returncode, 0)
        self.assertIn("different contents", other.stderr)
        self.assertEqual(len(self.comments()), 1)

    def test_a_failed_push_keeps_the_bundle_and_a_retry_submits_it(self):
        tarball = self.bundle("COMPLETE")
        moved = self.remote.with_suffix(".away")
        self.remote.rename(moved)
        failed = self.submit(tarball)
        self.assertNotEqual(failed.returncode, 0)
        self.assertTrue(tarball.exists())
        self.assertFalse(Path(self.env["FAKE_COMMENTS"]).exists())
        moved.rename(self.remote)
        self.assertEqual(self.submit(tarball).returncode, 0)
        self.assertEqual(len(self.comments()), 1)

    def test_a_remote_for_another_repository_is_refused(self):
        git(self.work, "remote", "set-url", "origin", str(self.tmp / "other.git"))
        r = self.submit(self.bundle("COMPLETE"))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("not lmmx/wikidata-pq", r.stderr)


if __name__ == "__main__":
    sys.exit(unittest.main())
