"""The benchmark's comment on a pull request: one per pull request, edited in place.

- `pending`: as the run starts. With no comment yet, one saying the result is pending.
  With one, its results stay and are marked as being recomputed: "(pending 🔁)" on the
  score, "(pending 🔁 <commit>)" beside the commit they are for.
- `final <summary.md>`: the run's results, in place of whatever the comment held.
- `failed`: the run ended without results; earlier results stay, marked as such.

Usage: comment.py pending|final|failed <pull request> <head commit> [summary.md]
Needs GH_TOKEN, GITHUB_REPOSITORY and RUN_URL.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

MARK = "<!-- bench -->"
SCORE = re.compile(r"^(\*\*after ran at .*?)(?: \((?:pending 🔁|failed ❌)[^)]*\))?$", re.M)
AFTER = re.compile(r"(\| after \(`[0-9a-f]+`\))(?: \((?:pending 🔁|failed ❌) `[0-9a-f]+`\))?")
STATUS = re.compile(r"^_Run: .*_$", re.M)


def gh(*args: str) -> str:
    return subprocess.run(["gh", "api", *args], capture_output=True, text=True, check=True).stdout


def main() -> None:
    mode, pr, head = sys.argv[1], sys.argv[2], sys.argv[3][:9]
    repo, run = os.environ["GITHUB_REPOSITORY"], os.environ["RUN_URL"]
    comments = json.loads(gh(f"repos/{repo}/issues/{pr}/comments", "--paginate", "--slurp"))
    existing = next((c for page in comments for c in page if c["body"].startswith(MARK)), None)
    status = f"_Run: [{mode if mode != 'final' else 'done'}]({run}) for `{head}`_"
    if mode == "final":
        body = f"{MARK}\n{Path(sys.argv[4]).read_text().rstrip()}\n\n{status}\n"
    elif existing is None:
        body = f"{MARK}\n## Benchmark: wikidata-pq pipeline\n\n**pending ⏳** for `{head}`\n\n{status}\n"
    else:
        tag = "pending 🔁" if mode == "pending" else "failed ❌"
        body = SCORE.sub(lambda m: f"{m.group(1)} ({tag})", existing["body"])
        body = AFTER.sub(lambda m: f"{m.group(1)} ({tag} `{head}`)", body)
        body = STATUS.sub(status, body) if STATUS.search(body) else f"{body.rstrip()}\n\n{status}\n"
    if existing:
        gh("-X", "PATCH", f"repos/{repo}/issues/comments/{existing['id']}", "-f", f"body={body}")
    else:
        gh(f"repos/{repo}/issues/{pr}/comments", "-f", f"body={body}")


if __name__ == "__main__":
    main()
