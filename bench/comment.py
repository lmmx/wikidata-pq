"""The benchmark's comment on a pull request: one per pull request, edited in place.

The rounds run in parallel jobs. The comment holds the rounds in so far (as hidden data),
so each job adds its own and renders them all (combine.py): "pending 🔁 1/3", "2/3", then
the full result.

- `pending [<store> <commit before>]`: as a job starts. A comment with rounds of an
  earlier commit keeps them, marked as being recomputed: "(pending 🔁)" on the score,
  "(pending 🔁 <commit>)" beside the commit they are for. With no rounds yet, the table
  with the commit before's figures from its own run in the results store, the commit
  after pending.
- `round <results.json> <total>`: this job's round, added to the rounds of this commit.
- `final <total> <results.json>...`: every round, from the run's artifacts.
- `failed`: the run ended without results; what the comment holds stays, marked so.

Usage: comment.py <mode> <pull request> <head commit> [args]
Needs GH_TOKEN, GITHUB_REPOSITORY and RUN_URL.
"""

import base64
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

from combine import render, render_pending, rounds_of, stored_rounds
from version import compatible

MARK = "<!-- bench -->"
DATA = re.compile(r"<!-- bench-data:([A-Za-z0-9+/=]*) -->")
SCORE = re.compile(r"^(\*\*after ran at .*?)(?: \((?:pending 🔁|failed ❌)[^)]*\))?$", re.M)
AFTER = re.compile(r"(\| after \(`[0-9a-f]+`\))(?: \((?:pending 🔁|failed ❌) `[0-9a-f]+`\))?")
STATUS = re.compile(r"^_Run: .*_$", re.M)


def gh(*args: str) -> str:
    return subprocess.run(["gh", "api", *args], capture_output=True, text=True, check=True).stdout


def find(repo: str, pr: str) -> dict | None:
    pages = json.loads(gh(f"repos/{repo}/issues/{pr}/comments", "--paginate", "--slurp"))
    return next((c for page in pages for c in page if c["body"].startswith(MARK)), None)


def data(comment: dict | None) -> dict:
    m = DATA.search(comment["body"]) if comment else None
    return json.loads(base64.b64decode(m.group(1))) if m else {"head": None, "rounds": {}}


def body(blob: dict, text: str, status: str) -> str:
    encoded = base64.b64encode(json.dumps(blob).encode()).decode()
    return f"{MARK}\n<!-- bench-data:{encoded} -->\n{text.rstrip()}\n\n{status}\n"


def write(repo: str, pr: str, comment: dict | None, text: str) -> None:
    if comment:
        gh("-X", "PATCH", f"repos/{repo}/issues/comments/{comment['id']}", "-f", f"body={text}")
    else:
        gh(f"repos/{repo}/issues/{pr}/comments", "-f", f"body={text}")


def main() -> None:
    mode, pr, head = sys.argv[1], sys.argv[2], sys.argv[3]
    repo, run = os.environ["GITHUB_REPOSITORY"], os.environ["RUN_URL"]
    short = head[:9]
    comment = find(repo, pr)
    blob = data(comment)

    if mode == "round":
        results, total = Path(sys.argv[4]), int(sys.argv[5])
        mine = rounds_of(json.loads(results.read_text()))[0]
        key = os.environ.get("ROUND", "1")
        for _ in range(8):  # jobs finishing together: add this round until it holds
            blob = data(comment)
            if blob["head"] != head:
                blob = {"head": head, "rounds": {}}
            blob["rounds"][key] = mine
            rounds = [blob["rounds"][k] for k in sorted(blob["rounds"], key=int)]
            n = len(rounds)
            status = f"_Run: [{'done' if n >= total else f'{n} of {total} rounds in'}]({run}) for `{short}`_"
            write(repo, pr, comment, body(blob, render(rounds, total), status))
            time.sleep(random.uniform(1, 3))
            comment = find(repo, pr)
            if key in data(comment)["rounds"] and data(comment)["head"] == head:
                return
        sys.exit("the comment kept losing this round")

    if mode == "final":
        total = int(sys.argv[4])
        rounds = [r for p in sys.argv[5:] for r in rounds_of(json.loads(Path(p).read_text()))]
        blob = {"head": head, "rounds": {str(i + 1): r for i, r in enumerate(rounds)}}
        missing = total - len(rounds)
        status = f"_Run: [done]({run}) for `{short}`" + (f", {missing} of {total} rounds failed" if missing > 0 else "") + "_"
        write(repo, pr, comment, body(blob, render(rounds, len(rounds)), status))
        return

    # pending or failed: what the comment holds stays, marked
    if mode == "pending" and blob["head"] == head:
        return  # another of this commit's jobs got here first
    status = f"_Run: [{mode}]({run}) for `{short}`_"
    if comment is None or (mode == "pending" and not blob["rounds"]):
        stored = Path(sys.argv[4]) / "runs" / "quick" / f"{sys.argv[5]}.json" if len(sys.argv) > 5 else None
        base = stored_rounds(stored) if stored and stored.exists() else []
        if base and all(compatible(r["bench"]) for r in base):  # only a comparable harness's figures
            text = render_pending(base, int(os.environ.get("ROUNDS", "3")), head)
        else:
            text = f"## Benchmark: wikidata-pq pipeline\n\n**pending ⏳** for `{short}`"
        text = body(blob, text, status)
    else:
        tag = "pending 🔁" if mode == "pending" else "failed ❌"
        text = SCORE.sub(lambda m: f"{m.group(1)} ({tag})", comment["body"])
        text = AFTER.sub(lambda m: f"{m.group(1)} ({tag} `{short}`)", text)
        text = STATUS.sub(status, text) if STATUS.search(text) else f"{text.rstrip()}\n\n{status}\n"
    write(repo, pr, comment, text)


if __name__ == "__main__":
    main()
