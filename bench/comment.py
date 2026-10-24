"""The benchmark's comment on a pull request: one per pull request, edited in place.

The rounds run in parallel jobs. The comment holds the rounds in so far (as hidden data),
so each job adds its own and renders them all (combine.py): "pending 🔁 1/3", "2/3", then
the full result.

- `pending [<store> <commit before>]`: as a job starts. A comment with rounds of an
  earlier commit keeps them, marked as being recomputed: "(pending 🔁)" on the score,
  "(pending 🔁 <commit>)" beside the commit they are for. With no rounds yet, the table
  with the commit before's figures from its own run in the results store, the commit
  after pending.
- `progress <done> <of> <total>`: this job's pipeline runs done and planned (ab.py's
  --on-run), shown with every job's as a bar on the status line.
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
STATUS = re.compile(r"^(?:\S+ )*_Run: .*_$", re.M)


def gh(*args: str) -> str:
    return subprocess.run(["gh", "api", *args], capture_output=True, text=True, check=True).stdout


def find(repo: str, pr: str) -> dict | None:
    pages = json.loads(gh(f"repos/{repo}/issues/{pr}/comments", "--paginate", "--slurp"))
    return next((c for page in pages for c in page if c["body"].startswith(MARK)), None)


def data(comment: dict | None) -> dict:
    m = DATA.search(comment["body"]) if comment else None
    return json.loads(base64.b64decode(m.group(1))) if m else {"head": None, "rounds": {}}


def status(blob: dict, state: str, run: str, head: str, total: int) -> str:
    """The status line: the run, and a bar of one cell per pipeline run planned over every
    round's job (a job not heard from yet plans 2: before and after). Done runs are blue
    while the run goes on, every cell green with a check once it is done."""
    prog = blob.get("progress", {})
    ours = prog.get("head") == head
    jobs = [prog.get("jobs", {}).get(str(k), [0, 2]) if ours else [0, 2] for k in range(1, total + 1)]
    of = sum(o for _, o in jobs)
    link = f"[{state}]({run}) for `{head[:9]}`"
    if state == "done":
        return f"🟩{'🟩' * (of - 1)} ✅ _Run: {link}_"
    if state != "running":
        return f"❌ _Run: {link}_"
    done = sum(d for d, _ in jobs)
    rounds = sum(1 for k in range(1, total + 1) if str(k) in blob["rounds"] and blob["head"] == head)
    return f"{'🟦' * done}{'⬜' * (of - done)} _Run: {link}: {done}/{of} pipeline runs, {rounds}/{total} rounds in_"


def advance(blob: dict, head: str, key: str, done: int, of: int) -> None:
    if blob.get("progress", {}).get("head") != head:
        blob["progress"] = {"head": head, "jobs": {}}
    blob["progress"]["jobs"][key] = [done, of]


def encode(blob: dict) -> str:
    return f"<!-- bench-data:{base64.b64encode(json.dumps(blob).encode()).decode()} -->"


def body(blob: dict, text: str, status: str) -> str:
    return f"{MARK}\n{encode(blob)}\n{text.rstrip()}\n\n{status}\n"


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
                blob = {"head": head, "rounds": {}, "progress": blob.get("progress", {})}
            blob["rounds"][key] = mine
            of = blob.get("progress", {}).get("jobs", {}).get(key, [0, 2])[1]
            advance(blob, head, key, of, of)  # this job's runs all done
            rounds = [blob["rounds"][k] for k in sorted(blob["rounds"], key=int)]
            state = "done" if len(rounds) >= total else "running"
            write(repo, pr, comment, body(blob, render(rounds, total), status(blob, state, run, head, total)))
            time.sleep(random.uniform(1, 3))
            comment = find(repo, pr)
            if key in data(comment)["rounds"] and data(comment)["head"] == head:
                return
        sys.exit("the comment kept losing this round")

    if mode == "progress":  # best effort: a later write carries a lost one's
        done, of, total = map(int, sys.argv[4:7])
        if comment is None:
            return
        advance(blob, head, os.environ.get("ROUND", "1"), done, of)
        text = comment["body"]
        text = DATA.sub(lambda _: encode(blob), text)
        line = status(blob, "running", run, head, total)
        text = STATUS.sub(lambda _: line, text) if STATUS.search(text) else f"{text.rstrip()}\n\n{line}\n"
        write(repo, pr, comment, text)
        return

    if mode == "final":
        total = int(sys.argv[4])
        rounds = [r for p in sys.argv[5:] for r in rounds_of(json.loads(Path(p).read_text()))]
        blob = {"head": head, "rounds": {str(i + 1): r for i, r in enumerate(rounds)}, "progress": blob.get("progress", {})}
        missing = total - len(rounds)
        line = status(blob, "done", run, head, total)
        if missing > 0:
            line = line[:-1] + f", {missing} of {total} rounds failed_"
        write(repo, pr, comment, body(blob, render(rounds, len(rounds)), line))
        return

    # pending or failed: what the comment holds stays, marked
    if mode == "pending" and blob["head"] == head:
        return  # another of this commit's jobs got here first
    if mode == "pending" and blob.get("progress", {}).get("head") != head:
        blob["progress"] = {"head": head, "jobs": {}}  # an empty bar
    line = status(blob, "running" if mode == "pending" else mode, run, head, int(os.environ.get("ROUNDS", "3")))
    if comment is None or (mode == "pending" and not blob["rounds"]):
        stored = Path(sys.argv[4]) / "runs" / "quick" / f"{sys.argv[5]}.json" if len(sys.argv) > 5 else None
        base = stored_rounds(stored) if stored and stored.exists() else []
        if base and all(compatible(r["bench"]) for r in base):  # only a comparable harness's figures
            text = render_pending(base, int(os.environ.get("ROUNDS", "3")), head)
        else:
            text = f"## Benchmark: wikidata-pq pipeline\n\n**pending ⏳** for `{short}`"
        text = body(blob, text, line)
    else:
        tag = "pending 🔁" if mode == "pending" else "failed ❌"
        text = SCORE.sub(lambda m: f"{m.group(1)} ({tag})", comment["body"])
        text = AFTER.sub(lambda m: f"{m.group(1)} ({tag} `{short}`)", text)
        text = DATA.sub(lambda _: encode(blob), text) if DATA.search(text) else text
        text = STATUS.sub(lambda _: line, text) if STATUS.search(text) else f"{text.rstrip()}\n\n{line}\n"
    write(repo, pr, comment, text)


if __name__ == "__main__":
    main()
