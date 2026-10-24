"""The benchmark's comment on a pull request: one per pull request, edited in place.

The rounds run in parallel jobs. The comment holds the rounds in so far (as hidden data),
so each job adds its own and renders them all (combine.py): "pending 🔁 1/3", "2/3", then
the full result.

- `pending [<store> <commit before>]`: as a job starts. A comment with rounds of an
  earlier commit keeps them, marked as being recomputed: "(pending 🔁)" on the score,
  "(pending 🔁 <commit>)" beside the commit they are for. With no rounds yet, the table
  with the commit before's figures from its own run in the results store, the commit
  after pending.
- `launch <total>`: this job (ROUND, ORDER) has launched: its pipeline runs are starting.
- `round <results.json> <total>`: this job's half of its round, added to this commit's.
- `final <total> <results.json>...`: every half, from the run's artifacts.
- `failed`: the run ended without results; what the comment holds stays, marked so.

The status line has two bars, a pair of cells a round, its forward job (➡️ until it
happens) and its reverse job (⬅️). Launches: "⏳ 🟦🟦 🟦⬅️ ➡️⬅️ launching 3/6 jobs", then
"🚀 🟦🟦 🟦🟦 🟦🟦 launched 6/6 jobs". Results: "🏁 🟩🟩 🟩⬅️ ➡️⬅️ running: 2/3 rounds
remaining", then "✅ 🟩🟩 🟩🟩 🟩🟩 done: 3 rounds in <the longest job, launch to
result>" (❌, and 🟥 for each half that failed, if any did).

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

from combine import duration, paired, render, render_pending, rounds_of, stored_rounds
from version import compatible

MARK = "<!-- bench -->"
DATA = re.compile(r"<!-- bench-data:([A-Za-z0-9+/=]*) -->")
SCORE = re.compile(r"^(\*\*after ran at .*?)(?: \((?:pending 🔁|failed ❌)[^)]*\))?$", re.M)
AFTER = re.compile(r"(\| after \(`[0-9a-f]+`\))(?: \((?:pending 🔁|failed ❌) `[0-9a-f]+`\))?")
STATUS = re.compile(r"^(?:\S+ )*(?:_Run: .*_|.*· \[run\]\(\S+\) for `[0-9a-f]+`)$", re.M)


def gh(*args: str) -> str:
    return subprocess.run(["gh", "api", *args], capture_output=True, text=True, check=True).stdout


def find(repo: str, pr: str) -> dict | None:
    pages = json.loads(gh(f"repos/{repo}/issues/{pr}/comments", "--paginate", "--slurp"))
    return next((c for page in pages for c in page if c["body"].startswith(MARK)), None)


def data(comment: dict | None) -> dict:
    m = DATA.search(comment["body"]) if comment else None
    return json.loads(base64.b64decode(m.group(1))) if m else {"head": None, "rounds": {}}


def phase(blob: dict, head: str) -> dict:
    """When each of this commit's rounds launched and came in, by round."""
    ph = blob.get("phase", {})
    if ph.get("head") != head:
        ph = blob["phase"] = {"head": head, "start": {}, "end": {}}
    return ph


JOBS = ("f", "r")  # each round's forward and reverse jobs (ab.py --order)
ARROW = {"f": "➡️", "r": "⬅️"}  # a job still to launch, or to come in


def job() -> str:
    """This job's key: its round and its order, "1f", "1r", ..."""
    return os.environ.get("ROUND", "1") + ("r" if os.environ.get("ORDER") == "reverse" else "f")


def status(blob: dict, state: str, run: str, head: str, total: int) -> str:
    """Two bars, a pair of cells a round: its forward job (➡️ until it happens) and its
    reverse job (⬅️). Launches: ⏳ while jobs launch (🟦 launched), 🚀 once all have.
    Results: 🏁 while they run (🟩 a half in), ✅ in its place once all are in (❌ if a
    round failed, 🟥). A round counts once both its cells are green."""
    ph = blob.get("phase", {}) if blob.get("phase", {}).get("head") == head else {"start": {}, "end": {}}
    halves = blob["rounds"] if blob["head"] == head else {}
    jobs = [f"{k}{o}" for k in range(1, total + 1) for o in JOBS]
    link = f" · [run]({run}) for `{head[:9]}`"
    if state not in ("running", "done"):
        return f"❌ **{state}**{link}"
    up = [j in ph["start"] or j in halves for j in jobs]
    n = sum(up)
    pairs = lambda cells: " ".join("".join(cells[i:i + len(JOBS)]) for i in range(0, len(cells), len(JOBS)))
    launches = f"{'🚀' if n >= len(jobs) else '⏳'} {pairs(['🟦' if u else ARROW[j[-1]] for j, u in zip(jobs, up)])} **" + (
        f"launched {n}/{len(jobs)} jobs**" if n >= len(jobs) else f"launching {n}/{len(jobs)} jobs**")
    got = [sum(f"{k}{o}" in halves for o in JOBS) for k in range(1, total + 1)]
    done = sum(g == len(JOBS) for g in got)
    bar = pairs(["🟩" if j in halves else "🟥" if state == "done" else ARROW[j[-1]] for j in jobs])
    if state == "done":
        took = [ph["end"][j] - ph["start"][j] for j in halves if j in ph["start"] and j in ph["end"]]
        failed = f" ({total - done} failed)" if done < total else ""
        results = f"{'❌' if failed else '✅'} {bar} **done: {done} rounds" + (
            f" in {duration(max(took), 0)}" if took else "") + f"{failed}**"
    else:
        results = f"🏁 {bar} **running: {total - done}/{total} rounds remaining**"
    return f"{launches} · {results}{link}"


def rounds_in(blob: dict) -> list[dict]:
    """The rounds with both halves in, each pair as one round."""
    return paired([blob["rounds"][k] for k in sorted(blob["rounds"])])


def settle(repo: str, pr: str, comment: dict | None, change, holds) -> None:
    """Apply `change` to the comment until `holds` sees it there: the round jobs write the
    one comment concurrently, and a write can land on top of another's."""
    for _ in range(8):
        text = change(comment)
        write(repo, pr, comment, text)
        time.sleep(random.uniform(1, 3))
        comment = find(repo, pr)
        if comment and holds(data(comment)):
            return
    sys.exit("the comment kept losing this job's change")


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

    if mode == "launch":
        total, key = int(sys.argv[4]), job()
        if comment is None:
            return

        def launched(c: dict) -> str:
            blob = data(c)
            phase(blob, head)["start"].setdefault(key, time.time())
            line = status(blob, "running", run, head, total)
            text = DATA.sub(lambda _: encode(blob), c["body"])
            return STATUS.sub(lambda _: line, text) if STATUS.search(text) else f"{text.rstrip()}\n\n{line}\n"

        settle(repo, pr, comment, launched, lambda b: key in b.get("phase", {}).get("start", {}))
        return

    if mode == "round":
        results, total = Path(sys.argv[4]), int(sys.argv[5])
        mine = rounds_of(json.loads(results.read_text()))[0]
        key, now = job(), time.time()

        def added(c: dict | None) -> str:
            blob = data(c)
            if blob["head"] != head:
                blob = {"head": head, "rounds": {}, "phase": blob.get("phase", {})}
            blob["rounds"][key] = mine
            phase(blob, head)["end"].setdefault(key, now)
            rounds = rounds_in(blob)
            state = "done" if len(rounds) >= total else "running"
            line = status(blob, state, run, head, total)
            if rounds:
                return body(blob, render(rounds, total), line)
            text = DATA.sub(lambda _: encode(blob), c["body"])  # no pair yet: the table stays as it is
            return STATUS.sub(lambda _: line, text) if STATUS.search(text) else f"{text.rstrip()}\n\n{line}\n"

        settle(repo, pr, comment, added, lambda b: b["head"] == head and key in b["rounds"])
        return

    if mode == "final":
        total = int(sys.argv[4])
        keyed = {}  # rounds/round-<k>-<order>/...: keyed as the jobs keyed them
        for p in sys.argv[5:]:
            half = rounds_of(json.loads(Path(p).read_text()))[0]
            keyed[f"{half['round']}{half['order'][0]}"] = half
        blob = {"head": head, "rounds": keyed, "phase": blob.get("phase", {})}
        rounds = rounds_in(blob)
        line = status(blob, "done", run, head, total)
        text = render(rounds, len(rounds)) if rounds else "## Benchmark: wikidata-pq pipeline\n\nNo round has both its halves ❌"
        write(repo, pr, comment, body(blob, text, line))
        return

    # pending or failed: what the comment holds stays, marked
    if mode == "pending" and blob["head"] == head:
        return  # another of this commit's jobs got here first
    if mode == "pending":
        phase(blob, head)  # every round not launched yet
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
