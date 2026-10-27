"""The PR tier's latency (bench 17.15.0): how long a pull request waits for the bench's
figures, this run's and the rolling median of the last WINDOW comparable runs', for the
finish job's comment (combine.render_latency_warning, render_durations).

A run's latency is from its start (the Actions run's `run_started_at`, so queueing for
runners counts) to the end of its last round job's "Comment: PR tier" step, when the
comment holds every round's PR-tier figures. Comparable runs are the Benchmark workflow's
pull request runs that have that step done in every round job (the PR tier's, bench
17.15.0 on), whatever their conclusion (a run cancelled by a later push during its extras
had its figures). The results store records master's runs only (full tier), so the runs
come from the Actions API. The median is the guard, not one run: a slow runner now and
then does not trip it.

Over LIMIT_S, the comment opens with a warning; the target is TARGET_S (bench/README.md,
"Keeping the PR tier fast").

Usage: latency.py <out.json> <rounds>   (needs GH_TOKEN, GITHUB_REPOSITORY, GITHUB_RUN_ID;
writes {} when it cannot tell)
"""

import json
import os
import re
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

TARGET_S = 6 * 60
LIMIT_S = 8 * 60
WINDOW = 10  # runs in the median
SCAN = 40  # pull request runs looked at, newest first, to find them
STEP = "Comment: PR tier"
JOB = re.compile(r"^round \((\d+), (forward|reverse)\)$")


def gh(*args: str) -> str:
    return subprocess.run(["gh", "api", *args], capture_output=True, text=True, check=True).stdout


def when(t: str) -> float:
    return datetime.fromisoformat(t.replace("Z", "+00:00")).timestamp()


def run_latency(repo: str, run: dict, jobs_needed: int) -> float | None:
    """A run's start to its last round job's PR-tier figures, in seconds; None for a run
    without them in every round job."""
    pages = json.loads(gh(f"repos/{repo}/actions/runs/{run['id']}/jobs?per_page=100", "--paginate", "--slurp"))
    ends = []
    for job in (j for page in pages for j in page["jobs"] if JOB.match(j["name"])):
        step = next((s for s in job.get("steps") or [] if s["name"] == STEP), None)
        if not step or step.get("conclusion") != "success" or not step.get("completed_at"):
            return None
        ends.append(when(step["completed_at"]))
    if len(ends) < jobs_needed:
        return None
    return round(max(ends) - when(run.get("run_started_at") or run["created_at"]), 1)


def latency(rounds: int) -> dict:
    repo, this = os.environ["GITHUB_REPOSITORY"], int(os.environ["GITHUB_RUN_ID"])
    current = json.loads(gh(f"repos/{repo}/actions/runs/{this}"))
    listed = json.loads(gh(f"repos/{repo}/actions/workflows/{current['workflow_id']}/runs"
                           f"?event=pull_request&per_page={SCAN}"))["workflow_runs"]
    found = []  # (run id, seconds), newest first, this run first
    for run in [current, *(r for r in listed if r["id"] != this)][:SCAN]:
        if run["id"] != this and run.get("status") != "completed":
            continue
        s = run_latency(repo, run, 2 * rounds)
        if s is not None:
            found.append({"run": run["id"], "s": s, "pr": [p["number"] for p in run.get("pull_requests") or []]})
        if len(found) >= WINDOW:
            break
    this_s = found[0]["s"] if found and found[0]["run"] == this else None
    return {"this_s": this_s, "median_s": statistics.median(f["s"] for f in found) if found else None,
            "runs": len(found), "window": WINDOW, "target_s": TARGET_S, "limit_s": LIMIT_S, "found": found}


def main() -> None:
    dest, rounds = Path(sys.argv[1]), int(sys.argv[2])
    try:
        out = latency(rounds)
    except (subprocess.CalledProcessError, KeyError, ValueError) as e:
        print(f"latency.py: cannot tell the PR tier's latency ({e})", file=sys.stderr)
        out = {}
    dest.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "found"}))
    if out.get("median_s") and out["median_s"] > LIMIT_S:
        print(f"::warning::the PR tier's median latency over the last {out['runs']} runs is "
              f"{out['median_s'] / 60:.1f} min, over its {LIMIT_S / 60:.0f}-minute limit")


if __name__ == "__main__":
    main()
