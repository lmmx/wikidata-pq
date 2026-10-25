"""The round jobs of this run whose results did not come in, and why, as far as their logs
tell: for the summary and the comment (combine.py), so a run that lost runners says
which, rather than showing fewer rounds as if still pending.

A job is lost when the finish job has no results.json from it (its artifact is missing).
Its log (the GitHub API; where it cannot be had, its annotations) gives the last run it
started ("round 1 (forward): after"), the
last heartbeat of that run (ab.Heartbeat: its stage, the lowest memory available), and how
it ended: a runner shut down mid-run or lost to the server is taken, as far as can be told,
as out of memory (a GitHub-hosted runner's agent is lost when its memory runs out, and
these jobs end no other way mid-run), with the heartbeat's memory as the evidence.

Usage: lost.py <out.json> <rounds> <results.json>...   (needs GH_TOKEN, GITHUB_REPOSITORY,
GITHUB_RUN_ID; writes [] when it cannot tell)
"""

import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

JOB = re.compile(r"^round \((\d+), (forward|reverse)\)$")
STARTED = re.compile(r"round \d+ \((?:forward|reverse)\): (\S+)\s*$")
BEAT = re.compile(r"\[bench\] (\S+) \+(\d+)s: (.+?); memory available at its lowest ([\d.]+) GB")
LOST = ("The runner has received a shutdown signal", "lost communication with the server",
        "The operation was canceled")
# (in a job's annotations, where its log cannot be had: its step ended by SIGTERM, as a
# runner shutting down ends it)
SIGTERM = "Process completed with exit code 143"


def gh(*args: str) -> str:
    return subprocess.run(["gh", "api", *args], capture_output=True, text=True, check=True).stdout


class _Unauthorised(urllib.request.HTTPRedirectHandler):
    """A redirect without the token: the API sends a job's log from storage elsewhere,
    signed in its URL, which refuses a request that also carries a token."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None:
            new.remove_header("Authorization")
        return new


def job_log(repo: str, job: int) -> tuple[str, str | None]:
    """A job's log, and why not where it cannot be had: through gh, else directly, the
    redirect to its storage without the token."""
    try:
        if log := gh(f"repos/{repo}/actions/jobs/{job}/logs"):
            return log, None
        first = "an empty log"
    except subprocess.CalledProcessError as e:
        first = (e.stderr or "").strip()[:200]
    req = urllib.request.Request(f"https://api.github.com/repos/{repo}/actions/jobs/{job}/logs",
                                 headers={"Authorization": f"Bearer {os.environ.get('GH_TOKEN', '')}",
                                          "Accept": "application/vnd.github+json"})
    try:
        with urllib.request.build_opener(_Unauthorised).open(req, timeout=60) as r:
            return r.read().decode(errors="replace"), None
    except (OSError, ValueError) as e:
        return "", f"gh: {first}; direct: {e}"


def why(log: str, conclusion: str) -> dict:
    """What a lost job's log says: the run it was in, its last heartbeat, how it ended."""
    lines = log.splitlines()
    started = [m.group(1) for line in lines if (m := STARTED.search(line))]
    beats = [m for line in lines if (m := BEAT.search(line))]
    out = {"conclusion": conclusion, "run": started[-1] if started else None}
    if beats:
        b = beats[-1]
        out |= {"stage": b.group(3), "after_s": int(b.group(2)), "memory_available_gb": float(b.group(4))}
    if any(s in log for s in (*LOST[:2], SIGTERM)):
        out["reason"] = "its runner was lost mid-run: out of memory, as far as can be told"
    elif conclusion == "cancelled" or LOST[2] in log:
        out["reason"] = "cancelled"
    else:
        errors = [line.split("##[error]", 1)[1].strip() for line in lines if "##[error]" in line]
        out["reason"] = f"failed: {errors[0]}" if errors else "failed"
    return out


def lost(total: int, results: list[Path]) -> list[dict]:
    repo, run = os.environ["GITHUB_REPOSITORY"], os.environ["GITHUB_RUN_ID"]
    have = set()
    for p in results:
        res = json.loads(p.read_text())
        have.add((res.get("round"), res.get("order")))
    pages = json.loads(gh(f"repos/{repo}/actions/runs/{run}/jobs", "--paginate", "--slurp"))
    jobs = {(int(m.group(1)), m.group(2)): j for page in pages for j in page["jobs"] if (m := JOB.match(j["name"]))}
    out = []
    for key in sorted((k, o) for k in range(1, total + 1) for o in ("forward", "reverse")):
        if key in have:
            continue
        job = jobs.get(key)
        entry = {"round": key[0], "order": key[1]}
        if job is None:
            out.append(entry | {"reason": "never ran"})
            continue
        log, error = job_log(repo, job["id"])
        if not log:  # (its annotations, then, for how it ended)
            try:
                log = "\n".join(f"##[error]{a['message']}" for a in json.loads(
                    gh(f"repos/{repo}/check-runs/{job['id']}/annotations")) if a.get("annotation_level") == "failure")
            except (subprocess.CalledProcessError, ValueError) as e:
                error = f"{error}; annotations: {(getattr(e, 'stderr', '') or str(e)).strip()[:200]}"
        found = entry | {"job": job["html_url"]} | why(log, job.get("conclusion") or job.get("status"))
        out.append(found | {"log_chars": len(log)} | ({"log_error": error} if error else {}))
    return out


def main() -> None:
    dest, total = Path(sys.argv[1]), int(sys.argv[2])
    try:
        found = lost(total, [Path(p) for p in sys.argv[3:]])
    except (subprocess.CalledProcessError, KeyError, ValueError) as e:
        print(f"lost.py: cannot tell which jobs were lost ({e})", file=sys.stderr)
        found = []
    dest.write_text(json.dumps(found, indent=1))
    for e in found:
        print(json.dumps(e))


if __name__ == "__main__":
    main()
