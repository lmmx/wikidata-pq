"""Which of the bench's checks a run makes (bench 17.15.0): the PR tier or the full tier.

Every run makes the **timed core** (each arm's split, pipeline, finalise, finalise on
three times the rows, claims_labels and the claims refs: ab.run_once) and **parity**. Then
its sections, each run untimed after the core and outside its figures:

- `bz2native`: the split by the native bz2 decoder against lbzip2's (bz2native.py);
- `split_gz`: the split from the gz dump, and on bench 10.0.0's orjson slice, with their
  parity (pipeline.split_gz);
- `overlap`: the split and both sets' pipelines one after another and at once, and over
  the emulated link (overlap.py, link.py), and the release wall modelled from them
  (overlap.release_wall);
- `misfit`: a chunk whose claims misfit claims.avsc (misfit.py), and its parity;
- `evolve`: a new claims field found mid-run (evolve.py), and its parity.

**The full tier** (a push to master, the nightly schedule, the Actions tab by default) runs
every section in each arm's run, after its core, as the harness always has.

**The PR tier** (every pull request) runs the core, parity and `bz2native` (PR_TIER). The
other sections, the extras, run on a pull request only when it is relevant to them:

- it changes a path an extra exercises (PATHS), or
- a package an extra exercises differs between the two arms' lockfiles (LOCKED: the
  `[[package]]` entry, version, source and checksum, compared across the checkouts, so a
  lockfile-only bump counts), or
- the full tier is asked for: the label `bench:full` on the pull request, or a line
  `bench: full` in its body.

On a pull request the extras run after both arms' cores (`deferred`), so the comment's
figures post once the PR tier is in and the extras update them after. `bz2native` is in
the PR tier, and also triggered by a change to mibz2 (the native decoder's crate, a
dependency of reshape) in reshape/Cargo.lock, so it runs on such a change even if it ever
leaves the PR tier.

New sections go in the full tier by default (CLAUDE.md, "Measuring"; bench/README.md).

Usage: tiers.py <before checkout> <after checkout> [--suite pr|full] [--label NAME]...
                [--labels-json JSON] [--body-file F] [--out suite.json]
Prints the selection as JSON; with GITHUB_OUTPUT set, also writes `suite`, `inline` and
`deferred` (each a comma-separated list) there.
"""

import argparse
import fnmatch
import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

# In the order each arm's run makes them (ab.run_once, as before bench 17.15.0)
SECTIONS = ["split_gz", "overlap", "misfit", "evolve", "bz2native"]
PR_TIER = {"bz2native"}
LABELS = {
    "bz2native": "the native bz2 decoder against lbzip2 (bz2native.py)",
    "split_gz": "the split from the gz dump and on bench 10.0.0's orjson slice (pipeline.py --split-gz)",
    "overlap": "the stages overlapped and over the emulated link (overlap.py, link.py), and the release wall "
               "modelled from them",
    "misfit": "the misfit chunk (misfit.py)",
    "evolve": "a new claims field (evolve.py)",
}

# The claims and schema code: what the misfit chunk and the new field exercise (the split's
# claims build, the pipeline's process step, genson's inference, the release schema)
CLAIMS = ["src/wikidata/claims_build.py", "src/wikidata/claims_schema.py", "src/wikidata/claims_numeric.py",
          "src/wikidata/process.py", "src/wikidata/release_schema.py", "src/wikidata/schema_evolution.py",
          "src/wikidata/schemas/**", "src/wikidata/partitioning/claims.py", "src/wikidata/dump.py",
          "reshape/src/claims.rs", "reshape/src/claims_arrow.rs"]
# A pull request's changed paths that make an extra run (a pattern ending in /** is a
# directory and everything under it). The bench's own modules for each extra are in its
# list; a change to the harness's orchestration (ab.py, combine.py, the workflow) runs
# none: ask for the full tier (`bench:full`) where it matters.
PATHS = {
    # The split and its dump reading: the dump's decoders and formats, routing, the split's
    # builders (Python and reshape's Rust), and the dump slices
    "split_gz": ["src/wikidata/dump.py", "src/wikidata/scholarly.py", "src/wikidata/claims_build.py",
                 "src/wikidata/entities_build.py", "src/wikidata/partitioning/fused.py", "src/wikidata/config.py",
                 "reshape/**", "bench/dumpslice.py", "bench/pipeline.py"],
    # The stages' orchestration (main's run, close_group and finalise, the pool's
    # background uploads, the finalise runs across processes, the committer) and the Hub
    # stand-in's paths (push, pull, cards, state), and the release model
    "overlap": ["src/wikidata/main.py", "src/wikidata/config.py", "src/wikidata/state.py", "src/wikidata/pool.py",
                "src/wikidata/cross_run.py", "src/wikidata/sort_by_id.py", "src/wikidata/cards.py",
                "src/wikidata/card_stats.py", "src/wikidata/initial.py", "src/wikidata/hub.py",
                "src/wikidata/push/**", "src/wikidata/pull/**", "bench/overlap.py", "bench/link.py",
                "bench/release.py"],
    "misfit": [*CLAIMS, "bench/misfit.py", "bench/dumpslice.py"],
    "evolve": [*CLAIMS, "bench/evolve.py"],
    "bz2native": ["src/wikidata/dump.py", "reshape/src/bz2_py.rs", "reshape/Cargo.toml", "bench/bz2native.py"],
}
# A package whose lockfile entry differing between the arms makes a section run:
# (lockfile, package name)
LOCKED = {
    "bz2native": [("reshape/Cargo.lock", "mibz2")],
    "misfit": [("uv.lock", "polars-genson")],
    "evolve": [("uv.lock", "polars-genson")],
}
FULL_LABEL = "bench:full"
FULL_LINE = re.compile(r"(?im)^\s*bench:\s*full\s*$")


def matches(path: str, pattern: str) -> bool:
    if pattern.endswith("/**"):
        return path.startswith(pattern[:-2])
    return fnmatch.fnmatchcase(path, pattern)


def lock_entry(checkout: Path, lockfile: str, name: str) -> list[dict] | None:
    """The lockfile's `[[package]]` entries named `name` (a lockfile can hold two
    versions of one package), None if the checkout has no such lockfile."""
    try:
        lock = tomllib.loads((checkout / lockfile).read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return None
    return sorted((p for p in lock.get("package", []) if p.get("name") == name), key=json.dumps)


def describe(entries: list[dict] | None) -> str:
    if entries is None:
        return "no lockfile"
    if not entries:
        return "absent"
    return ", ".join(e.get("version", "?") + (f" ({e['checksum'][:12]})" if e.get("checksum") else "")
                     for e in entries)


def changed_paths(before: Path, after: Path) -> list[str]:
    """The files that differ between the two checkouts' commits (both are worktrees of
    one repository with its history)."""
    a = subprocess.run(["git", "-C", before, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    out = subprocess.run(["git", "-C", after, "diff", "--name-only", a, "HEAD"], capture_output=True, text=True,
                         check=True).stdout
    return [p for p in out.splitlines() if p]


def select(before: Path, after: Path, suite: str, labels: list[str], body: str) -> dict:
    """The sections this run makes, and why each runs or is skipped."""
    reasons = {s: [] for s in SECTIONS}
    lock_changes = {}
    for s, locks in LOCKED.items():
        for lockfile, name in locks:
            x, y = lock_entry(before, lockfile, name), lock_entry(after, lockfile, name)
            if x != y:
                lock_changes[name] = f"{name} {describe(x)} → {describe(y)} in {lockfile}"
                reasons[s].append(lock_changes[name])
    forced = None
    if suite == "pr":
        if FULL_LABEL in labels:
            forced = f"the label `{FULL_LABEL}`"
        elif FULL_LINE.search(body or ""):
            forced = "a line `bench: full` in the pull request's body"
    if suite == "full" or forced:
        why = forced or "the full tier"
        for s in SECTIONS:
            reasons[s].insert(0, why)
        # (a pull request's full tier: the PR tier's sections in each arm's run, the extras
        # after both arms', so its first figures do not wait on them)
        return {"suite": suite, "forced": forced,
                "inline": [s for s in SECTIONS if not forced or s in PR_TIER],
                "deferred": [s for s in SECTIONS if forced and s not in PR_TIER], "skipped": [], "reasons": reasons,
                "lock_changes": lock_changes, "changed": None}
    paths = changed_paths(before, after)
    for s, patterns in PATHS.items():
        hit = sorted({p for p in paths if any(matches(p, pat) for pat in patterns)})
        if hit:
            reasons[s].append("changes " + ", ".join(f"`{p}`" for p in hit[:4])
                              + (f" and {len(hit) - 4} more" if len(hit) > 4 else ""))
    for s in PR_TIER:
        reasons[s].insert(0, "the PR tier")
    run = [s for s in SECTIONS if reasons[s]]
    return {"suite": suite, "forced": None,
            # (the PR tier's own sections in each arm's run; the extras after both arms')
            "inline": [s for s in run if s in PR_TIER],
            "deferred": [s for s in run if s not in PR_TIER],
            "skipped": [s for s in SECTIONS if s not in run], "reasons": reasons,
            "lock_changes": lock_changes, "changed": len(paths)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("before", type=Path)
    ap.add_argument("after", type=Path)
    ap.add_argument("--suite", choices=["pr", "full"], default="full")
    ap.add_argument("--label", action="append", default=[])
    ap.add_argument("--labels-json", help="the pull request's labels, a JSON list of names (the workflow's)")
    ap.add_argument("--body-file", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    body = args.body_file.read_text() if args.body_file and args.body_file.exists() else ""
    labels = args.label + (json.loads(args.labels_json or "null") or [])
    sel = select(args.before, args.after, args.suite, [l for l in labels if l], body)
    text = json.dumps(sel, indent=1)
    if args.out:
        args.out.write_text(text)
    print(text)
    if out := os.environ.get("GITHUB_OUTPUT"):
        with open(out, "a") as f:
            f.write(f"suite={sel['suite']}\ninline={','.join(sel['inline'])}\ndeferred={','.join(sel['deferred'])}\n")
    if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary, "a") as f:
            f.write(summary_md(sel) + "\n")


def summary_md(sel: dict) -> str:
    """The selection as the comment and the job summary give it."""
    tier = "full tier" if sel["suite"] == "full" else "PR tier"
    out = [f"**Checks: the {tier}**" + (f", the full tier asked for by {sel['forced']}" if sel["forced"] else "")
           + ": the timed core and parity, then:", ""]
    for s in SECTIONS:
        if s in sel["inline"] or s in sel["deferred"]:
            when = " (after both arms' cores)" if s in sel["deferred"] else ""
            out.append(f"- {LABELS[s]}: runs{when}, for {'; '.join(sel['reasons'][s])}")
        else:
            out.append(f"- {LABELS[s]}: not run on this PR (full tier on master)")
    return "\n".join(out)


if __name__ == "__main__":
    sys.exit(main())
