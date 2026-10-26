"""The benchmark's history as one small JSON file, for the docs site's dashboard
(docs/guide/bench-history.md, docs/javascripts/bench-history.js).

Reads every quick-tier run in the bench results store (the `bench-results` branch:
records.jsonl and runs/quick/<commit>.json) and the backfill runs committed under
docs/benchmarks/backfill/ (today's harness run on older master commits, from the
Benchmark workflow's Actions tab), and writes, for each figure:

- each master commit's mean over every run of it under one major version of the harness
  (both arms of every record count: the commit before of one record is a measurement of
  that commit too), with the 95% interval of the mean, overall and by runner CPU;
- each run's paired change (after over before, on the same runner, a round's two halves
  averaged, the interval over the rounds), as combine.py computes it for the comment.

A figure's meaning holds only within a major version (bench/CHANGELOG.md), so each
major version is its own line; figures that a later harness back-computes under an
earlier model (eta_10, pipeline_eta_9, finalise_eta_8, ...) join that model's line.

Nothing here runs in the pipeline. Usage:

    python scripts/bench_history.py [--store DIR | --ref origin/bench-results] [--out PATH]

Without --store, the store is read from the git ref (fetch it first:
`git fetch origin bench-results`). Commits' order, dates and subjects come from
`git log --first-parent` of --master (default origin/master, else HEAD).

To backfill an older commit: run the Benchmark workflow from the Actions tab on master,
quick tier, `candidate` master's head and `before` the older commit (the workflow runs
master's harness on both arms' own src/ and lockfiles), download its round-* artifacts,
and keep them with

    python scripts/bench_history.py --add-backfill <run id> <round dirs>/results.json...
"""

import argparse
import json
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKFILL = ROOT / "docs" / "benchmarks" / "backfill"
OUT = ROOT / "docs" / "assets" / "bench-history.json"

# Student's t at 97.5% for n - 1 degrees of freedom, by n (from 2; the last for n >= 10),
# as bench/combine.py
T95 = [None, 12.71, 4.30, 3.18, 2.78, 2.57, 2.45, 2.36, 2.31, 2.26]

# (key, label, unit, group, back-computed keys {key: model major}). Units: s, MB, n (a count).
FIGURES = [
    ("measured_wall", "Wall, split + pipeline + finalise (measured)", "s", "measured", {}),
    ("measured_cpu", "CPU, split + pipeline + finalise (measured)", "s", "measured", {}),
    # (with claims_labels' refs stage, run after finalise's first run: bench 14.1.0 on)
    ("measured_wall_refs", "Wall, split + pipeline + finalise + claims refs (measured)", "s", "measured", {}),
    ("measured_cpu_refs", "CPU, split + pipeline + finalise + claims refs (measured)", "s", "measured", {}),
    ("total_wall", "Wall, pipeline + finalise", "s", "measured", {}),
    ("wall", "Wall, pipeline", "s", "measured", {}),
    ("cpu_total", "CPU, pipeline (all processes)", "s", "measured", {}),
    ("finalise_wall", "Wall, finalise", "s", "measured", {}),
    ("finalise_cpu", "CPU, finalise", "s", "measured", {}),
    ("finalise_x3_wall", "Wall, finalise on 3× the rows", "s", "measured", {}),
    ("refs_wall", "Wall, claims refs", "s", "measured", {}),
    ("refs_cpu", "CPU, claims refs", "s", "measured", {}),
    # (the other tables' group files in finalise's first run: bench 15.1.0 on)
    ("finalise_files", "Group files, finalise (the other tables')", "n", "measured", {}),
    ("split_wall_total", "Wall, split + route", "s", "measured", {}),
    ("split_cpu_total", "CPU, split + route", "s", "measured", {}),
    ("gz_split_cpu_total", "CPU, split + route from the gz dump", "s", "measured", {}),
    ("eta", "ETA, a release (modelled)", "s", "eta", {"eta_10": 10, "eta_9": 9}),
    ("split_eta", "ETA: split (modelled)", "s", "eta", {"split_eta_10": 10}),
    ("pipeline_eta", "ETA: pipeline (modelled)", "s", "eta", {"pipeline_eta_10": 10, "pipeline_eta_9": 9}),
    ("finalise_eta", "ETA: finalise (modelled)", "s", "eta", {"finalise_eta_9": 9, "finalise_eta_8": 8}),
    ("gz_split_eta", "ETA: split from the gz dump (modelled)", "s", "eta", {}),
    ("eta_refs", "ETA, a release, with the claims refs (modelled)", "s", "eta", {}),
    # (a release's wall as its recipes overlap its stages, with the uploads at 7 MB/s:
    # bench 17.1.0 on, overlap.release_wall)
    ("release_wall_modelled", "Release wall, stages overlapped, with the uploads (modelled)", "s", "eta", {}),
    # (and with finalise's own transfers, at each upload rate: bench 17.2.0 on,
    # overlap.release_wall; the release wall above is 17.2.0's model from then, 17.1.0's
    # kept as release_wall_modelled_17_1)
    *((f"release_wall_link_{r}_modelled", f"Release wall with finalise's transfers, link at {r} MB/s (modelled)",
       "s", "eta", {}) for r in (7, 30, 60, 120)),
    # (finalise's three runs through the stand-in Hub, over an emulated link: bench 17.2.0
    # on, link.py)
    ("finalise_link_0_wall", "Wall, finalise's three runs through the stand-in Hub, no link", "s", "measured", {}),
    ("finalise_link_7_wall", "Wall, finalise's three runs over a 7 MB/s link", "s", "measured", {}),
    ("finalise_link_60_wall", "Wall, finalise's three runs over a 60 MB/s link", "s", "measured", {}),
    ("overlap_link_7_wall", "Wall, split and both pipelines at once, pushes over a 7 MB/s link", "s", "measured", {}),
    ("peak_sum_rss_mb", "Peak RSS, pipeline (all processes)", "MB", "memory", {}),
    ("worker_peak_mb", "Peak RSS, pipeline (one worker)", "MB", "memory", {}),
    ("split_peak_rss_mb", "Peak RSS, split", "MB", "memory", {}),
    ("finalise_peak_rss_mb", "Peak RSS, finalise (all processes)", "MB", "memory", {}),
    ("finalise_peak_rss_mb_x3", "Peak RSS, finalise on 3× the rows (all processes)", "MB", "memory", {}),
    ("refs_peak_rss_mb", "Peak RSS, claims refs (all processes)", "MB", "memory", {}),
    ("refs_worker_peak_mb", "Peak RSS, claims refs (one worker)", "MB", "memory", {}),
    ("release_bucket_worker_rss_mb", "Peak RSS at a release: a bucket sort (modelled)", "MB", "memory", {}),
    ("release_finalise_rss_mb", "Peak RSS at a release: finalise (modelled)", "MB", "memory", {}),
    ("disk_peak_mb", "Scratch disk, peak rise", "MB", "disk", {}),
    ("device_written_mb", "Device bytes written", "MB", "disk", {}),
    ("process_cpu", "CPU, process (the pipeline's chunks)", "s", "stages", {}),
    ("partition_cpu", "CPU, partition", "s", "stages", {}),
    ("close_cpu", "CPU, group merges", "s", "stages", {}),
    ("pipeline_claims_cpu", "CPU, the pipeline's claims stages", "s", "stages", {}),
    ("compact_cpu", "CPU, finalise: compact", "s", "stages", {}),
    ("sort_cpu", "CPU, finalise: sort", "s", "stages", {}),
]
# Where the CPU goes (phases.py, bench 6.0.0 on): each span's stages, a stage charted if
# it is this share of its span's timed CPU in the latest record that has the span
PHASE_SPANS = {"split": "the split", "pipeline": "the pipeline", "per_byte": "finalise, per byte"}
PHASE_SHARE = 0.08
PHASE_MAX = 5


# Milestones: the campaign's merges that moved an end-to-end figure (lmmx/wikidata-pq#97),
# labelled on the charts' x axes, and the commits backfilled with today's harness
MILESTONES = {84, 85, 93, 102, 109, 112, 118, 122, 124, 127, 132, 134}


def major(version: str) -> int:
    return int(version.split(".")[0])


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True).stdout


def interval(values: list[float]) -> float:
    n = len(values)
    return T95[min(n, len(T95)) - 1] * statistics.stdev(values) / n**0.5 if n > 1 else 0.0


def sig(x: float, digits: int = 4) -> float:
    return float(f"{x:.{digits}g}")


def summary(values: list[float]) -> dict | None:
    """The values' mean, the half-width of its 95% interval, and their count, rounded."""
    if not values:
        return None
    return {"m": sig(statistics.fmean(values)), "e": sig(interval(values), 2), "n": len(values)}


def cpu_name(cpu: str) -> str:
    """A runner's CPU model, short: "EPYC 7763", "Xeon 8370C"."""
    if m := re.search(r"EPYC (\w+)", cpu):
        return f"EPYC {m.group(1)}"
    if m := re.search(r"(?i)xeon\(r\)(?: platinum)? (\w+(?:-\w+)?)", cpu):
        return f"Xeon {m.group(1).upper()}"
    return cpu.strip()


def phase_values(run: dict) -> dict:
    """A run's phases (span -> stage -> own CPU-s) as spans of this file: the split, the
    pipeline, and finalise's part per byte (the run on three times the rows less the first,
    as the comment shows it)."""
    p = run.get("phases") or {}
    out = {}
    if p.get("split"):
        out["split"] = p["split"]
    if p.get("pipeline"):
        out["pipeline"] = p["pipeline"]
    if p.get("finalise") and p.get("finalise_x3"):
        out["per_byte"] = {f: p["finalise_x3"].get(f, 0.0) - p["finalise"].get(f, 0.0)
                           for f in set(p["finalise"]) | set(p["finalise_x3"])}
    return out


def flat(run: dict) -> dict:
    """A run's figures: its numbers, and its phases as "phase:<span>:<stage>"."""
    out = {k: v for k, v in run.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    for span, stages in phase_values(run).items():
        for f, v in stages.items():
            out[f"phase:{span}:{f}"] = v
    return out


def load_store(store: Path | None, ref: str) -> list[tuple[dict, list[dict]]]:
    """The store's quick-tier records, each with its raw results (a list of halves)."""
    def read(path: str) -> str:
        return (store / path).read_text() if store else git("show", f"{ref}:{path}")

    out = []
    for line in read("records.jsonl").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("tier") != "quick":
            continue
        try:
            raw = json.loads(read(f"runs/quick/{rec['commit']}.json"))
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
        out.append(({**rec, "source": "ledger"}, raw if isinstance(raw, list) else [raw]))
    return out


def load_backfill() -> list[tuple[dict, list[dict]]]:
    """The committed backfill runs: each a record (as records.jsonl's) and its halves."""
    out = []
    for path in sorted(BACKFILL.glob("*.json")):
        d = json.loads(path.read_text())
        out.append(({**d["record"], "source": "backfill"}, d["halves"]))
    return out


def arm_shas(half: dict) -> dict:
    return {a: v["sha"] for a, v in half["arms"].items()}


def paired(halves: list[dict], key: str) -> list[float]:
    """Each round's paired ratio of `key`, after over before (combine.paired, ratios_of):
    each half's on its own runner, a round's halves averaged."""
    by_round = defaultdict(dict)
    for i, h in enumerate(halves):
        runs = {r["arm"]: r for r in h["runs"]}
        b, a = flat(runs.get("before", {})).get(key), flat(runs.get("after", {})).get(key)
        if b and a is not None:
            # (a record from before the pairs, bench 1.x: each half a round of its own)
            by_round[h.get("round") if h.get("order") else ("half", i)][h.get("order")] = a / b
    return [statistics.fmean(v.values()) for k, v in by_round.items() if len(v) == 2 or isinstance(k, tuple)]


def phase_keys(entries: list[tuple[dict, list[dict]]]) -> list[tuple[str, str, str]]:
    """The stages charted (span, stage, key): the latest record with each span, its
    stages at PHASE_SHARE of the span's timed CPU or more, PHASE_MAX at most."""
    out = []
    for span in PHASE_SPANS:
        for rec, halves in reversed(entries):
            runs = [r for h in halves for r in h["runs"] if r["arm"] == "after"]
            sums = defaultdict(list)
            for r in runs:
                for f, v in phase_values(r).get(span, {}).items():
                    sums[f].append(v)
            if not sums:
                continue
            means = {f: statistics.fmean(v) for f, v in sums.items()}
            total = sum(v for v in means.values() if v > 0)
            top = sorted((f for f in means if total and means[f] / total >= PHASE_SHARE), key=lambda f: -means[f])
            out += [(span, f, f"phase:{span}:{f}") for f in top[:PHASE_MAX]]
            break
    return out


def build(entries: list[tuple[dict, list[dict]]], master: str) -> dict:
    log = [line.split("\x00") for line in git("log", "--first-parent", "--format=%H%x00%cI%x00%s", master).splitlines()]
    order = {sha: i for i, (sha, _, _) in enumerate(reversed(log))}
    meta = {sha: (date, subject) for sha, date, subject in log}

    phases = phase_keys(entries)
    figures = [(k, label, unit, group, back) for k, label, unit, group, back in FIGURES]
    figures += [(key, f"CPU, {PHASE_SPANS[span]}: {stage}", "s", "phases", {}) for span, stage, key in phases]
    keys = {k for k, *_ in figures} | {b for *_, back in figures for b in back}

    # Every run of a commit, by the major version that measured it
    runs_of = defaultdict(list)  # (sha, major) -> [(cpu, figures, record)]
    records = []
    for rec, halves in entries:
        m = major(rec["bench"])
        for h in halves:
            shas, cpu = arm_shas(h), cpu_name(h.get("machine", {}).get("cpu", "?"))
            for r in h["runs"]:
                sha = shas.get(r["arm"])
                if sha in order:
                    runs_of[(sha, m)].append((cpu, {k: v for k, v in flat(r).items() if k in keys}, rec))
        shas = arm_shas(halves[0])
        if shas.get("after") in order and shas.get("before") in order:
            records.append((rec, halves, shas))

    commits = sorted({sha for sha, _ in runs_of}, key=order.get)
    index = {sha: i for i, sha in enumerate(commits)}
    cpus = sorted({cpu for v in runs_of.values() for cpu, _, _ in v})

    out_figures = []
    for key, label, unit, group, back in figures:
        lines = defaultdict(list)  # (model major, back-computed) -> points
        for (sha, m), runs in runs_of.items():
            for src, model, computed in [(key, m, False), *((b, bm, True) for b, bm in back.items())]:
                vals = [(cpu, f[src]) for cpu, f, _ in runs if f.get(src) is not None]
                if not vals:
                    continue
                s = summary([v for _, v in vals])
                by_cpu = defaultdict(list)
                for cpu, v in vals:
                    by_cpu[cpu].append(v)
                lines[(model, computed, m)].append({
                    "c": index[sha], **s,
                    "k": {cpus.index(c): [(x := summary(v))["m"], x["e"], x["n"]] for c, v in sorted(by_cpu.items())}})
        # Paired changes: each record's after over its before, on the same runners
        changes = []
        for rec, halves, shas in records:
            ratios = [x for x in paired(halves, key) if x]
            if not ratios:
                continue
            # A backfill's commit after is its reference (master's head as it ran): the
            # change shown is the older commit's, against it
            at, vs = ("before", "after") if rec.get("reference") == "after" else ("after", "before")
            s = summary([100 * ((1 / x if at == "before" else x) - 1) for x in ratios])
            changes.append({"c": index[shas[at]], "vs": index[shas[vs]], **s, "bench": major(rec["bench"]),
                            "source": rec["source"]})
        if not lines:
            continue
        out_lines = []
        for (model, computed, measured_by), pts in sorted(lines.items()):
            out_lines.append({"model": model, "measured_by": measured_by, "computed": computed,
                              "points": sorted(pts, key=lambda p: p["c"])})
        out_figures.append({"key": key, "label": label, "unit": unit, "group": group,
                            "lines": out_lines, "changes": sorted(changes, key=lambda p: p["c"])})

    def pr(subject: str) -> int | None:
        m = re.search(r"\(#(\d+)\)\s*$", subject)
        return int(m.group(1)) if m else None

    benches = defaultdict(set)
    for sha, m in runs_of:
        benches[sha].add(m)
    return {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "master": git("rev-parse", master).strip(),
        "cpus": cpus,
        "commits": [{"sha": sha, "date": meta[sha][0], "subject": meta[sha][1], "pr": pr(meta[sha][1]),
                     "benches": sorted(benches[sha]), **({"milestone": True} if pr(meta[sha][1]) in MILESTONES else {})}
                    for sha in commits],
        # Where each major version's own records start: its first record's commit before
        "starts": {str(m): min(index[s["before"]] for rec, _, s in records if rec["source"] == "ledger" and major(rec["bench"]) == m)
                   for m in sorted({major(rec["bench"]) for rec, _, _ in records if rec["source"] == "ledger"})},
        "records": [{"after": index[s["after"]], "before": index[s["before"]], "bench": rec["bench"],
                     "source": rec["source"], "run": rec["run"]} for rec, _, s in records],
        "figures": out_figures,
    }


# A backfill run's raw results kept: its figures and where its CPU goes, not its samples
DROP = {"timeline", "closes", "pipeline_tables", "chunks", "parity", "split_manifests_sha256"}


def add_backfill(run: str, files: list[Path]) -> Path:
    """A Benchmark run from the Actions tab (its round artifacts' results.json files) as a
    backfill file: today's harness on an older commit, against master's head as the
    reference (its commit after)."""
    halves = []
    for f in files:
        d = json.loads(f.read_text())
        halves.append({
            "bench": d["bench"], "round": d.get("round"), "order": d.get("order"), "workload": d["workload"],
            "machine": {"cpu": d["machine"]["cpu"]}, "arms": {a: {"sha": v["sha"]} for a, v in d["arms"].items()},
            "parity_ok": d.get("parity_ok"),
            "runs": [{k: v for k, v in r.items() if k not in DROP and (k == "phases" or not isinstance(v, (dict, list)))}
                     for r in d["runs"]],
        })
    halves.sort(key=lambda h: (h["round"] or 0, h["order"] or ""))
    if not all(h["parity_ok"] for h in halves):
        sys.exit("parity failed: not a measurement to keep")
    shas = arm_shas(halves[0])
    record = {"bench": halves[0]["bench"], "tier": halves[0]["workload"]["tier"], "commit": shas["after"],
              "previous": shas["before"], "reference": "after", "rounds": len({h["round"] for h in halves}),
              "run": f"https://github.com/lmmx/wikidata-pq/actions/runs/{run}"}
    BACKFILL.mkdir(parents=True, exist_ok=True)
    out = BACKFILL / f"{shas['before'][:7]}-{run}.json"
    out.write_text(json.dumps({"record": record, "halves": halves}, separators=(",", ":")) + "\n")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--store", type=Path, help="a checkout of the bench-results branch")
    ap.add_argument("--ref", default="origin/bench-results", help="the store's git ref, without --store")
    ap.add_argument("--master", help="the branch whose commits are charted (default origin/master, else HEAD)")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--add-backfill", nargs="+", metavar=("RUN_ID", "RESULTS_JSON"),
                    help="keep a dispatched run (its rounds' results.json) under docs/benchmarks/backfill/, and stop")
    args = ap.parse_args()
    if args.add_backfill:
        print(add_backfill(args.add_backfill[0], [Path(f) for f in args.add_backfill[1:]]), file=sys.stderr)
        return
    master = args.master
    if master is None:
        master = "origin/master" if subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--verify", "-q", "origin/master"],
                                                   capture_output=True).returncode == 0 else "HEAD"
    entries = load_store(args.store, args.ref) + load_backfill()
    data = build(entries, master)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, separators=(",", ":"), allow_nan=False))
    print(f"{args.out}: {len(data['commits'])} commits, {len(data['figures'])} figures, "
          f"{len(entries)} runs ({sum(e[0]['source'] == 'backfill' for e in entries)} backfilled), "
          f"{args.out.stat().st_size / 1000:.0f} kB", file=sys.stderr)


if __name__ == "__main__":
    main()
