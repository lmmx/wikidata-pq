"""The native bz2 decoder's worker count through the real split (lmmx/mibz2#65): the CI
bench's split stage (pipeline.split_stage: split-dump then route-release, each in a
process of its own, in a fresh directory of its own) on the same dump slice, once for each
count of decoder threads (WIKIDATA_BZ2_THREADS), against the default, `auto` (the variable
unset: dump.bz2_threads gives os.cpu_count()).

A sweep, not a section of the bench (ab.py): run by hand, by the "bz2 threads" workflow,
or on a workstation by `just bench-bz2-threads`. Only the variable a release can already
set changes between arms. The split's parse workers and writer pools stay at their
defaults (os.cpu_count() - 2 workers, as many writer threads), the same for every arm, and
each run's own log is checked for them.

Each round runs every arm once, in an order rotated by the round and reversed every
len(arms) rounds (`arm_order`), so over 2 x len(arms) rounds (the default) each arm runs in
each position equally often: a machine that warms up or slows down biases no arm.

Each run gives split_stage's figures (the split's and the route's wall and CPU, of the
process and every process it waits for; the peak RSS of the tree; `split_parity_ok`, the
split's parts holding exactly the sample's rows; the manifests' digest), and the split's
log checked:

- the decoder it ran and that decoder's own count of threads (dump's `bz2:` line, from
  the decoder built), which must be the native decoder at the arm's count;
- no chunk already written (the split from nothing: no resume);
- the same parse workers and writer threads as every other run.

A failed check, a parity failure or a manifest digest unlike the first run's fails the
sweep (exit 1). With `--decode` (the default), the decoder alone too (bz2native.DECODE),
at the same count, so the decode's own best count can be set beside the split's.

The summary gives each arm's median wall, CPU and peak RSS, and its paired ratios to
`auto`: each round's ratio, their mean and its 95% interval (combine.interval). An arm is
*lower* or *higher* only when that interval excludes no change. The machine's logical CPUs
and physical cores head it: a result holds for that topology.

Usage:
    bz2threads.py --data DATA_DIR --out DIR [--threads auto,1,2,4] [--rounds N] [--src CHECKOUT]
                  [--no-decode]   (the slice: BENCH_DUMP_SLICE, or made in DIR by dumpslice.py)
    bz2threads.py --full "--release R --windows 24 --window-mb 25" --out DIR [--threads auto,4]
                  (release-shaped: bench/full/run.py run, the dump cached by its `fetch`)
    bz2threads.py --report DIR    (the summary again, from DIR/runs.jsonl)

**Release-shaped** (`--full`, for the counts the split's sweep picks): bench-full's run
(bench/full/run.py run: split, both sets' runs, finalise; overlapped by default, as a
release runs, the network and the Hub off the timed path) once an arm a round, in the same
balanced order, each from the cached dump into a fresh work directory (run.py's). Each
run's total wall, CPU and peak RSS, and its split stage's wall and CPU, are paired with
`auto`'s; its log checked as the split's is (the decoder built and its threads, no chunk
already written); and its published files' digest (report.post_checks) must be the first
run's. A split-stage gain is the sweep's; a release's is this one's.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pipeline  # noqa: E402  (sets WIKIDATA_RELEASE and HF_HUB_OFFLINE, as the bench runs)
from bz2native import DECODE, timed  # noqa: E402
from combine import interval  # noqa: E402

QUICK = ["c0", "r50", "s30", "s90"]  # (ab.TIERS["quick"]: the pull request check's 4 chunks)
SPLITTING = re.compile(r"with (\d+) workers and (\d+) writer threads \(([\d,]+) chunks already written\)")
DECODER = re.compile(r"^bz2: the (\S+) decoder, (\d+) threads", re.M)


def arm_order(rnd: int, arms: list[str]) -> list[str]:
    """Round `rnd`'s order (from 0): the arms rotated by the round, reversed every
    len(arms) rounds, so over 2 x len(arms) rounds each arm runs in each position equally
    often (mibz2's examples/ab.rs, arm_order)."""
    n = len(arms)
    order = arms[rnd % n:] + arms[:rnd % n]
    return order[::-1] if (rnd // n) % 2 else order


def topology() -> dict:
    """The CPUs this process may run on, and the physical cores they make (each CPU's
    package and core id), so 4 logical CPUs on 2 cores are 2 pairs of SMT siblings."""
    cpus = sorted(os.sched_getaffinity(0))
    cores = set()
    for c in cpus:
        t = Path(f"/sys/devices/system/cpu/cpu{c}/topology")
        try:
            cores.add(((t / "physical_package_id").read_text().strip(), (t / "core_id").read_text().strip()))
        except OSError:
            cores.add(("?", str(c)))
    model = next((line.split(":", 1)[1].strip() for line in open("/proc/cpuinfo") if line.startswith("model name")), "?")
    l3 = Path("/sys/devices/system/cpu/cpu0/cache/index3/size")
    return {"cpu": model, "logical_cpus": len(cpus), "os_cpu_count": os.cpu_count(), "physical_cores": len(cores),
            "l3": l3.read_text().strip() if l3.exists() else None,
            "memory_gb": round(next(int(line.split()[1]) for line in open("/proc/meminfo") if line.startswith("MemTotal")) / 2**20, 1)}


def arm_env(threads: str) -> None:
    """This process's environment, which split_stage's processes take, at `threads`."""
    os.environ["WIKIDATA_BZ2_DECODER"] = "native"
    os.environ.pop("WIKIDATA_BZ2_THREADS", None)
    if threads != "auto":
        os.environ["WIKIDATA_BZ2_THREADS"] = threads


def checked(log: str, threads: str) -> dict:
    """The split's log checked: the decoder it built and that decoder's threads, no chunk
    already written, and its parse workers and writer threads (returned, to be compared)."""
    s, d = SPLITTING.search(log), DECODER.search(log)
    if not s or not d:
        raise SystemExit(f"the split's log names no {'workers' if not s else 'decoder'} line: is this a commit "
                         "whose split logs its bz2 decoder (`bz2: the ... decoder, N threads`)?")
    want = os.cpu_count() if threads == "auto" else int(threads)
    if d[1] != "native" or int(d[2]) != want:
        raise SystemExit(f"arm {threads}: the split ran the {d[1]} decoder on {d[2]} threads, not native on {want}")
    if s[3] != "0":
        raise SystemExit(f"arm {threads}: the split found {s[3]} chunks already written: not a split from nothing")
    return {"threads_used": int(d[2]), "split_workers": int(s[1]), "writer_threads": int(s[2])}


def sweep(args) -> int:
    arms = args.threads.split(",")
    arms = arms if "auto" in arms else ["auto", *arms]
    rounds = args.rounds or 2 * len(arms)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(str((args.data / f"{f}.parquet").resolve()) for f in QUICK)
    free_gb = shutil.disk_usage(out).free / 1e9
    if free_gb < args.min_free_gb:
        raise SystemExit(f"{free_gb:.1f} GB free at {out}: the sweep wants {args.min_free_gb} GB (--min-free-gb)")
    commit = subprocess.run(["git", "-C", args.src, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    machine = topology() | {"arms": arms, "rounds": rounds, "commit": commit, "free_gb_at_start": round(free_gb, 1)}
    (out / "machine.json").write_text(json.dumps(machine, indent=1))
    ok, first = True, None
    with open(out / "runs.jsonl", "a") as runs:
        for rnd in range(rounds):
            for pos, threads in enumerate(arm_order(rnd, arms)):
                arm_env(threads)
                name = out / f"r{rnd}-{threads}.split.log"
                with open(name, "w") as log:
                    fig = pipeline.split_stage(args.src / "src", files, 0, out, log)
                fig |= checked(name.read_text(), threads)
                slice_ = Path(os.environ.get("BENCH_DUMP_SLICE") or out / "dump-slice.json.bz2")
                machine.setdefault("slice_mb", round(slice_.stat().st_size / 1e6, 1))
                same = {k: fig[k] for k in ("split_manifests_sha256", "split_workers", "writer_threads")}
                first = first or same
                fig["same_as_first"] = same == first
                if not (fig["split_parity_ok"] and fig["same_as_first"]):
                    ok = False
                    print(f"round {rnd}, {threads} threads: parity {fig['split_parity_ok']}, against the first run "
                          f"{same} != {first}", file=sys.stderr)
                rec = {"round": rnd, "position": pos, "arm": threads, "kind": "split", "wall": fig["split_wall"],
                       "cpu": fig["split_cpu"], "rss_mb": fig["split_peak_rss_mb"],
                       "mb_s": round(machine["slice_mb"] / fig["split_wall"], 2),
                       "free_gb": round(shutil.disk_usage(out).free / 1e9, 1), **fig}
                runs.write(json.dumps(rec) + "\n")
                runs.flush()
                print(f"round {rnd} #{pos} {threads:>4}: split {fig['split_wall']:.2f} s, {fig['split_cpu']:.1f} CPU-s, "
                      f"{fig['split_peak_rss_mb']} MB; {fig['threads_used']} decoder threads, {fig['split_workers']} "
                      f"workers, {fig['writer_threads']} writer threads; parity {fig['split_parity_ok']}", flush=True)
                if args.decode:
                    with open(out / "decode.out", "w") as o:
                        t = timed([sys.executable, "-c", DECODE.format(path=str(slice_), backend="native")], out,
                                  dict(os.environ), o, pipeline.tree_rss_mb)
                    _, used, _ = (out / "decode.out").read_text().split()
                    if int(used) != fig["threads_used"]:
                        raise SystemExit(f"arm {threads}: the decoder alone ran {used} threads, the split "
                                         f"{fig['threads_used']}")
                    runs.write(json.dumps({"round": rnd, "position": pos, "arm": threads, "kind": "decode", **t,
                                           "threads_used": int(used),
                                           "mb_s": round(machine["slice_mb"] / t["wall"], 2)}) + "\n")
                    runs.flush()
                    print(f"round {rnd} #{pos} {threads:>4}: decode {t['wall']:.2f} s, {t['cpu']:.1f} CPU-s "
                          f"({used} threads)", flush=True)
    (out / "machine.json").write_text(json.dumps(machine, indent=1))
    print(report(out))
    return 0 if ok else 1


def full(args) -> int:
    """The release-shaped comparison (`--full`): bench-full's run once an arm a round."""
    arms = args.threads.split(",")
    arms = arms if "auto" in arms else ["auto", *arms]
    rounds = args.rounds or 2 * len(arms)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "-C", args.src, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    machine = topology() | {"arms": arms, "rounds": rounds, "commit": commit, "full": args.full}
    (out / "machine.json").write_text(json.dumps(machine, indent=1))
    ok, first = True, None
    with open(out / "runs.jsonl", "a") as runs:
        for rnd in range(rounds):
            for pos, threads in enumerate(arm_order(rnd, arms)):
                arm_env(threads)
                o = out / f"full-r{rnd}-{threads}"
                subprocess.run([sys.executable, HERE / "full" / "run.py", "run", *shlex.split(args.full), "--out", o],
                               cwd=args.src, check=True)
                r = json.loads((o / "result.json").read_text())
                fig = checked((o / "run.log").read_text(), threads)
                split = next(s for s in r["stages"] if s["stage"] == "split")
                digest = r["post"].get("published_digest")
                first = first or digest
                rec = {"round": rnd, "position": pos, "arm": threads, "kind": "full", "wall": r["total"]["wall_s"],
                       "cpu": r["total"]["cpu_s"], "rss_mb": r["total"]["peak_rss_tree_mb"],
                       "split_wall": split["wall_s"], "split_cpu": split["cpu_s"],
                       "other_cpu_s": r["total"].get("other_cpu_s"), "ok": r["ok"], "digest": digest,
                       "same_as_first": digest == first and digest is not None, **fig}
                ok = ok and rec["ok"] and rec["same_as_first"]
                runs.write(json.dumps(rec) + "\n")
                runs.flush()
                print(f"round {rnd} #{pos} {threads:>4}: release-shaped {rec['wall']:.0f} s, {rec['cpu']:.0f} CPU-s, "
                      f"split {rec['split_wall']:.0f} s; digest the first run's: {rec['same_as_first']}", flush=True)
    print(report(out))
    return 0 if ok else 1


def paired(recs: list[dict], arm: str, key: str) -> list[float]:
    """Each round's ratio of `arm`'s figure `key` to `auto`'s in the same round."""
    by = {(r["round"], r["arm"]): r[key] for r in recs}
    return [by[(n, arm)] / by[(n, "auto")] for n in sorted({r["round"] for r in recs})
            if (n, arm) in by and by.get((n, "auto"))]


def change(ratios: list[float]) -> str:
    """The mean paired change, its 95% interval and its direction: "-12.0% ±3.1% (lower)"."""
    if not ratios:
        return ""
    m, h = statistics.fmean(ratios), interval(ratios)
    return f"{100 * (m - 1):+.1f}% ±{100 * h:.1f}% ({'lower' if m + h < 1 else 'higher' if m - h > 1 else 'within noise'})"


def report(out: Path) -> str:
    """The sweep's summary (out/summary.md), from out/runs.jsonl and out/machine.json."""
    m = json.loads((out / "machine.json").read_text())
    recs = [json.loads(line) for line in (out / "runs.jsonl").read_text().splitlines() if line]
    lines = [f"# The native bz2 decoder's threads through the split ({m['commit'][:10]})", "",
             f"- **Machine:** {m['cpu']}; {m['logical_cpus']} logical CPUs on {m['physical_cores']} physical cores "
             f"(L3 {m['l3']}); {m['memory_gb']} GB. `auto` is {m['os_cpu_count']} decoder threads here.",
             f"- **Held constant (each run's log):** {sorted({r['split_workers'] for r in recs if 'split_workers' in r})} "
             f"parse workers, {sorted({r['writer_threads'] for r in recs if 'writer_threads' in r})} writer threads; "
             "every split from nothing.",
             (f"- **Workload:** bench-full's run (`{m['full']}`)" if m.get("full") else
              f"- **Workload:** the CI bench's split stage (split, then route) of a {m.get('slice_mb')} MB bz2 slice")
             + f"; {m['rounds']} rounds, each arm once a round, the order rotated and reversed (`arm_order`).",
             "- **Paired:** each round's ratio to `auto` in the same round, the mean and its 95% interval; *lower* or "
             "*higher* only when the interval excludes no change.", ""]
    fl = [r for r in recs if r["kind"] == "full"]
    if fl:
        lines += ["## Release-shaped (bench-full run)", "", f"`run.py run {m.get('full')}`", "",
                  "| threads | wall (s), median | CPU-s | peak RSS (MB) | split wall (s) | split CPU-s | wall vs auto | "
                  "CPU vs auto | split wall vs auto | split CPU vs auto | runs |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for a in m["arms"]:
            ar = [r for r in fl if r["arm"] == a]
            if ar:
                med = lambda k: statistics.median(r[k] for r in ar)  # noqa: E731
                lines.append("| " + " | ".join([
                    f"{a} ({ar[0]['threads_used']})", f"{med('wall'):.0f}", f"{med('cpu'):.0f}", f"{med('rss_mb'):.0f}",
                    f"{med('split_wall'):.0f}", f"{med('split_cpu'):.0f}",
                    *(("control",) * 4 if a == "auto" else
                      (change(paired(fl, a, k)) for k in ("wall", "cpu", "split_wall", "split_cpu"))), str(len(ar))]) + " |")
        bad = [r for r in fl if not (r["ok"] and r["same_as_first"])]
        lines += ["", f"- **Parity:** {len(fl) - len(bad)} of {len(fl)} runs ok, their published files' digest the "
                  "first run's." + (" **Failed: " + ", ".join(f"r{r['round']} {r['arm']}" for r in bad) + ".**"
                                    if bad else ""), ""]
    for kind, title in (("split", "The split (split-dump, routing at its end)"), ("decode", "The decoder alone")):
        rs = [r for r in recs if r["kind"] == kind]
        if not rs:
            continue
        lines += [f"## {title}", "",
                  "| threads | wall (s), median | bz2 MB/s | CPU-s, median | peak RSS (MB), median (max) | "
                  "wall vs auto | CPU vs auto | peak RSS vs auto | runs |", "|---|---|---|---|---|---|---|---|---|"]
        med = {}
        for a in m["arms"]:
            ar = [r for r in rs if r["arm"] == a]
            if not ar:
                continue
            med[a] = statistics.median(r["wall"] for r in ar)
            cells = [f"{a} ({ar[0].get('threads_used', '?')})", f"{med[a]:.2f}",
                     f"{statistics.median(r['mb_s'] for r in ar):.1f}", f"{statistics.median(r['cpu'] for r in ar):.1f}",
                     f"{statistics.median(r['rss_mb'] for r in ar):.0f} ({max(r['rss_mb'] for r in ar)})",
                     *(("control",) * 3 if a == "auto" else (change(paired(rs, a, k)) for k in ("wall", "cpu", "rss_mb"))),
                     str(len(ar))]
            lines.append("| " + " | ".join(cells) + " |")
        if kind == "split":
            bad = [r for r in rs if not (r["split_parity_ok"] and r["same_as_first"])]
            lines += ["", f"- **Parity:** {len(rs) - len(bad)} of {len(rs)} splits hold exactly the sample's rows, with "
                      "the first run's manifests, workers and writer threads."
                      + (" **Failed runs: " + ", ".join(f"r{r['round']} {r['arm']}" for r in bad) + ".**" if bad else ""),
                      "- **The route** (no decoder) for scale: median "
                      f"{statistics.median(r['route_wall'] for r in rs):.2f} s wall, "
                      f"{statistics.median(r['route_cpu'] for r in rs):.1f} CPU-s."]
        fastest = min(med, key=med.get)
        lines += [f"- **Fastest by median wall:** {fastest}"
                  + (f": wall {change(paired(rs, fastest, 'wall'))}, CPU {change(paired(rs, fastest, 'cpu'))}, "
                     "against auto." if fastest != "auto" else " (the default)."), ""]
    text = "\n".join(lines)
    (out / "summary.md").write_text(text)
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", type=Path, default=HERE.parent / "data", help="the bench data (the sample's chunks)")
    ap.add_argument("--out", type=Path, help="the results' directory (each run's files in a fresh directory in it)")
    ap.add_argument("--src", type=Path, default=HERE.parent, help="the wikidata-pq checkout whose split runs")
    ap.add_argument("--threads", default="auto,1,2,4", help="decoder thread counts, comma-separated (auto: unset)")
    ap.add_argument("--rounds", type=int, help="rounds (default 2 x the arms: each arm in each position equally)")
    ap.add_argument("--decode", action=argparse.BooleanOptionalAction, default=True, help="also time the decoder alone")
    ap.add_argument("--min-free-gb", type=float, default=2.0, help="refuse to start with less disk free at --out")
    ap.add_argument("--full", help="release-shaped: bench/full/run.py run's arguments (e.g. \"--release 20261019 "
                    "--windows 24 --window-mb 25\"), the dump fetched first")
    ap.add_argument("--report", type=Path, help="print the summary of a sweep's directory, and stop")
    args = ap.parse_args(argv)
    if args.report:
        print(report(args.report))
        return 0
    if not args.out:
        ap.error("--out is required")
    args.src = args.src.resolve()
    return full(args) if args.full else sweep(args)


if __name__ == "__main__":
    sys.exit(main())
