"""Rounds of the benchmark (ab.py's results.json files, one or more rounds each) as one
result: per round, the commit before's and after's figures on that round's runner, and
from them, over the rounds, each figure's mean ± standard deviation, and the change from
before to after, which is paired within each round, so runners of different speeds can
share the work.

Usage: combine.py <total rounds> <summary.md> <results.json>...
"""

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

from release import RELEASE, RELEASE_ENTITIES
from version import VERSION

METRICS = [  # label, results key, kind, decimals measured to
    ("ETA, a release", "eta", "s", 0), ("ETA: pipeline", "pipeline_eta", "s", 0),
    ("ETA: finalise", "finalise_eta", "s", 0), ("ETA: split", "split_eta", "s", 0),
    ("wall, with finalise", "total_wall", "s", 2), ("wall", "wall", "s", 2),
    ("finalise: compact", "compact_wall", "s", 2), ("finalise: sort", "sort_wall", "s", 2),
    ("finalise, 3× the rows", "finalise_x3_wall", "s", 2),
    ("split: split", "split_wall", "s", 2), ("split: route", "route_wall", "s", 2), ("CPU, split", "split_cpu_total", "s", 1),
    ("throughput (entities/s)", "throughput", "rate", 0),
    ("CPU, all processes", "cpu_total", "s", 1),
    ("CPU, process", "process_cpu", "s", 1), ("CPU, partition", "partition_cpu", "s", 1),
    ("CPU, group merges", "close_cpu", "s", 1), ("CPU, finalise", "finalise_cpu", "s", 1), ("peak RSS, all processes", "peak_sum_rss_mb", "MB", 0),
    ("peak RSS, one worker", "worker_peak_mb", "MB", 0), ("disk used, peak rise", "disk_peak_mb", "MB", 0),
    ("device bytes written", "device_written_mb", "MB", 0),
]
# Figures kept with the metrics but not shown in the table: the fit's prediction of the
# finalise run on three times the rows (see pipeline.finalise_at_scale)
FIGURES = ["finalise_x3_wall_fit"]
# Sizes' units, smallest first: their size in MB, and the decimals they add to keep the
# measurement's precision (1234 MB is 1.234 GB)
SIZES = [("MB", 1, 0), ("GB", 1000, 3), ("TB", 10**6, 6)]


def duration(seconds: float, decimals: int, sign: bool = False) -> str:
    """1h23m45s, 10m23s, 11.42s: seconds keep their decimals only under a minute."""
    lead = ("+" if seconds > 0 else "-" if seconds < 0 else "") if sign else "-" if seconds < 0 else ""
    v = abs(seconds)
    if round(v) < 60:
        return f"{lead}{v:.{decimals}f}s"
    h, rest = divmod(round(v), 3600)
    m, s = divmod(rest, 60)
    return f"{lead}{h}h{m:02d}m{s:02d}s" if h else f"{lead}{m}m{s:02d}s"


def formatter(kind: str, before: float, decimals: int):
    """The row's format, set by its value before: a duration, or the largest size unit
    in which the value before is at least 1. Returns (unit label, format)."""
    if kind == "s":
        return "", lambda v, sign=False: duration(v, decimals, sign)
    if kind == "rate":
        return "", lambda v, sign=False: f"{v:{'+' if sign else ''},.{decimals}f}"
    name, size, extra = ([u for u in SIZES if abs(before) >= u[1]] or SIZES[:1])[-1]
    return f" ({name})", lambda v, sign=False: f"{v / size:{'+' if sign else ''},.{decimals + extra}f}"


def relative(ratio: float) -> str:
    """A change as a percentage, or as a multiple once it more than doubles (2.3×) or
    falls below half (÷2.5)."""
    return f"{ratio:.1f}×" if ratio > 2 else f"÷{1 / ratio:.1f}" if 0 < ratio < 0.5 else f"{100 * (ratio - 1):+.1f}%"


# Student's t at 97.5% for n - 1 degrees of freedom, by n (from 2; the last for n >= 10)
T95 = [None, 12.71, 4.30, 3.18, 2.78, 2.57, 2.45, 2.36, 2.31, 2.26]


def noise(diffs: list[float]) -> bool:
    """Whether paired changes are inside the 95% confidence interval of their mean: noise."""
    n = len(diffs)
    return n > 1 and abs(statistics.fmean(diffs)) <= T95[min(n, len(T95)) - 1] * statistics.stdev(diffs) / n**0.5


def interval(values: list[float]) -> float:
    """The half-width of the 95% confidence interval of the values' mean (0 for one value)."""
    n = len(values)
    return T95[min(n, len(T95)) - 1] * statistics.stdev(values) / n**0.5 if n > 1 else 0.0


def ratios_of(rounds: list[dict], k: str) -> list[float]:
    """Each round's paired ratio of figure `k`, after over before: each half's on its own
    runner, averaged over the round's halves, so a fast and a slow runner count alike (a
    record without halves: its figures' ratio)."""
    before, after = rounds[0]["names"]
    out = []
    for r in rounds:
        v = r.get("ratios", {}).get(k)
        if v is None and r["figures"][before].get(k) and r["figures"][after].get(k) is not None:
            v = r["figures"][after][k] / r["figures"][before][k]
        out.append(v)
    return out


def percent(ratios: list[float]) -> str:
    """The mean paired change as a percentage and its 95% interval: "+1.2% ±2.0%"."""
    return f"{100 * (statistics.fmean(ratios) - 1):+.1f}% ±{100 * interval(ratios):.1f}%"


def spread(values: list[float], f) -> str:
    """" ± <standard deviation>", in the values' format; nothing for one value."""
    return f" ± {f(statistics.stdev(values))}" if len(values) > 1 else ""


def rounds_of(res: dict) -> list[dict]:
    """ab.py's results as one record per round: each commit's figures on its runner."""
    before, after = list(res["arms"])
    out = []
    for rnd in sorted({r["round"] for r in res["runs"] if r["arm"] == after}):
        figures = {}
        for arm in (before, after):
            # the commit before, when cached, has its runs from the store, whatever their round
            runs = [r for r in res["runs"] if r["arm"] == arm and (r["round"] == rnd or (arm == before and res.get("before_cached")))]
            figures[arm] = {k: statistics.median(v) if (v := [r[k] for r in runs if r.get(k) is not None]) else None
                            for k in [m[1] for m in METRICS] + FIGURES}
            if timed := [r["phases"] for r in runs if r.get("phases")]:  # where the time goes (phases.py)
                figures[arm]["phases"] = mean_phases(timed)
            if res["workload"].get("entities"):  # the sample's throughput, derived from its wall time
                figures[arm]["throughput"] = res["workload"]["entities"] / figures[arm]["wall"]
        out.append({
            "cpu": res["machine"]["cpu"], "machine": res["machine"], "before_cached": res.get("before_cached", False),
            "parity_ok": res["parity_ok"], "parity": res["parity"][after],
            "arms": {a: {"sha": res["arms"][a]["sha"], "packages": res["arms"][a]["packages"]} for a in (before, after)},
            "names": [before, after], "figures": figures,
            "workload": res["workload"], "data": res["data_sha256sums"], "bench": res.get("bench", "1.0.0"),
            "round": res.get("round"), "order": res.get("order"), "cpus": [res["machine"]["cpu"]],
        })
    return out


def mean_phases(runs: list[dict]) -> dict:
    """Runs' phases (span -> function -> own CPU seconds) averaged, a function a run lacks
    counted as 0."""
    spans = {s for r in runs for s in r}
    return {s: {f: statistics.fmean(r.get(s, {}).get(f, 0.0) for r in runs)
                for f in {f for r in runs for f in r.get(s, {})}} for s in spans}


def paired(halves: list[dict]) -> list[dict]:
    """A round's forward and reverse halves (ab.py --order) as one round: each figure the
    mean of the two, so a change is the mean of the two halves' changes, and what running
    first or second does cancels. A round with one half in is left out until the other
    is; a record from before the pairs (no order) is a round of its own."""
    out, halves_of = [], {}
    for h in halves:
        if h.get("order") is None:
            out.append(h)
        else:
            halves_of.setdefault(h["round"], {})[h["order"]] = h
    for rnd in sorted(halves_of):
        pair = halves_of[rnd]
        if len(pair) < 2:
            continue
        f, r = pair["forward"], pair["reverse"]
        avg = lambda x, y: None if x is None or y is None else (x + y) / 2
        before, after = f["names"]
        ratio = lambda h, k: (h["figures"][after][k] / h["figures"][before][k]
                              if h["figures"][before][k] and h["figures"][after][k] is not None else None)
        out.append({**f, "order": "pair", "cpus": f["cpus"] + r["cpus"], "halves": [f, r],
                    "ratios": {k: avg(ratio(f, k), ratio(r, k)) for k in f["figures"][after] if k != "phases"},
                    "parity_ok": f["parity_ok"] and r["parity_ok"],
                    "parity": f["parity"] if f["parity_ok"] or not r["parity_ok"] else r["parity"],
                    "before_cached": f["before_cached"] and r["before_cached"],
                    "figures": {a: {k: (mean_phases([f["figures"][a][k], r["figures"][a][k]]) if k == "phases"
                                        else avg(f["figures"][a][k], r["figures"][a][k]))
                                    for k in f["figures"][a] if k != "phases" or k in r["figures"][a]}
                                for a in f["figures"]}})
    return out


# Where the time goes: each span's stages, a stage shown if it is this share of its span's
# instrumented CPU time in either commit (the rest summed as "other")
PHASE_SPANS = [
    ("the split and routing of the dump slice", lambda p: p.get("split")),
    ("the pipeline, every chunk", lambda p: p.get("pipeline")),
    ("finalise, per byte (the run on three times the rows less the first; what the ETA scales)",
     lambda p: None if "finalise" not in p or "finalise_x3" not in p else
     {f: p["finalise_x3"].get(f, 0.0) - p["finalise"].get(f, 0.0)
      for f in set(p["finalise"]) | set(p["finalise_x3"])}),
]
PHASE_SHARE = 0.01


def render_phases(rounds: list[dict]) -> list[str]:
    """The table of where the CPU time goes, by stage, before and after: each stage's own
    CPU time (phases.py) and share, and its paired change where it is beyond noise."""
    before, after = rounds[0]["names"]
    if not all(r["figures"][a].get("phases") for r in rounds for a in (before, after)):
        return []
    out = ["", "**Where the CPU goes**, by stage: each stage's own CPU time in its process (its worker's "
           "native threads included, the stages it calls apart), the mean over the rounds, and its share of "
           "the span's.", "",
           f"| stage | {before} | {after} | change |", "|---|---|---|---|"]
    for title, span in PHASE_SPANS:
        per = [(span(r["figures"][before]["phases"]), span(r["figures"][after]["phases"])) for r in rounds]
        if any(b is None or a is None for b, a in per):
            continue
        stages = {f for b, a in per for f in (*b, *a)}
        mb = {f: statistics.fmean(b.get(f, 0.0) for b, _ in per) for f in stages}
        ma = {f: statistics.fmean(a.get(f, 0.0) for _, a in per) for f in stages}
        tb, ta = sum(mb.values()) or 1.0, sum(ma.values()) or 1.0
        shown = sorted((f for f in stages if max(mb[f] / tb, ma[f] / ta) >= PHASE_SHARE), key=lambda f: -ma[f])
        out.append(f"| *{title}*: {tb:.1f}s → {ta:.1f}s | | | |")
        rest_b, rest_a = tb - sum(mb[f] for f in shown), ta - sum(ma[f] for f in shown)
        for f, vb, va in [*((f, mb[f], ma[f]) for f in shown), ("other", rest_b, rest_a)]:
            if f == "other" and max(vb / tb, va / ta) < 0.001:
                continue
            ratios = [a.get(f, 0.0) / b[f] for b, a in per if f != "other" and b.get(f, 0.0) > 0]
            change = ""
            if len(ratios) == len(per) and not noise([v - 1 for v in ratios]):
                change = f"**{percent(ratios)}**"
            elif vb > 0 and va == 0:
                change = "**gone**"
            elif vb == 0 and va > 0:
                change = "**new**"
            out.append(f"| `{f}` | {vb:.1f}s ({100 * vb / tb:.0f}%) | {va:.1f}s ({100 * va / ta:.0f}%) | {change} |")
    return out


def render(rounds: list[dict], total: int) -> str:
    n = len(rounds)
    before, after = rounds[0]["names"]
    sha = lambda a: (rounds[0]["arms"][a]["sha"] or "?")[:9]
    fig = lambda a, k: [r["figures"][a][k] for r in rounds]
    mean = statistics.fmean
    pending = f" (pending 🔁 {n}/{total})" if n < total else ""
    pm = lambda v, f: f(mean(v)) + spread(v, f)

    # The score: the whole run, finalise included, when every round has it
    whole = "total_wall" if all(r["figures"][a].get("total_wall") for r in rounds for a in (before, after)) else "wall"
    speeds = [1 / v for v in ratios_of(rounds, whole)]
    rounds_note = f"{n} round" + ("s" if n != 1 else "") + (f" of {total}" if n < total else "")
    w = rounds[0]["workload"]
    tier = f"{w.get('tier', 'quick')} tier, " if w.get("tier") else ""
    size = f" ({w['entities']:,} entities, {w['source_mb']:.0f} MB of source)" if w.get("entities") else ""
    out = [f"## Benchmark: wikidata-pq pipeline, {tier}{w['chunks']} chunks{size}, {rounds_note}{pending}", ""]
    out.append(f"**{after} ran at {pm(speeds, lambda v: f'{v:.2f}×')} the speed of {before}**"
               + f", {relative(1 / mean(speeds))} wall time")
    if all(r["figures"][a].get("eta") for r in rounds for a in (before, after)):
        etas = ratios_of(rounds, "eta")
        change = "no change beyond noise" if noise([v - 1 for v in etas]) else percent(etas)
        part = lambda a, k: duration(mean(fig(a, k)), 0)
        halves = [h for r in rounds for h in r.get("halves", [r])]
        models = sorted({h["cpu"] for h in halves})
        by_cpu = lambda c, a, k: duration(mean([h["figures"][a][k] for h in halves if h["cpu"] == c]), 0)
        fits = [abs(f / m - 1) for a in (before, after) for f, m in zip(fig(a, "finalise_x3_wall_fit"), fig(a, "finalise_x3_wall"))]
        has_split = all(r["figures"][a].get("split_eta") for r in rounds for a in (before, after))
        split_part = (f"the split {part(before, 'split_eta')} → {part(after, 'split_eta')}, " if has_split else "")
        out += ["", f"**Full Wikidata ETA:** {part(before, 'eta')} → {part(after, 'eta')} ({change}): {split_part}the pipeline "
                f"{part(before, 'pipeline_eta')} → {part(after, 'pipeline_eta')}, finalise "
                f"{part(before, 'finalise_eta')} → {part(after, 'finalise_eta')}", "",
                "<details><summary>The ETA by runner CPU, and how it is made</summary>", "",
                f"{RELEASE}'s {RELEASE_ENTITIES:,} entities, on {rounds[0]['machine']['cores']} cores. The pipeline's "
                "stages scaled by entities; finalise as it runs at the release's size (a key over "
                f"{RELEASE}'s bucket threshold sorted through buckets), measured on the sample and on three times its "
                "rows: its time per key as measured, its CPU time per source byte scaled to the release's bytes over "
                f"the cores. The line through the two runs is within {100 * max(fits):.0f}% of the second's wall time. "
                + ("The split (split-dump, then route-release) run on a slice of a dump made of the sample's own "
                   "entities: its CPU time per byte of the slice's bz2, scaled to the release's dump over the cores. "
                   "It leaves out downloading the dump, and every upload. " if has_split else
                   "It leaves out downloading, splitting and routing the dump, and every upload. ")
                + "The runners differ in "
                "speed by up to about 1.7×; the change is paired, each commit against the other on one runner.", "",
                "| runner CPU | runs | ETA, before | ETA, after | pipeline, before → after | finalise, before → after |",
                "|---|---|---|---|---|---|"]
        for c in models:
            k = sum(h["cpu"] == c for h in halves)
            out.append(f"| {c} | {k} | {by_cpu(c, before, 'eta')} | {by_cpu(c, after, 'eta')} | "
                       f"{by_cpu(c, before, 'pipeline_eta')} → {by_cpu(c, after, 'pipeline_eta')} | "
                       f"{by_cpu(c, before, 'finalise_eta')} → {by_cpu(c, after, 'finalise_eta')} |")
        out += ["", "</details>"]
    elif w.get("entities"):
        # Before bench 4.0.0: the pipeline alone, scaled by entities
        scale = lambda a: mean(fig(a, "wall")) / w["entities"] * RELEASE_ENTITIES
        walls = [a - b for a, b in zip(fig(after, "wall"), fig(before, "wall"))]
        change = "≈, within noise" if noise(walls) else duration(scale(after) - scale(before), 0, sign=True)
        out += ["", f"**Full Wikidata ETA (pipeline only):** {duration(scale(before), 0)} → "
                f"{duration(scale(after), 0)} ({change})."]
    parity_ok = all(r["parity_ok"] for r in rounds)
    cached = sum(r["before_cached"] for r in rounds)
    out += ["", f"**Parity:** {'passed' if parity_ok else 'FAILED'}", "",
            f"**{before}:** " + ("measured in this run" if not cached else
                                 f"in {cached} of {n} rounds an earlier measurement of the same pipeline on the "
                                 "same CPU model, from the results store; otherwise measured in the run"), ""]
    rows, quiet = [], 0  # (quiet: the figures whose change is noise)
    for label, k, kind, decimals in METRICS:
        vb, va = fig(before, k), fig(after, k)
        if None in vb or None in va:  # a figure these results do not have
            continue
        unit, f = formatter(kind, mean(vb), decimals)
        cb, ca = pm(vb, f), pm(va, f)
        diffs = [a - b for a, b in zip(va, vb)]
        ratios = [v for v in ratios_of(rounds, k) if v is not None]
        if ca == cb or all(d == 0 for d in diffs):  # the same as shown: no change to give
            change = ""
        elif not ratios:
            change = f"{f(mean(diffs), sign=True)}"
        elif noise([v - 1 for v in ratios]):
            quiet += 1  # its 95% interval holds no change: noise, the cell left blank
            change = ""
        else:
            r = mean(ratios)
            change = f"**{f(mean(diffs), sign=True)} ({percent(ratios) if '%' in relative(r) else relative(r)})**"
        rows.append(f"| {label}{unit} | {cb} | {'=' if ca == cb else ca} | {change} |")
    out += [f"| | {before} (`{sha(before)}`) | {after} (`{sha(after)}`) | change |", "|---|---|---|---|", *rows]
    out += render_phases(rounds)
    note = ("Each figure is the mean over the rounds ± their standard deviation, which takes in the runners' "
            "differences in speed. A round is a pair of runners, one running the commit before first and one the "
            "commit after. Each change is paired: each runner's after over its before, averaged over the round's two "
            "runners, then over the rounds, given with its 95% confidence interval (±). A change is given only "
            "when its interval holds no zero; otherwise the rounds cannot tell it from noise and the cell is blank.")
    cpus = Counter(c for r in rounds for c in r.get("cpus", [r["cpu"]]))
    m = rounds[0]["machine"]
    pk = lambda a: ", ".join(f"{k} {v}" for k, v in rounds[0]["arms"][a]["packages"].items() if v)
    pa, pb = rounds[0]["arms"][after]["packages"], rounds[0]["arms"][before]["packages"]
    out += [
        "", note if n > 1 else "One round.", "",
        "**Runners:** " + ", ".join(f"{c}" + (f" ×{k}" if k > 1 else "") for c, k in cpus.items())
        + f"; {m['cores']} cores, {m['memory_gb']} GB, Python {m['python']}  ",
        f"**Packages, {before}:** {pk(before)}  ",
        f"**Packages, {after}:** " + ("=" if pa == pb else ", ".join(f"{k} {v}" for k, v in pa.items() if v != pb.get(k))) + "  ",
        f"**Data:** the 11 chunk files, SHA256SUMS hashing to `{rounds[0]['data'][:16]}`  ",
        f"**Harness:** bench {rounds[0].get('bench', VERSION)}", "",
    ]
    shown = next((r for r in rounds if not r["parity_ok"]), rounds[0])
    out.append(f"<details><summary>Parity, {after} against {before}</summary>\n\n```json\n"
               f"{json.dumps(shown['parity'], indent=1)}\n```\n</details>")
    return "\n".join(out) + "\n"


def stored_rounds(path: Path) -> list[dict]:
    """A store's runs/<commit>.json (one results.json, or a list of them) as rounds."""
    data = json.loads(path.read_text())
    return paired([r for res in (data if isinstance(data, list) else [data]) for r in rounds_of(res)])


def render_pending(stored: list[dict], total: int, after_sha: str) -> str:
    """The table before any round is in: the commit before's figures from its own run in
    the results store (it was that run's commit after), the commit after pending."""
    n = len(stored)
    name = stored[0]["names"][1]
    fig = lambda k: [r["figures"][name][k] for r in stored]
    sha = (stored[0]["arms"][name]["sha"] or "?")[:9]
    out = [f"## Benchmark: wikidata-pq pipeline, {stored[0]['workload']['chunks']} chunks, 0 rounds of {total} "
           f"(pending 🔁 0/{total})", "",
           "**after:** pending ⏳", "",
           f"**before:** its figures from its own run in the results store ({n} round{'s' if n != 1 else ''}, "
           + ", ".join(sorted({r['cpu'] for r in stored})) + "), until this run's rounds measure it again", "",
           f"| | before (`{sha}`) | after (`{after_sha[:9]}`) | change |", "|---|---|---|---|"]
    for label, k, kind, decimals in METRICS:
        v = fig(k)
        if None in v:
            continue
        unit, f = formatter(kind, statistics.fmean(v), decimals)
        out.append(f"| {label}{unit} | {f(statistics.fmean(v))}{spread(v, f)} | pending ⏳ | pending |")
    return "\n".join(out) + "\n"


def main() -> None:
    total, dest = int(sys.argv[1]), Path(sys.argv[2])
    rounds = paired([r for p in sys.argv[3:] for r in rounds_of(json.loads(Path(p).read_text()))])
    dest.write_text(render(rounds, total) if rounds else "No round has both its halves.\n")


if __name__ == "__main__":
    main()
