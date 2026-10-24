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

from version import VERSION

METRICS = [  # label, results key, kind, decimals measured to
    ("wall", "wall", "s", 2), ("throughput (entities/s)", "throughput", "rate", 0),
    ("CPU, all processes", "cpu_total", "s", 1),
    ("CPU, process", "process_cpu", "s", 1), ("CPU, partition", "partition_cpu", "s", 1),
    ("CPU, group merges", "close_cpu", "s", 1), ("peak RSS, all processes", "peak_sum_rss_mb", "MB", 0),
    ("peak RSS, one worker", "worker_peak_mb", "MB", 0), ("disk used, peak rise", "disk_peak_mb", "MB", 0),
    ("device bytes written", "device_written_mb", "MB", 0),
]
# Sizes' units, smallest first: their size in MB, and the decimals they add to keep the
# measurement's precision (1234 MB is 1.234 GB)
SIZES = [("MB", 1, 0), ("GB", 1000, 3), ("TB", 10**6, 6)]
# The scale results are translated to: the last release whose size is on record
# (docs/guide/performance.md), its entities in both sets
RELEASE, RELEASE_ENTITIES = "20260928", 121_815_642


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
                            for _, k, _, _ in METRICS}
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
        out.append({**f, "order": "pair", "cpus": f["cpus"] + r["cpus"],
                    "parity_ok": f["parity_ok"] and r["parity_ok"],
                    "parity": f["parity"] if f["parity_ok"] or not r["parity_ok"] else r["parity"],
                    "before_cached": f["before_cached"] and r["before_cached"],
                    "figures": {a: {k: avg(f["figures"][a][k], r["figures"][a][k]) for k in f["figures"][a]}
                                for a in f["figures"]}})
    return out


def render(rounds: list[dict], total: int) -> str:
    n = len(rounds)
    before, after = rounds[0]["names"]
    sha = lambda a: (rounds[0]["arms"][a]["sha"] or "?")[:9]
    fig = lambda a, k: [r["figures"][a][k] for r in rounds]
    mean = statistics.fmean
    pending = f" (pending 🔁 {n}/{total})" if n < total else ""
    pm = lambda v, f: f(mean(v)) + spread(v, f)

    speeds = [b / a for b, a in zip(fig(before, "wall"), fig(after, "wall"))]
    rounds_note = f"{n} round" + ("s" if n != 1 else "") + (f" of {total}" if n < total else "")
    w = rounds[0]["workload"]
    tier = f"{w.get('tier', 'quick')} tier, " if w.get("tier") else ""
    size = f" ({w['entities']:,} entities, {w['source_mb']:.0f} MB of source)" if w.get("entities") else ""
    out = [f"## Benchmark: wikidata-pq pipeline, {tier}{w['chunks']} chunks{size}, {rounds_note}{pending}", ""]
    out.append(f"**{after} ran at {pm(speeds, lambda v: f'{v:.2f}×')} the speed of {before}**"
               + f", {relative(1 / mean(speeds))} wall time")
    if w.get("entities"):
        scale = lambda a: mean(fig(a, "wall")) / w["entities"] * RELEASE_ENTITIES
        walls = [a - b for a, b in zip(fig(after, "wall"), fig(before, "wall"))]
        change = "≈, within noise" if noise(walls) else duration(scale(after) - scale(before), 0, sign=True)
        out += ["", f"**Full Wikidata ETA:** {RELEASE}'s {RELEASE_ENTITIES:,} entities through these stages at "
                f"these rates, on {rounds[0]['machine']['cores']} cores: {duration(scale(before), 0)} → "
                f"{duration(scale(after), 0)} ({change}). A straight extrapolation of this sample; it leaves out "
                "downloading, splitting and routing the dump, the uploads, and finalise."]
    parity_ok = all(r["parity_ok"] for r in rounds)
    cached = sum(r["before_cached"] for r in rounds)
    out += ["", f"**Parity:** {'passed' if parity_ok else 'FAILED'}", "",
            f"**{before}:** " + ("measured in this run" if not cached else
                                 f"in {cached} of {n} rounds an earlier measurement of the same pipeline on the "
                                 "same CPU model, from the results store; otherwise measured in the run"), ""]
    out.append(f"| | {before} (`{sha(before)}`) | {after} (`{sha(after)}`) | change |")
    out.append("|---|---|---|---|")
    for label, k, kind, decimals in METRICS:
        vb, va = fig(before, k), fig(after, k)
        if None in vb or None in va:  # a figure these results do not have
            continue
        unit, f = formatter(kind, mean(vb), decimals)
        cb, ca = pm(vb, f), pm(va, f)
        diffs = [a - b for a, b in zip(va, vb)]
        ratios = [a / b for a, b in zip(va, vb) if b]
        if ca == cb or all(d == 0 for d in diffs):  # the same as shown: no change to give
            change = "="
        elif noise(diffs):
            change = "≈"  # inside the 95% confidence interval of the paired changes: noise
        else:
            r = mean(ratios) if ratios else None
            rel = "" if r is None else relative(r) + (spread(ratios, lambda v: f"{100 * v:.1f}%") if "%" in relative(r)
                                                      else spread(ratios, lambda v: f"{v:.1f}×"))
            change = f"{f(mean(diffs), sign=True)}{spread(diffs, f)}" + (f" ({rel})" if rel else "")
        out.append(f"| {label}{unit} | {cb} | {'=' if ca == cb else ca} | {change} |")
    note = ("Each figure is the mean over the rounds ± their standard deviation. A round is a pair of runners, one "
            "running the commit before first and one the commit after, its figures the mean of the two; each change is "
            "paired within a round. ≈: a change inside the 95% confidence interval of its rounds' changes, which they cannot tell "
            "from noise.")
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
