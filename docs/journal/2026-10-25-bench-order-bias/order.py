"""The bench's order effect: what running second rather than first does to a figure.

Reads benchmark runs, each a JSON list of a run's six round jobs (ab.py's results.json:
bench-results' runs/<tier>/<sha>.json, or a run's round-* artifacts gathered by
gather.py), and per figure reports:

- the order effect o, the mean log ratio of the arm run second over the arm run first,
  estimated in every run as (mean forward log ratio - mean reverse log ratio) / 2, which
  holds whatever the change between the arms is; pooled over runs, ± its 95% interval;
- the same from the runs whose arms are the same pipeline (equal cache keys: a
  harness-only or docs-only commit), where every job's second/first is o directly;
- the reported change (combine.paired, then the t interval over the three rounds) in
  those A/A runs: its mean, and how often it is flagged (|change| beyond its interval),
  which should be 5% if the interval is honest.

Usage: python3 -I order.py [--csv OUT] RUN.json...
"""

import argparse
import json
import math
import statistics
import sys

FIGURES = ["split_cpu_total", "split_cpu", "route_cpu", "orjson_split_cpu_total", "split_peak_rss_mb", "finalise_cpu", "measured_cpu", "measured_wall", "eta", "split_eta", "finalise_eta",
           "pipeline_eta", "cpu_total", "wall", "total_wall", "split_wall", "finalise_wall", "close_cpu",
           "gz_split_cpu_total", "peak_sum_rss_mb", "device_written_mb"]
T95 = [None, 12.71, 4.30, 3.18, 2.78, 2.57, 2.45, 2.36, 2.31, 2.26]


def tq(n):
    return T95[min(n, len(T95)) - 1] if n > 1 else math.inf


def load(path):
    jobs = json.load(open(path))
    jobs = jobs if isinstance(jobs, list) else [jobs]
    jobs = [j for j in jobs if j.get("order") in ("forward", "reverse")]
    if not jobs:
        return None
    before, after = list(jobs[0]["arms"])
    keys = [jobs[0]["arms"][a].get("cache_key") for a in (before, after)]
    out = []
    for j in jobs:
        fig = {}
        for a in (before, after):
            runs = [r for r in j["runs"] if r["arm"] == a]
            fig[a] = runs[0] if len(runs) == 1 else None
        if None not in fig.values():
            out.append({"order": j["order"], "round": j["round"], "fig": fig})
    return {"path": path, "bench": jobs[0].get("bench"), "sha": jobs[0]["arms"][after].get("sha"),
            "aa": keys[0] is not None and keys[0] == keys[1], "names": (before, after), "jobs": out}


def lr(job, names, k):
    """log(after / before) in one job, None where the figure is missing or zero."""
    b, a = (job["fig"][n].get(k) for n in names)
    return math.log(a / b) if a and b and a > 0 and b > 0 else None


def per_run(run, k):
    """(o, T, reported change, its interval, forward mean %, reverse mean %) for one run."""
    names = run["names"]
    f = {j["round"]: lr(j, names, k) for j in run["jobs"] if j["order"] == "forward"}
    r = {j["round"]: lr(j, names, k) for j in run["jobs"] if j["order"] == "reverse"}
    fv, rv = [v for v in f.values() if v is not None], [v for v in r.values() if v is not None]
    if not fv or not rv:
        return None
    o = (statistics.fmean(fv) - statistics.fmean(rv)) / 2
    # combine.paired: each round's ratio the mean of its halves' ratios; t interval over rounds
    pairs = [(math.exp(f[n]) + math.exp(r[n])) / 2 for n in f if n in r and f[n] is not None and r[n] is not None]
    if len(pairs) < 2:
        return None
    change = statistics.fmean(pairs) - 1
    half = tq(len(pairs)) * statistics.stdev(pairs) / len(pairs) ** 0.5
    return {"o": o, "change": change, "half": half, "fwd": math.expm1(statistics.fmean(fv)),
            "rev": math.expm1(statistics.fmean(rv)), "n": len(pairs), "fv": fv, "rv": rv}


def pooled(values):
    n = len(values)
    if n < 2:
        return (values[0] if values else math.nan), math.nan, n
    return statistics.fmean(values), tq(n) * statistics.stdev(values) / n**0.5, n


def pct(x):
    return f"{100 * x:+.2f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--min-bench", type=int, default=0, help="only runs of this bench major or later")
    ap.add_argument("--show", nargs="*", default=[], help="run shas (prefixes) to print per job")
    ap.add_argument("runs", nargs="+")
    args = ap.parse_args()
    runs = [r for p in args.runs if (r := load(p)) and r["jobs"]
            and int((r["bench"] or "0").split(".")[0]) >= args.min_bench]
    aa = [r for r in runs if r["aa"]]
    print(f"{len(runs)} runs ({len(aa)} A/A: {' '.join(r['sha'][:7] + '@' + str(r['bench']) for r in aa)})\n")
    print("| figure | order effect o, all runs | o, A/A runs (each job) | A/A: mean reported change "
          "| A/A: median ±interval | A/A flagged | all runs: median ±interval |")
    print("|---|---|---|---|---|---|---|")
    rows = []
    for k in FIGURES:
        res = [(r, x) for r in runs if (x := per_run(r, k))]
        if not res:
            continue
        o_all = pooled([x["o"] for _, x in res])
        # A/A: each job's log(second / first), o itself
        sf = []
        for r in aa:
            for j in r["jobs"]:
                v = lr(j, r["names"], k)
                if v is not None:
                    sf.append(v if j["order"] == "forward" else -v)
        o_aa = pooled(sf)
        aar = [x for r, x in res if r["aa"]]
        flagged = sum(abs(x["change"]) > x["half"] for x in aar)
        mean_aa = pooled([x["change"] for x in aar])
        med_aa = statistics.median([x["half"] for x in aar]) if aar else math.nan
        med_all = statistics.median([x["half"] for _, x in res])
        print(f"| `{k}` | {pct(o_all[0])} ±{100 * o_all[1]:.2f} (n={o_all[2]}) | {pct(o_aa[0])} ±{100 * o_aa[1]:.2f} "
              f"(n={o_aa[2]}) | {pct(mean_aa[0])} ±{100 * mean_aa[1]:.2f} | ±{100 * med_aa:.2f}% "
              f"| {flagged}/{len(aar)} | ±{100 * med_all:.2f}% |")
        for r, x in res:
            rows.append([r["sha"][:9], r["bench"], int(r["aa"]), k, x["o"], x["change"], x["half"], x["fwd"], x["rev"]])
    for prefix in args.show:
        for r in runs:
            if r["sha"].startswith(prefix):
                print(f"\n{r['sha'][:9]} (bench {r['bench']}{', A/A' if r['aa'] else ''}):")
                for k in ("split_cpu_total", "finalise_cpu", "measured_cpu", "measured_wall", "eta", "split_eta"):
                    if x := per_run(r, k):
                        print(f"  {k}: change {pct(x['change'])} ±{100 * x['half']:.2f}; forward "
                              f"{' '.join(f'{100 * math.expm1(v):+.1f}' for v in x['fv'])}; reverse "
                              f"{' '.join(f'{100 * math.expm1(v):+.1f}' for v in x['rv'])}; o {pct(x['o'])}")
    # The interval's calibration over the A/A runs, every figure but the duplicates: the
    # paired change over its standard error (interval / t) is t-distributed, 2 degrees of
    # freedom, if the rounds are independent and their noise normal
    t2 = {0.80: 1.886, 0.90: 2.920, 0.95: 4.303}
    ts, vr, by_run = [], {}, {}
    for k in FIGURES:
        if k in ("split_eta", "device_written_mb", "split_cpu", "orjson_split_cpu_total", "split_peak_rss_mb"):
            continue  # (the same figure twice, or not a time)
        xs = []
        for r in aa:
            if (x := per_run(r, k)) and x["n"] == 3 and x["half"] > 0:
                xs.append(x)
                by_run.setdefault(r["path"], []).append(abs(x["change"]) > x["half"])
        ts += [x["change"] / (x["half"] / 4.303) for x in xs]
        if len(xs) > 2:
            # observed variance of the changes over A/A runs, over the mean variance the
            # intervals claim (their half-width / t, squared)
            vr[k] = (statistics.fmean(x["change"] ** 2 for x in xs)
                     / statistics.fmean((x["half"] / 4.303) ** 2 for x in xs), len(xs))
    if not ts:
        return
    print(f"\nA/A calibration, {len(ts)} (run, figure) changes: " + "; ".join(
        f"|t| > {q:.2f} in {sum(abs(t) > q for t in ts) / len(ts):.1%} (expected {1 - c:.0%})" for c, q in t2.items()))
    # (a bootstrap over the A/A runs, whose figures move together, for the flagged share)
    import random
    rng, groups = random.Random(0), list(by_run.values())
    shares = sorted(sum(sum(g) for g in pick) / sum(len(g) for g in pick)
                    for pick in ([rng.choice(groups) for _ in groups] for _ in range(2000)))
    print(f"flagged at 95%: {sum(map(sum, groups))}/{sum(map(len, groups))} over {len(groups)} A/A runs, "
          f"95% bootstrap interval {shares[50]:.1%} to {shares[1949]:.1%}")
    print("observed / claimed variance of the A/A change: " + ", ".join(
        f"`{k}` {v:.2f} (n={n})" for k, (v, n) in vr.items()))
    if args.csv:
        with open(args.csv, "w") as f:
            f.write("sha,bench,aa,figure,o,change,half,forward,reverse\n")
            f.writelines(",".join(map(str, row)) + "\n" for row in rows)


if __name__ == "__main__":
    sys.exit(main())
