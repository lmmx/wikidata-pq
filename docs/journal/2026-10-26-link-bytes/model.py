"""A release's wall with other zstd levels or encodings, modelled by bench 17.2.0's own
release model (bench/overlap.release_wall, the `release_wall_link_{r}_modelled` rows) on
master's bench run (master_runs.json: 718e028, run 40259387706, the 6 jobs' "after" arm,
the walls their mean). Its inputs are perturbed by what pages.py measured on the
quick-tier sample (group_levels.jsonl, published_levels.jsonl):

- **Bytes:** each set's and table's group files and published files on the sample
  (`finalise_link_bytes`) times the table's size at the level over its size at master's
  level. The build branch's uploads (`upload_eta_modelled`) scale with the group bytes.
- **CPU:** the extra compress CPU at the level, per sample byte of the table's files,
  times the release's bytes of it (the sample's, scaled as the model scales them), over
  the tier's workers. That is how the bench's ETAs take CPU (CPU-s a byte over the
  workers).
  - The group files are written in the split (claims_build's genson writer at zstd 1,
    the fused term and sitelink writer at 3, entities by Polars at 3), so their extra CPU
    goes on `split_eta`.
  - The published files are written in finalise's sort (zstd 3), so their extra CPU goes
    on `finalise_eta`: the claims' on its claims share, the rest on the other tables'.
  - Decompress CPU is left out. It was flat across levels in the sweep.
- **Rates:** 7, 15, 30, 60 and 120 MB/s (17.2.0 has 7, 30, 60 and 120; 15 is added here).

Nothing here is measured: every wall is modelled.

Usage: python model.py
"""

import copy
import json
import sys
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "bench"))
import overlap  # noqa: E402
from release import RELEASE_SOURCE_BYTES  # noqa: E402

RATES = (7e6, 15e6, 30e6, 60e6, 120e6)
overlap.RELEASE_LINK_RATES = RATES
MASTER_PUB = 3  # compact._write_file_parallel
MASTER_GROUP = {"claims": 1}  # genson's writer; the fused writer and Polars at 3
MASTER_GROUP_DEFAULT = 3


def load(path: Path) -> dict:
    return {(r["table"], r["level"]): r for r in map(json.loads, open(path))}


def factors(sweep: dict, levels: dict, master) -> tuple[dict, dict]:
    """By table: its bytes at its level over master's, and the extra compress CPU a byte."""
    f, d = {}, {}
    for t in {t for t, _ in sweep}:
        lv = levels.get(t, levels.get("*"))
        if lv is None:
            continue
        m, r = sweep[(t, master(t))], sweep[(t, lv)]
        f[t] = r["mb"] / m["mb"]
        d[t] = (r["compress_cpu"] - m["compress_cpu"]) / (m["mb"] * 1e6)
    return f, d


def perturb(run: dict, fp: dict, dp: dict, fg: dict, dg: dict) -> dict:
    run = copy.deepcopy(run)
    lb = run["finalise_link_bytes"]
    scale = {k: RELEASE_SOURCE_BYTES[k] / n for k, n in lb["sample"].items() if n}
    workers = run["workers"]
    run["split_eta"] += sum(dg.get(t, 0) * b * scale[k] for k, v in lb["group"].items()
                            for t, b in v.items()) / workers
    before = sum(b * scale[k] for k, v in lb["group"].items() for b in v.values())
    after = sum(b * scale[k] * fg.get(t, 1) for k, v in lb["group"].items() for t, b in v.items())
    run["upload_eta_modelled"] *= after / before
    fin, share = run["finalise_eta"], run["finalise_claims_share"]
    d_claims = sum(dp.get("claims", 0) * v.get("claims", 0) * scale[k] for k, v in lb["published"].items()) / workers
    d_rest = sum(dp.get(t, 0) * b * scale[k] for k, v in lb["published"].items()
                 for t, b in v.items() if t != "claims") / workers
    claims_cpu, rest_cpu = fin * share + d_claims, fin * (1 - share) + d_rest
    run["finalise_eta"] = claims_cpu + rest_cpu
    run["finalise_claims_share"] = claims_cpu / run["finalise_eta"]
    lb["group"] = {k: {t: b * fg.get(t, 1) for t, b in v.items()} for k, v in lb["group"].items()}
    lb["published"] = {k: {t: b * fp.get(t, 1) for t, b in v.items()} for k, v in lb["published"].items()}
    return run


def walls(runs: list[dict]) -> tuple[dict, float, float]:
    out = [overlap.release_wall(r) for r in runs]
    return ({rate: mean(o[f"release_wall_link_{rate / 1e6:.0f}_modelled"] for o in out) for rate in RATES},
            mean(o["release_published_gb_modelled"] for o in out), mean(o["release_group_gb_modelled"] for o in out))


def hms(s: float) -> str:
    s = round(s)
    return f"{s // 3600}h{s % 3600 // 60:02d}m"


def row(name: str, runs: list[dict], base: dict, fp, dp, fg, dg) -> str:
    pr = [perturb(r, fp, dp, fg, dg) for r in runs]
    w, p, g = walls(pr)
    d_split = (pr[0]["split_eta"] - runs[0]["split_eta"]) / 60
    d_fin = (pr[0]["finalise_eta"] - runs[0]["finalise_eta"]) / 60
    return (f"| {name} | {g:.1f} | {p:.1f} | {d_split:+.1f} | {d_fin:+.1f} | "
            + " | ".join(f"{(w[r] - base[r]) / 60:+.1f}" for r in RATES) + " |")


def main() -> None:
    runs = json.load(open(HERE / "master_runs.json"))["runs"]
    pub, grp = load(HERE / "published_levels.jsonl"), load(HERE / "group_levels.jsonl")
    base, p0, g0 = walls(runs)
    head = ("| case | G, GB | P, GB | split ETA, min | finalise ETA, min | "
            + " | ".join(f"{r / 1e6:.0f} MB/s" for r in RATES) + " |")
    print(head)
    print("|---" * (5 + len(RATES)) + "|")
    print(f"| master (the walls) | {g0:.1f} | {p0:.1f} | | | " + " | ".join(hms(base[r]) for r in RATES) + " |")
    none = ({}, {})

    def p(levels):
        return factors(pub, levels, lambda t: MASTER_PUB)

    def g(levels):
        return factors(grp, levels, lambda t: MASTER_GROUP.get(t, MASTER_GROUP_DEFAULT))

    cases = []
    for lv in (1, 6, 9, 12, 15, 19):
        cases.append((f"published: every table at {lv}", p({"*": lv}), none))
    for lv in (1, 6, 9, 12, 19):
        cases.append((f"published: claims at {lv}, the rest at 3", p({"*": 3, "claims": lv}), none))
    for lv in (6, 9, 12, 15, 19):
        cases.append((f"published: claims at 3, the rest at {lv}", p({"*": lv, "claims": 3}), none))
    for lv in (6, 9, 12):
        cases.append((f"published: claims at 1, the rest at {lv}", p({"*": lv, "claims": 1}), none))
    for lv in (3, 6, 9, 12, 19):
        cases.append((f"group: claims at {lv}, the rest at 3", none, g({"*": 3, "claims": lv})))
    for lv in (1, 6, 9, 12, 19):
        cases.append((f"group: claims at 1, the rest at {lv}", none, g({"*": lv, "claims": 1})))
    for lv in (6, 9):
        cases.append((f"both: claims at 1, the rest at {lv}", p({"*": lv, "claims": 1}), g({"*": lv, "claims": 1})))
    for name, (fp, dp), (fg, dg) in cases:
        print(row(name, runs, base, fp, dp, fg, dg))

    # statement_id as DELTA_BYTE_ARRAY (enc.py): the claims' bytes, no CPU (measured cheaper)
    print()
    print(head)
    print("|---" * (5 + len(RATES)) + "|")
    fpe, fge = {"claims": 1 - 3.20 / 75.67}, {"claims": 1 - 1.83 / 73.11}
    for name, a, b in (("statement_id DELTA_BYTE_ARRAY: published", fpe, {}),
                       ("statement_id DELTA_BYTE_ARRAY: group files", {}, fge),
                       ("statement_id DELTA_BYTE_ARRAY: both", fpe, fge)):
        print(row(name, runs, base, a, {}, b, {}))

    # Sensitivity: the non-claims tables gaining more at a release, whose pages are larger
    # (bysize.py: descriptions' pages of 128 KiB or more go 0.73x from 3 to 9, against
    # 0.81x over all its pages): descriptions 0.70x at 9 and 0.80x at 6, the other three
    # term and sitelink tables 10% smaller again than measured, at the same CPU
    for sweep in (pub, grp):
        for lv in (6, 9):
            for t in ("descriptions", "labels", "links", "aliases"):
                m3 = sweep[(t, 3)]["mb"]
                sweep[(t, lv)] = sweep[(t, lv)] | {"mb": m3 * (0.70 if lv == 9 else 0.80) if t == "descriptions"
                                                   else sweep[(t, lv)]["mb"] * 0.9}
    print()
    print(head)
    print("|---" * (5 + len(RATES)) + "|")
    for name, fpp, fgg in (("sensitivity: published claims at 1, the rest at 9", p({"*": 9, "claims": 1}), none),
                           ("sensitivity: group claims at 1, the rest at 9", none, g({"*": 9, "claims": 1})),
                           ("sensitivity: both, claims at 1, the rest at 9", p({"*": 9, "claims": 1}),
                            g({"*": 9, "claims": 1})),
                           ("sensitivity: both, claims at 1, the rest at 6", p({"*": 6, "claims": 1}),
                            g({"*": 6, "claims": 1}))):
        print(row(name, runs, base, *fpp, *fgg))


if __name__ == "__main__":
    main()
