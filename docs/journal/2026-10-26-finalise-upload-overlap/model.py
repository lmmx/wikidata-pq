"""Finalise's published upload overlapped with its compute (L1a), modelled at a release.

Each finalise run (`just release`: main, scholar, main) is a sequence of tables. Serial
(WIKIDATA_FINALISE_PENDING_UPLOAD_GB=0, today), each table is downloaded (compaction's
group files, unless kept), sorted, then committed and verified, one after the other.
Overlapped, one committer commits and verifies each table in order as soon as its write
is checked, while the writer goes on to the next table; the run ends when the committer
has caught up. claims_labels' build waits for the set's own labels to be done (it reads
their sorted local copy). The runs are separate processes, so nothing overlaps across
them.

Inputs are #167's (critical_path.py) and bench 17.0.0's figures on record (#164's run):
nothing is measured here. Seconds, MB (10^6 B), MB/s.

Usage: python model.py
"""

import importlib.util
from pathlib import Path

_cp = Path(__file__).parents[1] / "2026-10-25-release-critical-path" / "critical_path.py"
_spec = importlib.util.spec_from_file_location("critical_path", _cp)
cp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cp)

CLAIMS_SHARE = 0.826  # finalise_claims_share, bench 17.0.0 (#164's run)
CLAIMS_MB = 84_195  # release_claims_mb, both sets (bench 17.0.0)
MAIN_CLAIMS_MB = 43_000  # the main set's sorted claims (#150)
# The other tables' bytes at a release, by set: G less the claims (~40 GB), the main set's
# from its published tables (labels 6.5, descriptions 5.5, aliases 1.4, links 1.3 GB in
# docs/dataset_cards_metadata.json, entities ~1.5 GB), the rest the scholarly set's
OTHERS_MB = {"main": 16_200, "scholar": cp.G - CLAIMS_MB - 16_200}
OTHER_SHARE = {"labels": 0.40, "descriptions": 0.34, "aliases": 0.09, "links": 0.08, "entities": 0.09}
CL_MB = {"main": 3_190, "scholar": cp.CLAIMS_LABELS_MB - 3_190}  # claims_labels' published bytes
CL_BUILD = cp.CLAIMS_LABELS / 2  # each set's claims_labels build


def tables(set_: str) -> list[tuple[str, float, float]]:
    """(table, compute s, bytes MB) in finalise's order for a release: claims first."""
    claims_mb = MAIN_CLAIMS_MB if set_ == "main" else CLAIMS_MB - MAIN_CLAIMS_MB
    fin = cp.FINALISE  # the three runs' sort, split by bytes within claims and the rest
    claims_cpu = fin * CLAIMS_SHARE * claims_mb / CLAIMS_MB
    rest_cpu = fin * (1 - CLAIMS_SHARE) * OTHERS_MB[set_] / sum(OTHERS_MB.values())
    out = [("claims", claims_cpu, claims_mb)]
    out += [(t, rest_cpu * s, OTHERS_MB[set_] * s) for t, s in OTHER_SHARE.items()]
    return out


def run(seq, rate: float, kept: bool, overlap: bool, cl: tuple[float, float] | None,
        pending_mb: float = float("inf")) -> float:
    """One finalise run's wall. seq: its tables before claims_labels, in order; cl:
    claims_labels' (build s, bytes MB) when this run builds it. Overlapped, the writer
    starts no table while more than pending_mb is written and not yet committed, and
    claims_labels' build waits for the set's labels and claims to be done."""
    t = 0.0  # the writer's clock
    tables: list[tuple[str, float, float]] = []  # (name, committed at, MB)
    link = 0.0  # the committer's clock
    for name, cpu, mb in seq + ([("claims_labels", cl[0], cl[1])] if cl else []):
        if overlap:
            if name == "claims_labels":
                t = max([t] + [c for n, c, _ in tables if n in ("labels", "claims")])
            while sum(m for _, c, m in tables if c > t) > pending_mb:
                t = min(c for _, c, _ in tables if c > t)
        down = 0.0 if (kept or name == "claims_labels") else mb / cp.DOWN
        t += down + cpu
        if overlap:
            link = max(link, t) + mb / rate
        else:
            t = link = t + mb / rate
        tables.append((name, link, mb))
    return max(t, link)


def release(rate: float, kept: bool, overlap: bool, pending_mb: float = float("inf"),
            claims_last: bool = False) -> float:
    """The release's wall: the build branch's uploads end (critical_path.uploads), then
    finalise's three runs, one after another."""
    wall, _ = cp.uploads(rate)
    for set_, cl in (("main", None), ("scholar", (CL_BUILD, CL_MB["scholar"])), ("main", (CL_BUILD, CL_MB["main"]))):
        seq = tables(set_) if cl is None or set_ == "scholar" else []
        if claims_last:
            seq = seq[1:] + seq[:1]
        wall += run(seq, rate, kept, overlap, cl, pending_mb)
        if cl is None:
            wall += cp.REFS
    return wall


def main() -> None:
    rates = (7, 15, 30, 60, 120)
    print("| orchestration | " + " | ".join(f"{r} MB/s" for r in rates) + " |")
    print("|---|" + "---|" * len(rates))
    rows = {
        "serial (today; `=0`)": dict(kept=False, overlap=False),
        "**L1a: overlapped, 64 GB bound (the default)**": dict(kept=False, overlap=True, pending_mb=64_000),
        "L1a, no bound": dict(kept=False, overlap=True),
        "L1a, 16 GB bound": dict(kept=False, overlap=True, pending_mb=16_000),
        "L1a, claims last (`WIKIDATA_FINALISE_CLAIMS_LAST=1`)": dict(kept=False, overlap=True, pending_mb=64_000,
                                                                    claims_last=True),
        "serial, kept group files (L3)": dict(kept=True, overlap=False),
        "L1a + L3": dict(kept=True, overlap=True, pending_mb=64_000),
    }
    walls = {name: [release(r, **kw) for r in rates] for name, kw in rows.items()}
    for name, w in walls.items():
        print(f"| {name} | " + " | ".join(cp.hms(x) for x in w) + " |")
    print()
    print("| saving | " + " | ".join(f"{r} MB/s" for r in rates) + " |")
    print("|---|" + "---|" * len(rates))
    for name, ref in (("**L1a (default) against serial**", "serial (today; `=0`)"),
                      ("L1a, claims last, against serial", "serial (today; `=0`)"),
                      ("L1a + L3 against L3", "serial, kept group files (L3)")):
        key = {"**L1a (default) against serial**": "**L1a: overlapped, 64 GB bound (the default)**",
               "L1a, claims last, against serial": "L1a, claims last (`WIKIDATA_FINALISE_CLAIMS_LAST=1`)",
               "L1a + L3 against L3": "L1a + L3"}[name]
        cells = [f"−{round((b - w) / 60)} min" for b, w in zip(walls[ref], walls[key])]
        print(f"| {name} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
