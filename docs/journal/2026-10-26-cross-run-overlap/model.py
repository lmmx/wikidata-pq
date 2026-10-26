"""Cross-run overlap in finalise, modelled on bench 17.4.0's release wall
(bench/overlap.py: `_phase`, `_uploads_end`, `_main_uploads_end`, `finalise_link`), from
master's own runs on record.

Inputs: master's bench record (296adf4, bench 17.4.0, quick tier;
`runs/quick/296adf44ea33d5a65912af4e2f8d32829dc3b328.json` on the bench-results branch),
each "after" arm's run, medians over them. Every figure printed is modelled, none
measured. Release seconds, bytes, bytes a second.

Master (L1a and L2 in): `just release` runs finalise three times, main, scholar, main,
each its own process. Each process ends only when its committer has caught up (its last
commit and verify), then the cards. The next starts from cold: its compaction's download
and its sort, the link idle until its first table is written. Within a run, L1a's
committer commits table N while table N+1 is written; across runs, nothing overlaps.

The gaps measured here, at each boundary between runs (`gaps`):
- **the CPU idle**, from the run's writer's last table written to the run's end (its
  committer's tail), when the next run's compute could have started;
- **the link idle**, from the run's last commit to the next run's first commit, when the
  next run's downloads and sorts could have run under the last commits.

The orchestrations (`release`):
- `today`: master, bench 17.4.0's model to the second.
- `release only`: `xrun` between the second and third runs only, as `release` alone
  could run it, the first run (L2's early finalise) ending inside `overlap-release`.
- `xrun`: the next run's writer starts once the run before has written its last table
  (and, for the scholarly run, once its own set's pushes have ended: finalise's guard),
  its commits queued on the one link behind the run before's (FIFO). claims_labels' build
  still waits for the set's own claims and labels to be done, and for the other set's
  labels to be done (claims_labels.py reads both sets' sorted labels).
  - `gate`: a table to write not started while the other set's finalise has more than
    this written and not yet committed (WIKIDATA_FINALISE_CROSS_RUN_GB), on top of each
    run's own L1a bound (64 GB at master).
- One process with one committer is the same timeline at a release's scale (the
  processes' start-up and ledger reads are seconds), so it isn't a separate row.

Finalise's peak on disk (`disk`, an estimate): the most, at any table's write end, of the
table's working set (claims 3x its group bytes: sources, fragments, parts; another table
2x; claims_labels 2x its published bytes) plus every table written and not yet committed
at 2x its published bytes (its sorted files and its sources in the local copy) plus every
table committed, at its published bytes (its local copy; claims' none, deleted after its
refs). Not the dump, nor kept group files.

Usage: python model.py runs/quick/296adf44ea33d5a65912af4e2f8d32829dc3b328.json
"""

import json
import statistics
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parents[3] / "bench"
sys.path.insert(0, str(BENCH))
import overlap as ov  # noqa: E402
from link import CLAIMS_LABELS_BUILD, DOWNLOAD_RATE  # noqa: E402
from release import RELEASE_SOURCE_BYTES  # noqa: E402

H = 3600
RATES = (7e6, 15e6, 30e6, 60e6, 120e6)
RUNS = (("main", False), ("scholar", True), ("main", True))  # (set, builds claims_labels)


def hms(s: float) -> str:
    s = round(s)
    return f"{s // H}h{s % H // 60:02d}m"


def inputs(run: dict):
    """finalise_link's tables: each set's group bytes, published bytes and compute."""
    lb = run["finalise_link_bytes"]
    scale = {k: RELEASE_SOURCE_BYTES[k] / n for k, n in lb["sample"].items() if n}
    group = {k: {t: b * scale[k] for t, b in v.items()} for k, v in lb["group"].items()}
    pub = {k: {t: b * scale[k] for t, b in v.items()} for k, v in lb["published"].items()}
    fin, share = run["finalise_eta"], run["finalise_claims_share"]
    claims = sum(g.get("claims", 0) for g in group.values())
    rest = sum(b for g in group.values() for t, b in g.items() if t != "claims")
    cpu = {k: {t: fin * (share * b / claims if t == "claims" else (1 - share) * b / rest)
               for t, b in g.items()} for k, g in group.items()}
    tables = {k: [t for t in ov.FINALISE_TABLES if t in group[k]] for k in group}
    return lb, group, pub, cpu, tables


def release(run: dict, rate: float, mode: str, kept: bool | None = None, gate: float = float("inf")) -> dict:
    """The release wall and finalise's timeline, `mode` `today` or `xrun`. `gate`: under
    `xrun`, a run starts no table while the other set's finalise has more than this
    written and not yet committed (on top of its own L1a bound)."""
    lb, group, pub, cpu, tables = inputs(run)
    kept = lb["kept"] if kept is None else kept
    pending_bound = lb["pending"]
    ph = ov._phase(run, True)
    uploads = run["upload_eta_modelled"] * 7e6 / rate
    uploaded = ov._uploads_end(ph, uploads)
    if lb.get("early"):
        start = max(ph.split_end, ph.compute_end, ov._main_uploads_end(ph, uploads))
        free = max(start, uploaded)  # (the scholarly set's last pushes, FIFO before its commits)
    else:
        start = max(ph.split_end, ph.compute_end, uploaded)
        free = start
    t = start  # the writer's clock (each run's, one after another)
    done: dict[tuple[str, str], float] = {}
    queued: list[tuple[float, float]] = []  # (committed at, bytes), every run's
    marks = []  # per run: (start, its writer's end, its first commit's start, its end)
    written: list[tuple[str, str, float, float]] = []  # (set, table, committed at, bytes)
    disk = 0.0  # finalise's peak on disk (`ws`, `held`: an estimate, below)
    for i, (k, cl) in enumerate(RUNS):
        if (mode == "today" and i) or (mode == "release only" and i == 1):
            t = free = max(t, free)  # (the next process once this one has ended)
        if i == 2:  # (the main set's second process once its first has ended: its lock)
            t = max(t, marks[0][3])
        if k == "scholar":
            t = max(t, uploaded)  # (finalise's guard: its own set's groups all pushed)
        run_start, first = t, None
        mine: list[tuple[float, float]] = []
        names = (tables[k] if i < 2 else []) + (["claims_labels"] if cl else [])
        for name in names:
            if name == "claims_labels":
                other = "scholar" if k == "main" else "main"
                t = max(t, done.get((k, "claims"), t), done.get((k, "labels"), t), done.get((other, "labels"), t))
                t += CLAIMS_LABELS_BUILD / 2
            else:
                others = [(c, b) for kk, _, c, b in written if kk != k] if mode != "today" else []
                while ((pending_bound and sum(b for c, b in mine if c > t) > pending_bound)
                       or sum(b for c, b in others if c > t) > gate):
                    t = min(c for c, _ in mine + others if c > t)
                down = 0.0 if kept else group[k].get(name, 0) / DOWNLOAD_RATE
                t += down + cpu[k].get(name, 0)
            ws = (3 if name == "claims" else 2) * group[k].get(name, 0) if name != "claims_labels" else 2 * pub[k].get(name, 0)
            held = sum((2 * b if c > t else (0 if n == "claims" else b)) for kk, n, c, b in written)
            disk = max(disk, held + ws)
            up = pub[k].get(name, 0) / rate
            s = max(free, t)
            first = s if first is None else first
            free = s + up
            done[(k, name)] = free
            if k == "main" and name == "claims" and not cl:
                free += run.get("refs_eta") or 0
            queued.append((free, pub[k].get(name, 0)))
            mine.append((free, pub[k].get(name, 0)))
            written.append((k, name, free, pub[k].get(name, 0)))
        marks.append((run_start, t, first, max(t, free)))
    end = max(t, free)
    return {"wall": end, "start": start, "marks": marks, "finalise": end - start, "disk": disk}


def main(path: str) -> None:
    runs = [r for rec in json.load(open(path)) for r in rec["runs"] if r["arm"] == "after"]

    def med(f) -> float:
        return statistics.median(f(r) for r in runs)

    def row(label, cells):
        print(f"| {label} | " + " | ".join(cells) + " |")

    head = "| | " + " | ".join(f"{r / 1e6:.0f} MB/s" for r in RATES) + " |"
    sep = "|---|" + "---|" * len(RATES)
    print(f"{len(runs)} runs of master (296adf4, bench 17.4.0); medians.\n")
    print("Check: bench 17.4.0's own `release_wall_link_{r}_modelled` against `today` here:\n")
    print("| | 7 MB/s | 30 MB/s | 60 MB/s | 120 MB/s |\n|---|---|---|---|---|")
    row("bench 17.4.0", [hms(med(lambda r: r[f"release_wall_link_{x}_modelled"])) for x in (7, 30, 60, 120)])
    row("`today` here", [hms(med(lambda r: release(r, x * 1e6, "today")["wall"])) for x in (7, 30, 60, 120)])

    print("\nThe gaps at master (`today`), min:\n")
    print(head + "\n" + sep)
    for i, label in ((0, "run 1 (main) → run 2 (scholar)"), (1, "run 2 (scholar) → run 3 (main)")):
        row(f"CPU idle at the end of {label}: its committer's tail",
            [f"{med(lambda r: (lambda m: m[i][3] - m[i][1])(release(r, x, 'today')['marks'])) / 60:.0f}"
             for x in RATES])
        row(f"link idle across {label}: last commit to the next run's first",
            [f"{med(lambda r: (lambda m: m[i + 1][2] - m[i][3])(release(r, x, 'today')['marks'])) / 60:.0f}"
             for x in RATES])
    row("finalise from its start (L2: the main set's last push) to its end",
        [hms(med(lambda r: release(r, x, "today")["finalise"])) for x in RATES])

    cases = (("**today** (master: L1a, L2)", "today", float("inf")),
             ("cross-run, no gate (the next run's tables bound by its own 64 GB)", "xrun", float("inf")),
             ("cross-run, gated at the other set's 64 GB unsent", "xrun", 64e9),
             ("cross-run, gated at 32 GB", "xrun", 32e9),
             ("cross-run, gated at 16 GB", "xrun", 16e9),
             ("cross-run, gated at 0 (the next run's compute once the last commit is in)", "xrun", 0.0),
             ("cross-run in `release` only (runs 2 → 3; run 1 ends in overlap-release), 32 GB", "release only", 32e9))
    for kept in (False, True):
        print(f"\nRelease walls, L3 {'on (group files kept)' if kept else 'off (the default)'}:\n")
        print(head + "\n" + sep)
        walls = {}
        for label, mode, gate in cases:
            walls[label] = [med(lambda r: release(r, x, mode, kept, gate)["wall"]) for x in RATES]
            row(label, [hms(w) for w in walls[label]])
        print(f"\nSavings, L3 {'on' if kept else 'off'} (against today):\n")
        print(head + "\n" + sep)
        ref = walls[cases[0][0]]
        for label, _, _ in cases[1:]:
            row(label, [f"{(w - b) / 60:+.0f} min" for b, w in zip(ref, walls[label])])
        print(f"\nFinalise's peak on disk, GB (an estimate: below), L3 {'on' if kept else 'off'}:\n")
        print(head + "\n" + sep)
        for label, mode, gate in cases:
            row(label, [f"{med(lambda r: release(r, x, mode, kept, gate)['disk']) / 1e9:.0f}" for x in RATES])

if __name__ == "__main__":
    main(sys.argv[1])
