"""L2 of #167: the main set's first finalise started before the build branch's uploads
end, modelled on bench 17.2.0's release wall (bench/overlap.py: `_phase`,
`_uploads_end`, `finalise_link`), from master's own runs on record.

Inputs: master's bench record (26e9259, bench 17.3.0, quick tier, CI run 40265661727;
`runs/quick/26e9259e13911801ec8984be45f5b81b281b74bb.json` on the bench-results branch),
each "after" arm's run, medians over them. Every figure printed is modelled, none
measured. Release seconds, bytes, bytes a second.

Kept from 17.2.0: the split and both sets' runs (`_phase`: when each share of the runs'
CPU time is done), the build branch's uploads on one link, finalise's three runs (main,
scholar, main) one process after another, each table downloaded (unless kept), computed,
then committed by L1a's committer, the tables' bytes and compute as `finalise_link` takes
them.

Added here, to place L2:
- **The groups by set.** 17.2.0 takes 60 groups of equal bytes and no set. Here each set
  has 30 groups of its own bytes (its group files on the sample, scaled as
  `finalise_link` scales them: main ~59 GB, scholar ~65 GB). The k-th of a set is ready
  when the runs have done k/30 of their CPU time (both sets progress together, 17.2.0's
  one rule), main's first at a tie.
- **A group's close as pool._Upload runs it** (`merge` > 0): one thread a set merges a
  group, pushes it, then takes the next (coupled; L5, merge ahead, was rejected), so a
  set's link use has a gap of a merge between its pushes. A merge takes `m_split`
  during the split and `m_after` after it (#173's estimate: ~1 min and ~0.5 min), and
  `m_fin` for the scholarly set while the main set's finalise computes (it then shares
  the cores: ~0.75 min with finalise at nice 10, as the runs are; ~6 min at nice 0,
  where the CFS gives the nice-10 merge ~7% of a core). `merge=False` is 17.2.0's rule:
  a group pushed once ready.
- **The link:** non-preemptive, a group's push once the link is free; FIFO by when the
  group is merged, unless the main set has its turn (`turn`, push/turn.py):
  - `plain`: a waiting main group pushed before a waiting scholarly one;
  - `always`: that, and the link held for the main set while it merges its next group,
    when that group is ready as the link frees (the scholarly run waits while the main
    run closes groups back to back);
  - `after split`: as `always`, once the split has ended; FIFO before (built: the main
    run holds its turn only once the split has numbered every chunk).
  Finalise's commits, while the scholarly pushes still run, take the link's idle time
  between them (as a second upload on the link would).
- **Finalise's start:**
  - `today`: once `overlap-release` has ended: the split, both sets' runs, every push.
  - `gate`: the main set's first finalise once its own run has ended (every main group
    verified), the scholarly set's run still pushing. main.finalise's guards
    (`get_next_chunk`, `unfinished_group`) read the set's own state dir already.
  - `local`: once the main set's last group has *merged*, compacting from the group files
    not yet uploaded (held for it: `tail` GB) and the build branch's (the rest,
    downloaded at its compaction); its commits once its groups are verified. Not built:
    the bound on what reading the tail locally would add.
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
A_SET = 30
ORDER = ("claims", "labels", "descriptions", "aliases", "links", "entities")
SETS = ("main", "scholar")


def hms(s: float) -> str:
    s = round(s)
    return f"{s // H}h{s % H // 60:02d}m"


def tables_of(run: dict):
    """Each set's tables' release bytes (group files, published files) and compute."""
    lb = run["finalise_link_bytes"]
    scale = {k: RELEASE_SOURCE_BYTES[k] / n for k, n in lb["sample"].items()}
    group = {k: {t: b * scale[k] for t, b in v.items()} for k, v in lb["group"].items()}
    pub = {k: {t: b * scale[k] for t, b in v.items()} for k, v in lb["published"].items()}
    fin, share = run["finalise_eta"], run["finalise_claims_share"]
    claims = sum(g["claims"] for g in group.values())
    rest = sum(b for g in group.values() for t, b in g.items() if t != "claims")
    cpu = {k: {t: fin * (share * b / claims if t == "claims" else (1 - share) * b / rest)
               for t, b in g.items()} for k, g in group.items()}
    return group, pub, cpu


class Link:
    """The upload link's busy intervals; a job takes the earliest idle time from `t`."""

    def __init__(self):
        self.busy: list[tuple[float, float]] = []

    def take(self, t: float, dur: float) -> float:
        """Fill idle time from t with dur seconds of upload (split across gaps): its end."""
        for a, b in sorted(self.busy):
            if b <= t:
                continue
            if a > t:
                use = min(dur, a - t)
                self.busy.append((t, t + use))
                dur -= use
                if dur <= 1e-9:
                    return t + use
            t = max(t, b)
        self.busy.append((t, t + dur))
        return t + dur


def build(ph, group, rate, *, merge, gate, turn=None, m_split=60.0, m_after=30.0, m_fin=45.0):
    """The build branch's pushes, each set's groups closed one at a time. Returns the
    link, each set's last push's end, the main set's last merge, and each main group's
    push end. `gate`: the main set's finalise starts at its last push (the scholarly
    merges then take m_fin)."""
    size = {k: sum(group[k].values()) / A_SET for k in SETS}
    ready = {k: [ph.closes(i / A_SET) for i in range(1, A_SET + 1)] for k in SETS}
    idx = {k: 0 for k in SETS}
    free_at = {k: 0.0 for k in SETS}  # (the set's close thread)
    merged: dict[str, float | None] = {k: None for k in SETS}
    end = {k: 0.0 for k in SETS}
    main_ends: list[float] = []
    link, link_free = Link(), 0.0
    fin_start = None
    while any(idx[k] < A_SET for k in SETS):
        for k in SETS:
            if merged[k] is None and idx[k] < A_SET:
                start = max(ready[k][idx[k]], free_at[k])
                if not merge:
                    m = 0.0
                elif fin_start is not None and k == "scholar" and start >= fin_start:
                    m = m_fin
                else:
                    m = m_split if start < ph.split_end else m_after
                merged[k] = start + m
        cands = [k for k in SETS if merged[k] is not None]
        t = max(link_free, min(merged[k] for k in cands))
        now = [k for k in cands if merged[k] <= t]
        active = turn in ("plain", "always") or (turn == "after split" and link_free >= ph.split_end)
        k = min(now, key=lambda k: ((k != "main") if active else 0, merged[k], SETS.index(k)))
        if (active and turn != "plain" and k != "main" and merged["main"] is not None
                and max(ready["main"][idx["main"]], free_at["main"]) <= link_free):
            k = "main"  # (its next group ready as the link frees: the link held through its merge)
        s = max(link_free, merged[k])
        link_free = s + size[k] / rate
        link.busy.append((s, link_free))
        free_at[k] = end[k] = link_free
        if k == "main":
            main_ends.append(link_free)
        merged[k] = None
        idx[k] += 1
        if gate and k == "main" and idx[k] == A_SET:
            fin_start = link_free
    last_merge = max(ready["main"][-1], main_ends[-2] if len(main_ends) > 1 else 0) + (m_after if merge else 0)
    return link, end, last_merge, main_ends


def one(k, tables, cl, group, pub, cpu, run, rate, kept, *, t0=0.0, link=None, local=None,
        commit_after=0.0):
    """One finalise run from t0 (finalise_link's `one`, in absolute time): its end. Its
    commits take `link`'s idle time (none busy: a link of its own); `local(t)` the share
    of a table's group files on disk at its start (no download); no commit before
    `commit_after`."""
    pending = run["finalise_link_bytes"]["pending"]
    link = link or Link()
    t = free = t0
    done, queued = {}, []
    for name in tables + (["claims_labels"] if cl else []):
        if name == "claims_labels":
            t = max(t, *(done.get(n, t0) for n in ("claims", "labels")))
            t += CLAIMS_LABELS_BUILD / 2
        else:
            while sum(b for c, b in queued if c > t) > pending:
                t = min(c for c, _ in queued if c > t)
            have = local(t) if local else 0.0
            down = 0.0 if kept else group[k].get(name, 0) * (1 - have) / DOWNLOAD_RATE
            t += down + cpu[k].get(name, 0)
        up = pub[k].get(name, 0) / rate
        free = link.take(max(free, t, commit_after), up)
        done[name] = free
        if k == "main" and name == "claims" and not cl:
            free += run.get("refs_eta") or 0
        queued.append((free, pub[k].get(name, 0)))
    return max(t, free)


MODES = {
    "today": dict(gate=False),
    "gate": dict(gate=True),
    "gate, main first": dict(gate=True, turn="plain"),
    "gate, main first, held": dict(gate=True, turn="always"),
    "gate, main first, held after the split": dict(gate=True, turn="after split"),
    "local, main first, held after the split": dict(gate=True, turn="after split"),
}


def release(run: dict, rate: float, kept: bool, mode: str, merge: bool, m_fin: float = 45.0,
            m_split: float = 60.0, m_after: float = 30.0) -> dict:
    group, pub, cpu = tables_of(run)
    ph = ov._phase(run, True)
    tbl = {k: [t for t in ORDER if t in group[k]] for k in group}
    kw = MODES[mode]
    link, end, last_merge, main_ends = build(ph, group, rate, merge=merge, m_fin=m_fin, m_split=m_split,
                                             m_after=m_after, **kw)
    base = max(ph.split_end, ph.compute_end)
    if mode == "today":
        start = max(base, end["main"], end["scholar"])
        r1 = one("main", tbl["main"], False, group, pub, cpu, run, rate, kept, t0=start)
    elif not mode.startswith("local"):
        start = max(base, end["main"])
        r1 = one("main", tbl["main"], False, group, pub, cpu, run, rate, kept, t0=start, link=link)
    else:  # (local)
        start = max(ph.split_end, last_merge)
        n = len(main_ends)
        r1 = one("main", tbl["main"], False, group, pub, cpu, run, rate, kept, t0=start, link=link,
                 local=lambda t: sum(e > t for e in main_ends) / n, commit_after=end["main"])
    tail = sum(e > start for e in main_ends) * sum(group["main"].values()) / A_SET
    r1 = max(r1, end["scholar"])  # (overlap-release ends with the scholarly run)
    r2 = one("scholar", tbl["scholar"], True, group, pub, cpu, run, rate, kept, t0=r1)
    r3 = one("main", [], True, group, pub, cpu, run, rate, kept, t0=r2)
    return {"start": start, "wall": r3, "main_end": end["main"], "scholar_end": end["scholar"],
            "compute_end": ph.compute_end, "tail_gb": tail / 1e9}


def bench_wall(run: dict, rate: float, kept: bool) -> float:
    """17.2.0's own release_wall_link at this rate (kept: with no download)."""
    r = dict(run, finalise_link_bytes=dict(run["finalise_link_bytes"], kept=kept))
    ph = ov._phase(r, True)
    up = ov._uploads_end(ph, r["upload_eta_modelled"] * 7e6 / rate)
    return max(ph.split_end, ph.compute_end, up) + ov.finalise_link(r, rate)["wall"]


def main(path: str) -> None:
    runs = [r for rec in json.load(open(path)) for r in rec["runs"] if r["arm"] == "after"]

    def med(f) -> float:
        return statistics.median(f(r) for r in runs)

    def row(label, cells):
        print(f"| {label} | " + " | ".join(cells) + " |")

    head = "| | " + " | ".join(f"{r / 1e6:.0f} MB/s" for r in RATES) + " |"
    sep = "|---|" + "---|" * len(RATES)
    print(f"{len(runs)} runs of master (26e9259); medians.\n")
    print("The build branch's uploads, a set's groups closed one at a time (merges 1 min in the split, 0.5 after):\n")
    print(head + "\n" + sep)
    for label, mode, key in (("the runs' compute ends", "today", "compute_end"),
                             ("every push ends (FIFO)", "today", "scholar_end"),
                             ("the main set's pushes end (FIFO)", "today", "main_end"),
                             ("the main set's pushes end, main first", "gate, main first", "main_end"),
                             ("the main set's pushes end, main first, held", "gate, main first, held", "main_end"),
                             ("the main set's pushes end, held after the split (built)",
                              "gate, main first, held after the split", "main_end"),
                             ("every push ends, held after the split (built)",
                              "gate, main first, held after the split", "scholar_end")):
        row(label, [hms(med(lambda r: release(r, x, False, mode, True)[key])) for x in RATES])
    row("the main set's groups not yet pushed at its last merge (GB, `local`'s tail)",
        [f"{med(lambda r: release(r, x, False, 'local, main first, held after the split', True)['tail_gb']):.1f}"
         for x in RATES])
    built = "gate, main first, held after the split"
    cases = [("today, pushed once ready (17.2.0's rule)", "today", False, 45.0),
             ("built, pushed once ready (17.2.0's rule)", built, False, 45.0),
             ("**today**", "today", True, 45.0),
             ("gate", "gate", True, 45.0),
             ("gate, main first", "gate, main first", True, 45.0),
             ("gate, main first, held from the start", "gate, main first, held", True, 45.0),
             ("**gate, main first, held after the split (built)**", built, True, 45.0),
             ("the same, finalise at nice 0 (scholarly merges 6 min)", built, True, 360.0),
             ("local, held after the split (not built: the bound)", "local, main first, held after the split",
              True, 45.0)]
    for kept in (False, True):
        print(f"\nRelease walls, L3 {'on (group files kept)' if kept else 'off (the default)'}:\n")
        print(head + "\n" + sep)
        row("bench 17.2.0's `release_wall_link`", [hms(med(lambda r: bench_wall(r, x, kept))) for x in RATES])
        walls = {}
        for label, mode, merge, m_fin in cases:
            walls[label] = [med(lambda r: release(r, x, kept, mode, merge, m_fin)["wall"]) for x in RATES]
            row(label, [hms(w) for w in walls[label]])
        print(f"\nSavings, L3 {'on' if kept else 'off'} (against `today` under the same group rule):\n")
        print(head + "\n" + sep)
        for label, mode, merge, _ in cases:
            if mode == "today":
                continue
            ref = walls["today, pushed once ready (17.2.0's rule)" if not merge else "**today**"]
            row(label, [f"{(x - b) / 60:+.0f} min" for b, x in zip(ref, walls[label])])
    print("\nSavings (L3 off) by a group's merge wall in the split (0.5 min after it), held from the start "
          "against held after the split (built):\n")
    print("| a merge in the split | " + " | ".join(f"{r / 1e6:.0f} MB/s" for r in RATES) + " |")
    print(sep)
    for m in (0.3, 1.0, 2.4, 5.0, 7.0):
        cells = []
        for x in RATES:
            b = med(lambda r: release(r, x, False, "today", True, m_split=m * 60)["wall"])
            w = [med(lambda r: release(r, x, False, mode, True, m_split=m * 60)["wall"])
                 for mode in ("gate, main first, held", built)]
            cells.append(" / ".join(f"{(v - b) / 60:+.0f}" for v in w) + " min")
        row(f"{m:g} min", cells)


if __name__ == "__main__":
    main(sys.argv[1])
