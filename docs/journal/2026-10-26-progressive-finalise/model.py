"""A release's wall with a synchronised split and a regional finalise, against master's,
from master's bench record and this folder's closure curves (closure.json). Every figure
printed is modelled, none measured. Stdlib only.

Inputs:
- master's bench record: bench 17.4.0 (296adf4), quick tier, `runs/quick/
  296adf44ea33d5a65912af4e2f8d32829dc3b328.json` on the bench-results branch; each
  "after" arm's run, medians over the 6;
- closure.json (closure.py): the share of each set's tables' published bytes final at
  each fraction f of a synchronised split, by layout.

Variants:
- **(a) master:** the L1b entry's model of master (L1a + L2 built, bench 17.2.0's model
  underneath: `2026-10-26-l1b-preupload-evidence/model.py`, `release(run, rate, kept, None)`).
  `a_sim` is the same release in this file's simulator (nothing closes before the split
  ends, group files round-tripped), to show what the simulator's simplifications cost.
- **(b)** a synchronised split and a regional finalise, the group files still uploaded to
  the build branch and downloaded back: a region is finalised once it is closed *and* the
  groups holding its rows are uploaded (the groups laid out by region, so each region's
  slices are downloaded once).
- **(c)** (b) without the group files: a closed region is finalised from the split's own
  files on disk and committed to the build branch region by region (`create_commit`, as
  today); a table's tail after the split is committed once that table's tail is written.
- **(d)** (c) with L1b: every part goes up as it is written (`preupload_lfs_files`), the
  tail's included.
- **c0, d0:** (c) and (d) with nothing closed before the split ends: design C (no group
  files) alone, to separate what the frontier adds from what dropping the round trip does.

The simulator (fluid, 20 s steps): 4 cores. The split's CPU is its ETA's on every core; it
uses at most its sample's cores (3.54 of 4) and runs no faster than the download (103.5 GB
at 14.5 MB/s, the 8 shards' ranges fetched in step). The runs (both sets' pipelines) at
nice 10 in what the split leaves (a_sim, b), or absorbed into the split (c, d: `rho` of
their CPU kept, design A). Finalise's CPU by table and set as `tables_of` gives it,
times `eta` for the cost of many regions; finalised in table order (main claims first),
closed bytes first; during the split it takes at most `cap` cores (the bucket sorts'
memory beside the split's), all 4 after. A computed region goes onto one upload link
(FIFO, at the rate), with the group pushes in (a_sim, b); finalise waits while 64 GB is
pending (FINALISE_PENDING_UPLOAD_BYTES). claims_labels: built once everything else is
(CLAIMS_LABELS_BUILD, 300 s), then uploaded.

    python3 -I model.py rec174.json
"""

import importlib.util
import json
import os
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("l1b", HERE.parent / "2026-10-26-l1b-preupload-evidence" / "model.py")
l1b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(l1b)
l2 = l1b.l2

RATES = (7e6, 15e6, 30e6, 60e6, 120e6)
CORES = 4
DT = 20.0
SETS = ("main", "scholar")
ORDER = ("claims", "labels", "descriptions", "aliases", "links", "entities")
PENDING = 64e9
CL_BUILD = 300.0
DUMP = 103.5e9
SOURCE = 94.2e9  # both sets' chunk files at a release (RELEASE_SOURCE_BYTES)
SCRATCH = 5e9  # a region's sort fragments and parts in progress (a region is <~1% of the claims)


def hms(s: float) -> str:
    s = round(s)
    return f"{s // 3600}h{s % 3600 // 60:02d}m"


def curve_of(closure: dict, layout: str, shift: float = 0.0):
    """closed(st, table, f): the share of (st, table)'s bytes final at f (shift: the curve
    moved later by `shift` of the split, a pessimistic case)."""
    t = closure["tables"]

    def closed(st: str, table: str, f: float) -> float:
        if f >= 1.0:
            return 1.0
        f = max(0.0, f - shift)
        d = t[st][table][layout]
        g = min(99, int(f * 100))
        return d["closed"][g] / d["bytes"]

    return closed


def simulate(run: dict, rate: float, closure: dict, *, variant: str, layout: str = "regional, 128 MiB",
             rho: float = 0.5, eta: float = 1.05, cap: float = 2.0, shift: float = 0.0) -> dict:
    group, pub, cpu = l2.tables_of(run)
    pub_cl = {st: run["finalise_link_bytes"]["published"][st]["claims_labels"]
              * l2.RELEASE_SOURCE_BYTES[st] / run["finalise_link_bytes"]["sample"][st] for st in SETS}
    G = sum(sum(v.values()) for v in group.values())
    split_cpu = run["split_eta"] * CORES
    split_cores = min(CORES, run["split_cpu_total"] / run["split_wall_total"])
    runs_cpu = run["pipeline_eta"] * min(CORES, run["cpu_total"] / run["wall"])
    download = l2.ov._phase(run, True).download
    progressive = variant not in ("a_sim", "c0", "d0")
    groups = variant in ("a_sim", "b")
    stream = variant in ("d", "d0")
    if not groups:
        split_cpu += rho * runs_cpu
        runs_cpu = 0.0
    closed = curve_of(closure, layout, shift) if progressive else (lambda st, t, f: 1.0 if f >= 1 else 0.0)
    keys = [(st, t) for st in SETS for t in ORDER]
    W = {k: cpu[k[0]].get(k[1], 0) * CORES * eta for k in keys}  # core-seconds
    P = {k: pub[k[0]].get(k[1], 0) for k in keys}
    done = {k: 0.0 for k in keys}  # share computed
    held = {k: 0.0 for k in keys}  # bytes written, not yet committed (c: a region at a time)
    t = f = runs_done = 0.0
    queue = up_g = 0.0  # bytes waiting for the link; group bytes uploaded
    g_made = 0.0
    cl_left, cl_queued = CL_BUILD * CORES, False
    peak_disk = 0.0
    fin_during = 0.0
    log = []
    while True:
        dl = min(1.0, t / download)
        # the share of the dump whose rows a region's finalise can read
        avail_f = f
        if groups:
            avail_f = min(f, up_g / G if G else 1.0) if f < 1 or up_g < G - 1 else 1.0
        backlog = {k: max(0.0, closed(k[0], k[1], avail_f) - done[k]) * W[k] for k in keys}
        A = sum(backlog.values())
        fin_cap = (cap if f < 1 else CORES) if queue < PENDING else 0.0
        fin = min(fin_cap, A / DT)
        if f < 1:
            sp = min(split_cores, CORES - fin, (dl - f) * split_cpu / DT if dl < 1 else split_cores)
            sp = max(sp, 0.0)
            f = min(1.0, f + sp * DT / split_cpu)
            if f >= 1 - 1e-12:
                f = 1.0
            left = CORES - fin - sp
        else:
            left = CORES - fin
        if runs_cpu:
            rq = max(0.0, runs_cpu * f - runs_done)
            r_alloc = min(rq / DT, left if f < 1 else CORES - fin)
            runs_done += r_alloc * DT
            g_new = G * runs_done / runs_cpu - g_made
            g_made += g_new
            queue += g_new
        # finalise, in table order
        budget = fin * DT
        for k in keys:
            if budget <= 0:
                break
            if backlog[k] <= 0:
                continue
            use = min(budget, backlog[k])
            budget -= use
            share = use / W[k] if W[k] else 0.0
            done[k] += share
            b = share * P[k]
            if stream:
                queue += b
            else:
                held[k] += b
        if f < 1:
            fin_during += (fin * DT - budget) / CORES
        for k in keys:  # (c, b: a region's parts committed once the region is written)
            if not stream and held[k] and closed(k[0], k[1], avail_f) - done[k] <= 1e-9:
                queue += held[k]
                held[k] = 0.0
        # the link: group pushes and published bytes, FIFO (one queue)
        sent = min(queue, rate * DT)
        if groups:
            # (the group bytes are ahead of the published bytes made after them: fluid FIFO)
            g_wait = g_made - up_g
            g_sent = min(sent, g_wait)
            up_g += g_sent
        queue -= sent
        everything = f >= 1 and all(done[k] >= 1 - 1e-9 or W[k] == 0 for k in keys) and runs_done >= runs_cpu - 1e-6
        if everything and cl_left > 0:
            cl_left -= CORES * DT
        if everything and cl_left <= 0 and not cl_queued:
            queue += sum(held.values()) + sum(pub_cl.values())
            held = {k: 0.0 for k in keys}
            cl_queued = True
        open_src = SOURCE * max(0.0, f - sum(done[k] * P[k] for k in keys) / sum(P.values()))
        disk = DUMP + open_src + queue + sum(held.values()) + (SCRATCH if fin else 0)
        if groups:
            disk = DUMP + SOURCE * max(0.0, f - (up_g / G if G else 0)) + queue + SCRATCH
        peak_disk = max(peak_disk, disk)
        t += DT
        if len(log) == 0 or t - log[-1][0] >= 900:
            log.append((t, f, queue))
        if cl_queued and queue <= 1e-6:
            break
        if t > 3e5:
            raise RuntimeError("did not end")
    return {"wall": t, "fin_during_split_core_h": fin_during * CORES / 3600, "peak_disk_gb": peak_disk / 1e9,
            "split_end": next((x for x, ff, _ in log if ff >= 1), t)}


def main(path: str) -> None:
    runs = [r for rec in json.load(open(path)) for r in rec["runs"] if r["arm"] == "after"]
    closure = json.load(open(HERE / "closure.json"))

    def med(fn) -> float:
        return statistics.median(fn(r) for r in runs)

    head = "| | " + " | ".join(f"{r / 1e6:.0f} MB/s" for r in RATES) + " |"
    sep = "|---|" + "---|" * len(RATES)
    print(f"{len(runs)} runs of master (296adf4, bench 17.4.0); medians. Modelled, not measured.\n")
    rows = [
        ("(a) master: L1a + L2 (the L1b entry's model)", lambda r, x: l1b.release(r, x, False, None)),
        ("(a) master, L3 on (group files kept)", lambda r, x: l1b.release(r, x, True, None)),
        ("a_sim: this simulator, nothing closed early, group files round-tripped",
         lambda r, x: simulate(r, x, closure, variant="a_sim")["wall"]),
        ("(b) synchronised split, regional finalise, group round trip kept",
         lambda r, x: simulate(r, x, closure, variant="b")["wall"]),
        ("**(c) (b) without the group files**", lambda r, x: simulate(r, x, closure, variant="c")["wall"]),
        ("**(d) (c) + L1b preupload**", lambda r, x: simulate(r, x, closure, variant="d")["wall"]),
        ("c0: no group files, but nothing closed before the split ends (design C alone)",
         lambda r, x: simulate(r, x, closure, variant="c0")["wall"]),
        ("d0: c0 + L1b preupload", lambda r, x: simulate(r, x, closure, variant="d0")["wall"]),
    ]
    walls = {}
    print("Release walls:\n")
    print(head + "\n" + sep)
    for label, fn in rows:
        walls[label] = [med(lambda r: fn(r, x)) for x in RATES]
        print(f"| {label} | " + " | ".join(hms(w) for w in walls[label]) + " |")
    ref = walls[rows[0][0]]
    print("\nAgainst (a) master, L3 off:\n")
    print(head + "\n" + sep)
    for label, _ in rows[1:]:
        print(f"| {label} | " + " | ".join(f"{(w - b) / 60:+.0f} min" for b, w in zip(ref, walls[label])) + " |")

    sim = walls[rows[2][0]]
    print("\nAgainst a_sim (the same simulator, so its simplifications cancel):\n")
    print(head + "\n" + sep)
    for label, _ in rows[3:]:
        print(f"| {label} | " + " | ".join(f"{(w - b) / 60:+.0f} min" for b, w in zip(sim, walls[label])) + " |")
    print("\n(c)'s sensitivity (walls):\n")
    print(head + "\n" + sep)
    sens = [
        ("(c) central: rho 0.5, eta 1.05, cap 2 cores, regional 128 MiB parts", {}),
        ("rho 0 (the runs' CPU gone with the group files)", dict(rho=0.0)),
        ("rho 1 (all of the runs' CPU kept)", dict(rho=1.0)),
        ("eta 1.2 (regions cost 20% more finalise CPU)", dict(eta=1.2)),
        ("cap 1 core during the split", dict(cap=1.0)),
        ("cap 0.5 core during the split (finalise at nice 10)", dict(cap=0.5)),
        ("today's part boundaries (500 MiB, keys whole)", dict(layout="today (500 MiB parts)")),
        ("regional 32 MiB parts", dict(layout="regional, 32 MiB")),
        ("regional 128 MiB, every key in at least 4 files", dict(layout="regional, 128 MiB, every key in at least 4 files")),
        ("the same, and (d)'s preupload", dict(layout="regional, 128 MiB, every key in at least 4 files", variant="d")),
        ("closure 10% of the split later (pessimistic curve)", dict(shift=0.10)),
        ("closure 20% of the split later", dict(shift=0.20)),
    ]
    for label, kw in sens:
        kw = {"variant": "c", **kw}
        w = [med(lambda r: simulate(r, x, closure, **kw)["wall"]) for x in RATES]
        print(f"| {label} | " + " | ".join(hms(v) for v in w) + " |")

    print("\nPeak disk (modelled, GB: the dump, chunk files of regions not yet finalised, parts "
          "waiting for the link, a region's scratch) and finalise CPU done during the split (core-h):\n")
    print(head + "\n" + sep)
    for v in ("a_sim", "b", "c", "d", "c0"):
        cells = []
        for x in RATES:
            res = [simulate(r, x, closure, variant=v) for r in runs]
            cells.append(f"{statistics.median(q['peak_disk_gb'] for q in res):.0f} GB, "
                         f"{statistics.median(q['fin_during_split_core_h'] for q in res):.1f} core-h")
        print(f"| {v} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main(sys.argv[1])
