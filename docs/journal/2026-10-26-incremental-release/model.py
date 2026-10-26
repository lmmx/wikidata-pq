"""An incremental release's wall, modelled from master's bench 17.9.0 record (stdlib only,
with bench/overlap.py for master's own walls): results/model.txt.

    git show origin/bench-results:runs/quick/b110f78160a96ddaa771ae7a0cedaf36c07c5968.json > rec179.json
    python3 -I model.py rec179.json ../../../bench results

Every figure is modelled; the inputs measured are the bench's (per run: the split's CPU and
its bz2 decode's share, the pipeline's and finalise's ETAs, the finalise phases), this
entry's churn (results/churn.json), the hash's cost beside the decode (hash_bench.py) and
the dedup model (results/dedup.json). Each of the record's 6 runs of master (the `after`
arm) is modelled, and the mean of the walls given, as the bench gives its release walls.

**Master** is bench/overlap.py's release_wall, at 7, 15, 30, 60 and 120 MB/s (the bench's
own, 15 MB/s added the same way).

**The incremental release** (4 cores, in 10 s steps):
- **The download:** the dump at 14.5 MB/s (release.RELEASE_DOWNLOAD_RATE), as today.
- **The split** follows it: every line decoded (lbzip2) and hashed, and only the changed
  and new ones reshaped, built and written. Its CPU a dump byte is
  `split CPU × (decode share × (1 + hash) + (1 - decode share) × churn)`, with the split's
  CPU its ETA on 4 cores (`split_eta × 4`), the decode share the run's own
  (`bz2_decode_cpu / split_cpu_total`), the hash 5% of the decode (measured 3–5%), and
  `churn` the bytes-weighted share of entities changed, new or gone. It uses at most 4
  cores, and decodes no byte before it is downloaded.
- **The pipeline** for the changed rows: its release CPU (`release_pipeline_cpu`) times the
  churn, 1.5 times over (its parts' fixed costs on fewer rows), with the split.
- **The merge** (finalise, incremental): every published part rebuilt from the release
  before's sorted rows less the changed and gone ids, plus the new rows (a merge: no
  compaction, no bucket sort). Its CPU is `mu` of finalise's (`finalise_eta × 4`):
  finalise's phases (this record) put 80% of its listed CPU in what a merge keeps: the
  encode (`_write_file_parallel`), the checks (`_check_written`, `_check_sort`), the reads
  of the sorted files (`_read_key`, `_read_bucket`), `write_table` and the metadata. mu
  0.8 central, 0.6 and 1.0 the bounds. No group files: the changed rows stay on local disk
  (design C), so nothing goes to the build branch and back.
- **claims_labels:** rebuilt whole once the labels are merged, `CLAIMS_LABELS_BUILD`
  (300 s), as the bench takes it; its rows are a ref's labels, so a label edit changes only
  that ref's rows.
- **The previous release's sorted files:** read from local disk (the local copy,
  HUB_COPY_DIR); the alternative, their download at 75 MB/s (link.DOWNLOAD_RATE, the
  pipeline's own Hub use), is 29 min beside a 1h59m dump download, so hidden whenever the
  downlink takes ~33 MB/s beside the dump.
- **The upload:** the published bytes (128.9 GB) times (1 − dedup) on one link at the rate,
  each part as it is written (the background committer, L1a), FIFO.
- **I0:** the merge once the split has ended (it may run on the cores the split leaves).
- **F0, F1:** the same simulator with every entity processed (churn 1, no hash, mu 1:
  the full split, pipeline and finalise), no group files (design C), without and with the
  frontier: the incremental's own gain is I − F at the same dedup, since Xet would dedup a
  full release's unchanged pages as well (the writer is deterministic).
- **I1:** with the progressive frontier (#180, the 8 synchronised sweeps): a share of the
  merge is available once that share of the published bytes is final, by #180's closure
  curve (both sets, regional 128 MiB parts), against the split's progress. The merge takes
  the cores the split leaves, and up to `cap` cores (2, as #180's finalise) when the link
  is short of written parts, the split slowing to give them.
"""

import json
import os
import sys

DUMP = 103_488_697_311
DL = 14.5e6
PUB = 128.9e9
CL_BYTES = 4.4e9  # claims_labels of both sets (in PUB)
CL_BUILD = 300.0
CORES = 4
HASH = 0.05
RATES = (7e6, 15e6, 30e6, 60e6, 120e6)
# #180's closure (both sets, regional 128 MiB parts): the share of published bytes final
# at the fraction f of the synchronised split (results/closure.txt there)
CLOSURE = [(0.0, 0.0), (0.1, 0.0), (0.2, 0.06), (0.3, 0.15), (0.4, 0.23), (0.5, 0.32), (0.6, 0.39),
           (0.7, 0.49), (0.8, 0.59), (0.9, 0.68), (0.99, 0.75), (1.0, 1.0)]


def closure(f: float) -> float:
    for (a, x), (b, y) in zip(CLOSURE, CLOSURE[1:]):
        if f <= b:
            return x + (y - x) * (f - a) / (b - a) if b > a else y
    return 1.0


def hm(s: float) -> str:
    m = round(s / 60)
    return f"{m // 60}h{m % 60:02d}m"


def incremental(run: dict, rate: float, churn: float, dedup: float, mu: float = 0.8, frontier: bool = False,
                cap: float = 2.0, dt: float = 10.0, hash_share: float = HASH) -> dict:
    split_cpu = run["split_eta"] * CORES
    dshare = run["bz2_decode_cpu"] / run["split_cpu_total"]
    sigma = split_cpu * (dshare * (1 + hash_share) + (1 - dshare) * churn) / DUMP  # core-s a dump byte
    sigma += run["release_pipeline_cpu"] * churn * 1.5 / DUMP
    merge_cpu = mu * run["finalise_eta"] * CORES
    up_total = PUB * (1 - dedup)
    t = x = merged_cpu = up = 0.0
    split_end = cl_end = None
    while True:
        downloaded = min(DUMP, DL * (t + dt))
        # the merge's work available: I1 by the closure curve, I0 after the split
        f = x / DUMP
        avail = merge_cpu * (closure(f) if frontier else (1.0 if split_end is not None else 0.0)) - merged_cpu
        written = (PUB - CL_BYTES) * merged_cpu / merge_cpu + (CL_BYTES if cl_end is not None and t >= cl_end else 0.0)
        pending = written * (1 - dedup) - up
        # cores: the split first, up to what the download lets it decode; the merge the rest,
        # and (I1) up to `cap` when the link is short of parts, the split giving way
        want_split = min(CORES, (downloaded - x) * sigma / dt) if x < DUMP else 0.0
        m_cores = max(0.0, CORES - want_split)
        if frontier and avail > 0 and pending < rate * 600:
            m_cores = max(m_cores, min(cap, CORES))
        m_cores = min(m_cores, avail / dt) if avail > 0 else 0.0
        s_cores = min(want_split, CORES - m_cores)
        x = min(DUMP, x + s_cores * dt / sigma)
        if DUMP - x < 1e3:
            x = DUMP
        merged_cpu = min(merge_cpu, merged_cpu + m_cores * dt)
        if merge_cpu - merged_cpu < 1e-6:
            merged_cpu = merge_cpu
        if split_end is None and x >= DUMP:
            split_end = t + dt
        written = (PUB - CL_BYTES) * merged_cpu / merge_cpu
        if merged_cpu >= merge_cpu and cl_end is None:
            cl_end = t + dt + CL_BUILD  # claims_labels: built once the rest is merged
        if cl_end is not None and t + dt >= cl_end:
            written = PUB
        up = min(written * (1 - dedup), up + rate * dt)
        t += dt
        if cl_end is not None and t >= cl_end and up >= up_total - 1:
            return {"wall": t, "split_end": split_end, "merge_end": cl_end - CL_BUILD}
        if t > 4e5:
            raise RuntimeError(f"no end: x {x:.3g} split_end {split_end} merged {merged_cpu:.4g}/{merge_cpu:.4g} up {up:.4g}/{up_total:.4g} cl_end {cl_end}")


def main() -> None:
    rec, bench, out = sys.argv[1], sys.argv[2], sys.argv[3]
    sys.path.insert(0, bench)
    import overlap  # noqa: E402

    overlap.RELEASE_LINK_RATES = RATES
    runs = [u for x in json.load(open(rec)) for u in x["runs"] if u["arm"] == "after"]
    churn = json.load(open(os.path.join(out, "churn.json")))
    dd = json.load(open(os.path.join(out, "dedup.json")))
    # the churn the split processes: bytes-weighted changed + gone, plus the new entities' bytes
    weeks = ["20261012-20261019", "20261005-20261012"]
    c_bytes = sum(churn["pairs"][p]["rate_bytes"] + churn["pairs"][p]["new_bz2_bytes"] / DUMP for p in weeks) / 2
    d512 = {m: sum(dd["512"][p]["total_dedup_gb_mult"][str(m)] for p in weeks) / 2 / 124.3 for m in (1, 2, 5)}
    mean = lambda xs: sum(xs) / len(xs)
    lines = ["# An incremental release's wall (modelled)\n"]
    lines.append(f"Inputs: master's bench 17.9.0 record (b110f78, its 6 runs); churn by bytes {100 * c_bytes:.2f}% a week "
                 f"(the two weeks' mean, new entities included); dedup at 512 KiB pages "
                 + ", ".join(f"{m}× churn {100 * d512[m]:.0f}%" for m in (1, 2, 5)) + ".\n")
    dsh = mean([r["bz2_decode_cpu"] / r["split_cpu_total"] for r in runs])
    sc = mean([r["split_eta"] * CORES for r in runs])
    lines.append("## The split's CPU at a release (scaled from the bench by bz2 bytes)\n")
    lines.append(f"- split CPU: {sc / 3600:.1f} core-h ({hm(sc / CORES)} on 4 cores); the bz2 decode alone "
                 f"{100 * dsh:.0f}% of it: {dsh * sc / 3600:.1f} core-h ({hm(dsh * sc / CORES)})")
    lines.append(f"- a hashing-only pass (decode + lines + id + xxh3): {dsh * sc * (1 + HASH) / 3600:.1f} core-h, "
                 f"{hm(dsh * sc * (1 + HASH) / CORES)} on 4 cores, against the dump's download {hm(DUMP / DL)}")
    lines.append(f"- the rest (reshape, build, encode, writes): {(1 - dsh) * sc / 3600:.1f} core-h; at the measured churn "
                 f"{(1 - dsh) * sc * c_bytes / 60:.0f} core-min of it\n")

    master = [overlap.release_wall(r) for r in runs]
    head = "| | " + " | ".join(f"{r / 1e6:.0f} MB/s" for r in RATES) + " |"
    sep = "|---|" + "---|" * len(RATES)
    rows = [("master (bench 17.9.0, its model)", [mean([m[f"release_wall_link_{r / 1e6:.0f}_modelled"] for m in master])
                                                  for r in RATES])]
    res = {"master": rows[0][1]}
    # the same simulator, every entity processed (the full split and pipeline, finalise's
    # whole CPU, no hash), no group files (design C): the incremental's own share is the
    # difference at the same dedup (Xet dedups a full release's identical pages as well)
    for label, kw in [("F0: full, no group files (design C), finalise after the split, dedup {d}", {"frontier": False}),
                      ("F1: F0 + the frontier (about #180's (c)), dedup {d}", {"frontier": True})]:
        for dname, dval in (("modelled", d512[1]), ("none", 0.0)):
            walls = [mean([incremental(r, rate, 1.0, dval, mu=1.0, hash_share=0.0, **kw)["wall"] for r in runs])
                     for rate in RATES]
            name = label.format(d=f"{100 * dval:.0f}%" if dval else "none")
            rows.append((name, walls))
            res[name] = walls
    for label, kw in [
        ("I0: incremental, merge after the split, dedup {d}", {"frontier": False}),
        ("I1: I0 + the frontier (#180's sweeps), dedup {d}", {"frontier": True}),
    ]:
        for mult in (1, 2, 5):
            for dname, dval in (("modelled", d512[mult]), ("none", 0.0)):
                walls = [mean([incremental(r, rate, c_bytes * mult, dval, **kw)["wall"] for r in runs]) for rate in RATES]
                name = label.format(d=f"{100 * dval:.0f}%" if dval else "none") + f", churn {mult}×"
                rows.append((name, walls))
                res[name] = walls
    lines.append("## Release walls (modelled; means of the 6 runs)\n")
    lines.append(head)
    lines.append(sep)
    for name, walls in rows:
        lines.append(f"| {name} | " + " | ".join(hm(w) for w in walls) + " |")
    # the merge's share and mu's sensitivity at the measured churn, I1, dedup modelled
    lines.append("\n## Sensitivity (I1, measured churn, dedup modelled)\n")
    lines.append(head)
    lines.append(sep)
    for name, kw in [("mu 0.6 (the merge keeps 60% of finalise's CPU)", {"mu": 0.6}),
                     ("mu 0.8 (central)", {"mu": 0.8}),
                     ("mu 1.0 (the merge costs a whole finalise)", {"mu": 1.0}),
                     ("merge capped at 1 core while the link waits", {"cap": 1.0}),
                     ("merge up to 4 cores while the link waits", {"cap": 4.0})]:
        walls = [mean([incremental(r, rate, c_bytes, d512[1], frontier=True, **kw)["wall"] for r in runs]) for rate in RATES]
        lines.append(f"| {name} | " + " | ".join(hm(w) for w in walls) + " |")
        res["sens " + name] = walls
    # where the time goes at 60 MB/s, I0, measured churn
    one = [incremental(r, 60e6, c_bytes, d512[1]) for r in runs]
    lines.append(f"\nI0 at 60 MB/s, measured churn: split ends {hm(mean([o['split_end'] for o in one]))} (the download "
                 f"{hm(DUMP / DL)}), the merge {hm(mean([o['merge_end'] for o in one]))}, the release "
                 f"{hm(mean([o['wall'] for o in one]))}.")
    lines.append(f"Merge CPU (mu 0.8): {mean([0.8 * r['finalise_eta'] * CORES for r in runs]) / 3600:.1f} core-h; "
                 f"finalise's today {mean([r['finalise_eta'] * CORES for r in runs]) / 3600:.1f} core-h.")
    open(os.path.join(out, "model.txt"), "w").write("\n".join(lines) + "\n")
    json.dump(res, open(os.path.join(out, "model.json"), "w"), indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
