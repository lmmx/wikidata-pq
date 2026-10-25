"""A release's critical path at master, with the transfers the bench leaves out.

Recipe order modelled: `just overlap-release` (download ∥ split ∥ both sets' runs ∥ the
build branch's uploads), then `just release` (finalise main, scholar, main; claims_labels
in the last two). Every input is a figure on record (see the journal entry); nothing is
measured here. Times in seconds, bytes in MB (10^6 B), rates in MB/s.

Usage: python critical_path.py            (prints the tables in the journal entry)
"""

H = 3600

# -- Inputs ------------------------------------------------------------------------------
# Bench 17.1.0's recomputation of #159's run 40238218967 (after = 0c61bcc; master since
# differs only in claims_labels, #164), from #165's body
SPLIT_END = 3 * H + 42 * 60 + 30  # the split's ETA (bz2, 4 cores)
DOWNLOAD_END = 1 * H + 58 * 60 + 57  # 103.5 GB of bz2 at 14.5 MB/s
COMPUTE_END = (
    4 * H + 11 * 60 + 2
)  # the runs' CPU left after the split, at the runs' cores
PIPELINE_DURING_SPLIT = 0.72  # share of the runs' CPU done during the split
FINALISE = (
    2 * H + 22 * 60 + 36
)  # finalise's ETA (three runs), 17.1.0's scholarly anchor
REFS = 20  # the claims refs' ETA
UPLOAD_7 = 4 * H + 55 * 60 + 59  # upload_eta_modelled at 7 MB/s (bench 17.0.0)
G = UPLOAD_7 * 7.0  # the build branch's bytes at a release, MB (~124 GB)
GROUPS = 60  # 30 a set (GROUP_TARGET_COUNT)

# Not in any model: the published upload, claims_labels, the build branch's download
CLAIMS_LABELS_MB = (
    5_000  # main 3.19 GB (docs/dataset_cards_metadata.json), scholar ~1-2 GB
)
P = (
    G + CLAIMS_LABELS_MB
)  # the sorted files committed (≈ G: 44.7 GB of groups → ~43 GB of parts)
CLAIMS_LABELS = 5 * 60  # its build at a release: 13-22 CPU-min (#164) on 4 cores
FIRST_PART = (
    10 * 60
)  # finalise's start to its first sorted claims part (bucketing, a wave)
LAST_PART_MB = 500  # a claims part file (COMPACT_FILE_BYTES-sized)
DOWN = 75.0  # the Hub's download rate: unknown, 50-100 MB/s assumed (E1's range)


def hms(s: float) -> str:
    s = round(s)
    return f"{s // H}h{s % H // 60:02d}m"


def group_closes() -> list[float]:
    """Each group's close: the runs' CPU share done, 72% of it evenly over the split and the
    rest evenly over the compute left after it (17.1.0's rule)."""
    out = []
    for i in range(1, GROUPS + 1):
        f = i / GROUPS
        if f <= PIPELINE_DURING_SPLIT:
            out.append(SPLIT_END * f / PIPELINE_DURING_SPLIT)
        else:
            out.append(
                SPLIT_END
                + (COMPUTE_END - SPLIT_END)
                * (f - PIPELINE_DURING_SPLIT)
                / (1 - PIPELINE_DURING_SPLIT)
            )
    return out


def uploads(rate: float) -> tuple[float, float]:
    """The build branch's uploads on one shared link, a group after the one before:
    (their end, the link's idle time from the first close to that end)."""
    end, busy = 0.0, 0.0
    closes = group_closes()
    for c in closes:
        dt = G / GROUPS / rate
        end = max(end, c) + dt
        busy += dt
    return end, end - closes[0] - busy


def uploads_by_set(
    rate: float, merge_split: float, merge_after: float, decoupled: bool
) -> float:
    """The build branch's uploads with each set's merges on its own thread, as pool.py runs
    them: a group is ready (its chunks split) at 17.1.0's close, alternating sets; its merge
    takes merge_split seconds during the split (the runs' share of the cores left at nice
    10) and merge_after after it. Coupled (today, `_Upload`: merge, push and verify in one
    thread, one group at a time), a set's next merge starts once its push ends; decoupled
    (L5), once its previous merge ends. Pushes share one FIFO link. Returns the uploads' end."""
    ready = group_closes()
    sets = {0: [], 1: []}
    for i, r in enumerate(ready):
        sets[i % 2].append(r)
    # one event loop: each set's next merge, then its push queued on the link
    link_free = 0.0
    last_merge = {0: 0.0, 1: 0.0}
    last_push = {0: 0.0, 1: 0.0}
    idx = {0: 0, 1: 0}
    end = 0.0
    while idx[0] < len(sets[0]) or idx[1] < len(sets[1]):
        best = None
        for k in (0, 1):
            if idx[k] >= len(sets[k]):
                continue
            r = sets[k][idx[k]]
            start = max(r, last_merge[k] if decoupled else last_push[k])
            m = merge_split if start < SPLIT_END else merge_after
            done = start + m
            if best is None or done < best[0]:
                best = (done, k)
        done, k = best
        last_merge[k] = done
        push_start = max(done, link_free, last_push[k])
        link_free = last_push[k] = push_start + G / GROUPS / rate
        end = max(end, link_free)
        idx[k] += 1
    return end


def release(rate: float, kept: bool, streamed: bool, early: bool) -> dict:
    """The release's wall under one orchestration.

    kept: WIKIDATA_KEEP_GROUP_FILES on (no download of G at finalise).
    streamed: each sorted part uploaded as it is written (preupload), commits as today;
        otherwise commit_table uploads a table's parts after its write, finalise waiting.
    early: finalise's compute starts at the runs' compute end, from the kept files, and
        only its commits wait for the build branch's uploads (needs kept)."""
    u_end, idle = uploads(rate)
    fin_cpu = FINALISE + REFS + CLAIMS_LABELS
    down = 0.0 if kept else G / DOWN
    start = COMPUTE_END if (early and kept) else u_end
    if not streamed:
        # today: per table, download → compute → upload, one after another
        wall = start + down + fin_cpu + P / rate
        if (
            early and kept
        ):  # only the first table's compute (main claims, ~40%) is ahead
            wall = (
                max(u_end, start + 0.4 * FINALISE)
                + 0.6 * FINALISE
                + REFS
                + CLAIMS_LABELS
                + P / rate
            )
    else:
        compute_end = start + down + fin_cpu
        link_start = max(u_end, start + (G / 2 / DOWN if not kept else 0) + FIRST_PART)
        wall = max(compute_end + LAST_PART_MB / rate, link_start + P / rate)
    return {"uploads_end": u_end, "link_idle": idle, "wall": wall}


def main() -> None:
    print("Build-branch uploads (17.1.0's close rule, one shared link):")
    print(
        "| upload rate | uploads end | past the compute's end | link idle between closes |"
    )
    print("|---|---|---|---|")
    for r in (7, 15, 30, 60, 120):
        e, idle = uploads(r)
        print(f"| {r} MB/s | {hms(e)} | +{hms(e - COMPUTE_END)} | {hms(idle)} |")
    print()
    print(
        f"G = {G / 1000:.1f} GB, P = {P / 1000:.1f} GB; finalise compute "
        f"{hms(FINALISE + REFS + CLAIMS_LABELS)} (with claims_labels' build)"
    )
    print()
    cases = [
        ("bench 17.1.0 (no finalise transfers)", None),
        (
            "today: G downloaded, commits after each table's write",
            dict(kept=False, streamed=False, early=False),
        ),
        ("L3: kept group files", dict(kept=True, streamed=False, early=False)),
        ("L1: parts uploaded as written", dict(kept=False, streamed=True, early=False)),
        ("L1 + L3", dict(kept=True, streamed=True, early=False)),
        (
            "L2 + L3: finalise's compute before the uploads end",
            dict(kept=True, streamed=False, early=True),
        ),
        ("L1 + L2 + L3", dict(kept=True, streamed=True, early=True)),
    ]
    rates = (7, 15, 30, 60, 120)
    print("| orchestration | " + " | ".join(f"{r} MB/s" for r in rates) + " |")
    print("|---|" + "---|" * len(rates))
    for name, kw in cases:
        cells = []
        for r in rates:
            if kw is None:
                u_end, _ = uploads(r)
                cells.append(hms(max(u_end, COMPUTE_END) + FINALISE + REFS))
            else:
                cells.append(hms(release(r, **kw)["wall"]))
        print(f"| {name} | " + " | ".join(cells) + " |")
    print()
    print(
        "The merge in the upload thread (uploads' end), merges after the split 1.5 min a group:"
    )
    print(
        "| a group's merge during the split | 7 MB/s, coupled (today) | 7 MB/s, decoupled (L5) | 30 MB/s, coupled | 30 MB/s, decoupled |"
    )
    print("|---|---|---|---|---|")
    for m in (0, 3, 6, 9):
        cells = [
            hms(uploads_by_set(r, m * 60, 90, d))
            for r in (7, 30)
            for d in (False, True)
        ]
        print(f"| {m} min | " + " | ".join(cells) + " |")
    print()
    floor = [hms(group_closes()[0] + (G + P) / r) for r in rates]
    print("| floor: the link alone (first close + G + P) | " + " | ".join(floor) + " |")
    cpu = hms(COMPUTE_END + FINALISE + REFS + CLAIMS_LABELS)
    print(f"| floor: the compute alone | {cpu} (every rate) |")


if __name__ == "__main__":
    main()
