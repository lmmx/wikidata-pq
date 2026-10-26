"""What a late id costs (stdlib only): a late id (one read after its region closed) forces
a rebuild of the one part whose fixed string range holds it, from that part's own verified
output and the late rows (a sorted merge). Modelled, not measured.

For k late ids, each in a part closed early chosen at random (by bytes), the expected
number of distinct parts hit, the bytes invalidated, and the rebuild's CPU, local I/O
and upload. The part size and count: the regional 128 MiB layout (closure.py), the parts
closed before the split's end (about 75% of the 128.9 GB published). CPU a part: the
finalise's claims CPU a published byte (bench 17.4.0's tables_of: 7,016 s on 4 cores for
94.6 GB, 0.30 core-s a MB), the whole sort's cost, an upper bound for a merge. Upload: the
whole part (no deduplication; Xet's content-defined chunking would send less, unmeasured).

    python3 -I redo.py
"""

PART = 128 * 1024**2
CLOSED_EARLY = 0.75 * 128.9e9
PARTS = round(CLOSED_EARLY / PART)
CORE_S_PER_MB = 7016 * 4 / 94.6e3
RATES = (7e6, 30e6)


def main() -> None:
    print(f"{PARTS} parts of {PART / 1e6:.0f} MB closed before the split ends.\n")
    print("| late ids k | parts hit (expected) | bytes invalidated | rebuild CPU (upper) | local I/O (read + write) "
          "| upload at 7 MB/s | at 30 MB/s |")
    print("|---|---|---|---|---|---|---|")
    for k in (0, 1, 10, 100, 1000, 3700):
        hit = PARTS * (1 - (1 - 1 / PARTS) ** k)
        b = hit * PART
        cpu = b / 1e6 * CORE_S_PER_MB
        up = [b / r for r in RATES]
        print(f"| {k:,} | {hit:.1f} | {b / 1e9:.2f} GB | {cpu / 3600:.2f} core-h | {2 * b / 1e9:.1f} GB | "
              + " | ".join(f"{u / 60:.0f} min" for u in up) + " |")


if __name__ == "__main__":
    main()
