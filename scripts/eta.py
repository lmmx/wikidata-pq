"""Estimate wikidata pipeline progress and ETA from audit chunk files.

Usage: python scripts/eta.py
"""

import csv
import re
from pathlib import Path

AUDIT_DIR = Path(__file__).parent.parent / "audit"
SOURCE_SIZES = Path(__file__).parent / "source_size" / "chunk_totals.csv"

CHUNK_RE = re.compile(r"chunk_(\d+)\.parquet$")


def main():
    files = list(AUDIT_DIR.glob("*/chunk_*.parquet"))
    if not files:
        print("No audit chunk files found yet.")
        return

    # earliest chunk_0 file gives us the run's start time
    chunk0_files = [f for f in files if CHUNK_RE.search(f.name).group(1) == "0"]
    start_time = min(f.stat().st_mtime for f in chunk0_files)

    # latest chunk file (by number) gives current position and last-seen time
    by_num = {}
    for f in files:
        n = int(CHUNK_RE.search(f.name).group(1))
        mtime = f.stat().st_mtime
        if n not in by_num or mtime > by_num[n]:
            by_num[n] = mtime
    current_chunk = max(by_num)
    last_time = by_num[current_chunk]

    elapsed_hours = (last_time - start_time) / 3600
    if elapsed_hours <= 0:
        print("Not enough elapsed time yet to estimate a rate.")
        return

    sizes = {}
    with open(SOURCE_SIZES) as fh:
        for row in csv.DictReader(fh):
            sizes[int(row["chunk_index"])] = float(row["size_gb"])

    total_gb = sum(sizes.values())
    done_gb = sum(gb for idx, gb in sizes.items() if idx <= current_chunk)
    remaining_gb = total_gb - done_gb

    rate_gb_per_hour = done_gb / elapsed_hours
    eta_hours = remaining_gb / rate_gb_per_hour if rate_gb_per_hour > 0 else float("inf")

    pct = 100 * done_gb / total_gb

    print(f"Current chunk:     {current_chunk}")
    print(f"Elapsed:            {elapsed_hours:.2f} h")
    print(f"Processed:          {done_gb:.1f} GB / {total_gb:.1f} GB ({pct:.2f}%)")
    print(f"Rate:               {rate_gb_per_hour:.1f} GB/h")
    print(f"ETA:                {eta_hours:.1f} h remaining")


if __name__ == "__main__":
    main()
