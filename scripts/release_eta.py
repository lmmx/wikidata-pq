"""Progress and ETA of a release's processing (`run-release`), across both of its sets.

Each set's chunk sizes come from its manifest (releases/{set}/data/manifest.jsonl), and
which chunks are done, and when, from the claims audit files written as each chunk is
partitioned (releases/{set}/audit/claims/chunk_{N}.parquet). The rate is source bytes per
hour since the first chunk of the set being processed; the ETA applies it to what remains
of both sets.

Usage: python scripts/release_eta.py 20260928
"""

import json
import re
import sys
import time
from pathlib import Path

RELEASES = Path(__file__).parent.parent / "releases"
CHUNK_RE = re.compile(r"chunk_(\d+)\.parquet$")


def set_progress(name: str) -> dict:
    data = RELEASES / name / "data" / "manifest.jsonl"
    sizes = {}
    if data.exists():
        for line in data.read_text().splitlines():
            if line:
                e = json.loads(line)
                sizes[e["chunk"]] = e["bytes"]
    done = {}
    for f in (RELEASES / name / "audit" / "claims").glob("chunk_*.parquet"):
        done[int(CHUNK_RE.search(f.name).group(1))] = f.stat().st_mtime
    return {"sizes": sizes, "done": done}


def main() -> None:
    release = sys.argv[1]
    sets = {"scholar": set_progress(f"{release}-scholar"), "main": set_progress(release)}
    remaining = 0
    rate = None
    for name, s in sets.items():
        total = sum(s["sizes"].values())
        done_bytes = sum(s["sizes"].get(c, 0) for c in s["done"])
        remaining += total - done_bytes
        print(
            f"{name:8} {len(s['done']):,} of {len(s['sizes']):,} chunks, "
            f"{done_bytes / 1e9:.1f} of {total / 1e9:.1f} GB"
        )
        if s["done"] and len(s["done"]) < len(s["sizes"]):
            hours = (time.time() - min(s["done"].values())) / 3600
            if hours > 0:
                rate = done_bytes / hours
    if rate:
        print(f"rate     {rate / 1e9:.2f} GB/h (the set in progress)")
        print(f"ETA      {remaining / rate:.1f} h for both sets' processing")
    else:
        print("No set in progress yet to measure a rate from")


if __name__ == "__main__":
    main()
