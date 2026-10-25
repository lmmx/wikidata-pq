#!/usr/bin/env python3
"""Sort a sample of claims through buckets with one commit's code and print its part
files (name, rows, bytes, sha256) and which buckets each holds, as JSON. Run once with
master's src on PYTHONPATH and once with this branch's, given master's plan (`--plan`,
its files.jsonl), to compare the part files byte for byte; and once with this branch's
own plan (no `--plan`).

The sample and settings are scripts/test_sort.py's: N_SOURCES files of slices of the
claims key's row groups, the last shuffled; buckets of a twentieth of the sample's
bytes, part files of a quarter, row groups of 4 MiB in memory.

Usage: PYTHONPATH=<src> python compare.py <claims key dir> <rows a slice> <work dir> [--plan files.jsonl]
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))


def main() -> None:
    src, rows, work = Path(sys.argv[1]).resolve(), sys.argv[2], Path(sys.argv[3]).resolve()
    plan_from = Path(sys.argv[5]).resolve() if len(sys.argv) > 5 and sys.argv[4] == "--plan" else None
    sys.argv = [sys.argv[0], str(src), rows]
    import test_sort

    work.mkdir(parents=True)
    os.chdir(work)
    os.environ["WIKIDATA_RELEASE"] = "sorttest"
    from wikidata import compact
    from wikidata import sort_by_id as sbi
    from wikidata.config import Table

    sources = test_sort.make_sample(sbi._src_dir(Table.CLAIMS) / "all")
    total = sum(p.stat().st_size for p in sources)
    sbi.SORT_BUCKET_BYTES = total // 20
    sbi.COMPACT_FILE_BYTES = total // 4
    compact.COMPACT_ROW_GROUP_BYTES = sbi.COMPACT_ROW_GROUP_BYTES = 4 * 1024**2
    sbi.SORT_BUCKET_WRITE_BYTES = 1024**2
    if plan_from is not None:
        listed = sorted((json.loads(line) for line in plan_from.read_text().splitlines() if line), key=lambda e: e["name"])
        plan = [[int(b.removeprefix("final-").removesuffix(".parquet")) for b in e["buckets"]] for e in listed]
        sbi._part_plan = lambda sources, buckets: plan
    files = sbi.sort_bucketed(Table.CLAIMS, "all", sources)
    listed = {e["name"]: e["buckets"] for e in sbi._read_jsonl(sbi._out_dir(Table.CLAIMS) / "files.jsonl")}
    out = {"files": files, "buckets": [listed[f["name"]] for f in files], "work": str(work)}
    print("RESULT " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
