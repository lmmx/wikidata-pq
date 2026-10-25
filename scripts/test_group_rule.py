#!/usr/bin/env python3
"""Test that a set's published files do not depend on how its chunks are grouped: the
same chunks, run through the pipeline and finalise (bench/pipeline.py, with KEEP=1) in
groups of different sizes, publish the same files, byte for byte, key by key: finalise's
first run's sorted part files (`final/`), and its run on three times the rows
(`final_x3/`, the first run's group files three times over).

The group rule (push.groups.group_end: a fixed number of source bytes a group, in place of
a threshold projected from the set's total) changes only which chunks share a group, so
this is what it must leave alone: the group files are intermediates (each key's rows in
chunk order, whatever the groups), and the sort plans from the rows alone (#147).

Usage: python scripts/test_group_rule.py CHUNK.parquet ... [--groups 1,2,5] [--chunks N]
(the chunks the bench's tier runs: the quick tier's c0, r50, s30, s90 and 5 chunks)
"""

import argparse
import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def published(keep: Path) -> dict[str, str]:
    return {str(p.relative_to(keep)): hashlib.sha256(p.read_bytes()).hexdigest()
            for d in ("final", "final_x3") for p in sorted((keep / d).rglob("*.parquet"))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--groups", default="1,2,5", help="chunks a group, one run each")
    ap.add_argument("--chunks", type=int, default=5)
    args = ap.parse_args()
    runs = {}
    with tempfile.TemporaryDirectory() as tmp:
        for size in map(int, args.groups.split(",")):
            out = Path(tmp) / f"groups-{size}.json"
            # (the cards' figures in the run's own dir, not the checkout's docs/releases)
            env = {**os.environ, "KEEP": "1", "WIKIDATA_CARD_FIGURES_DIR": str(Path(tmp) / f"card-figures-{size}")}
            # per, workers, group size, chunks
            cmd = [sys.executable, str(ROOT / "bench" / "pipeline.py"), str(ROOT / "src"), str(out),
                   "8", "2", str(size), str(args.chunks), *map(str, map(os.path.abspath, args.files))]
            log = out.with_suffix(".log")
            with log.open("w") as f:
                code = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
            if code:
                raise SystemExit(f"groups of {size}: the run failed\n{log.read_text()[-3000:]}")
            runs[size] = published(out.with_suffix(".keep"))
            print(f"groups of {size} chunks: {sum(k.startswith('final/') for k in runs[size])} published files, "
                  f"{sum(k.startswith('final_x3/') for k in runs[size])} at three times the rows", flush=True)
        (first, ref), *rest = runs.items()
        ok = bool(ref)
        for size, got in rest:
            same = got == ref
            ok &= same
            diff = sorted(k for k in set(got) | set(ref) if got.get(k) != ref.get(k))
            print(f"{'ok  ' if same else 'FAIL'} groups of {size} against groups of {first}: "
                  f"{'every published file the same, byte for byte' if same else f'{len(diff)} differ: {diff[:5]}'}",
                  flush=True)
    if not ok:
        sys.exit(1)
    print("all passed")


if __name__ == "__main__":
    main()
