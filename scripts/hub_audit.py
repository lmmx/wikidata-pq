#!/usr/bin/env python3
"""What is on the Hub for a release: every table of both sets, from the Parquet footers
only (no rows read), against what the pipeline recorded locally.

For each repo: its branches and tags; then for `main` and the release's branch
(build-{release}, if it is still there): files, keys and rows, the rows checked against
the local card metadata (docs/releases/{set}/dataset_cards_metadata.json), each key's
`part-{i}-of-{n}` files checked complete, and entities checked against the entities
routed to the set (releases/{release}/data/route.jsonl). The main set is also shown
beside 20260507 (docs/dataset_cards_metadata.json).

Usage: python scripts/hub_audit.py [RELEASE]  (default 20260928)
"""

import json
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RELEASE = sys.argv[1] if len(sys.argv) > 1 else "20260928"
os.environ["WIKIDATA_RELEASE"] = RELEASE

import pyarrow.parquet as pq  # noqa: E402
from huggingface_hub import HfApi, HfFileSystem  # noqa: E402
from tqdm import tqdm  # noqa: E402

from wikidata.config import HF_USER, Table  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SETS = {"main": ("wikidata-", RELEASE), "scholar": ("wikidata-scholar-", f"{RELEASE}-scholar")}
BRANCH = f"build-{RELEASE}"
PART = re.compile(r"part-(\d+)-of-(\d+)\.parquet$")


def routed() -> dict[str, int]:
    counts = {"main": 0, "scholar": 0}
    path = ROOT / "releases" / RELEASE / "data" / "route.jsonl"
    for line in path.read_text().splitlines():
        r = json.loads(line)
        for s in counts:
            counts[s] += (r[s] or {}).get("rows", 0)
    return counts


def local_rows(set_dir: str) -> dict[str, dict[str, int]]:
    path = ROOT / "docs" / "releases" / set_dir / "dataset_cards_metadata.json"
    meta = json.loads(path.read_text()) if path.exists() else {}
    return {t: {k: v["rows"] for k, v in keys.items()} for t, keys in meta.items()}


def footer_rows(fs: HfFileSystem, url: str) -> int:
    with fs.open(url, "rb", block_size=64 * 1024) as f:
        return pq.ParquetFile(f).metadata.num_rows


def audit(api: HfApi, fs: HfFileSystem, repo: str, rev: str) -> tuple[dict[str, int], list[str]]:
    """Rows per key at `rev`, and the problems with its file names."""
    files = [
        f.path
        for f in api.list_repo_tree(repo, repo_type="dataset", revision=rev, recursive=True)
        if f.path.endswith(".parquet")
    ]
    urls = [f"datasets/{repo}@{rev.replace('/', '%2F')}/{p}" for p in files]
    with ThreadPoolExecutor(16) as pool:
        rows = list(tqdm(pool.map(lambda u: footer_rows(fs, u), urls), total=len(urls), desc=f"{repo}@{rev}", unit="file", leave=False))
    per_key: dict[str, int] = defaultdict(int)
    parts: dict[str, list[tuple[int, int]]] = defaultdict(list)
    problems = []
    for path, n in zip(files, rows):
        key = path.split("/")[0] if "/" in path else "(top level)"
        per_key[key] += n
        m = PART.search(path)
        if m:
            parts[key].append((int(m[1]), int(m[2])))
        else:
            problems.append(f"not a sorted file name: {path}")
    for key, ps in parts.items():
        ns = {n for _, n in ps}
        if len(ns) != 1 or sorted(i for i, _ in ps) != list(range(next(iter(ns)))):
            problems.append(f"{key}: parts {sorted(ps)} are not 0..n-1 of one n")
    return dict(per_key), problems


def main() -> None:
    api, fs = HfApi(), HfFileSystem()
    expected = routed()
    previous = {t: sum(v["rows"] for v in keys.values()) for t, keys in json.loads((ROOT / "docs" / "dataset_cards_metadata.json").read_text()).items()}
    summary = []
    for s, (prefix, set_dir) in SETS.items():
        local = local_rows(set_dir)
        for table in Table:
            repo = f"{HF_USER}/{prefix}{table}"
            refs = api.list_repo_refs(repo, repo_type="dataset")
            branches = sorted(b.name for b in refs.branches)
            tags = sorted(t.name for t in refs.tags)
            print(f"\n{repo}\n  branches: {', '.join(branches)}; tags: {', '.join(tags) or 'none'}")
            for rev in ["main"] + ([BRANCH] if BRANCH in branches else []):
                per_key, problems = audit(api, fs, repo, rev)
                total = sum(per_key.values())
                want = local.get(str(table), {})
                notes = []
                if per_key != want:
                    diff = sorted(k for k in set(per_key) | set(want) if per_key.get(k) != want.get(k))
                    notes.append(f"DIFFERS from local metadata in {len(diff)} keys, e.g. " + ", ".join(f"{k}: Hub {per_key.get(k)}, local {want.get(k)}" for k in diff[:3]))
                if table == Table.ENTITIES and total != expected[s]:
                    notes.append(f"ENTITIES SHORT: {total:,} of {expected[s]:,} routed to this set")
                if s == "main" and previous.get(str(table)):
                    notes.append(f"{total / previous[str(table)]:.3f} x 20260507's {previous[str(table)]:,}")
                notes += problems[:5] + ([f"... {len(problems) - 5} more"] if len(problems) > 5 else [])
                print(f"  {rev}: {len(per_key)} keys, {total:,} rows")
                for n in notes:
                    print(f"    {n}")
                bad = bool(problems) or any("DIFFERS" in n or "SHORT" in n for n in notes)
                summary.append((repo, rev, total, "PROBLEM" if bad else "ok"))
    print("\nSummary")
    for repo, rev, total, status in summary:
        print(f"  {status:8} {repo}@{rev}: {total:,} rows")


if __name__ == "__main__":
    main()
