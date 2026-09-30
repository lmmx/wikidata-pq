"""Figures for the dataset cards, from the local copy of the Hub repos (download-wikidata).

Usage: python scripts/card_stats.py [hub_dir]   (default: hub)

1. Checks the local copy against docs/dataset_cards_metadata.json: files, bytes and rows
   (from Parquet footers) per key.
2. Lists regional language codes (keys with a hyphen) per language-split table.
3. For labels, descriptions and aliases: how many ids have any row, an `en` row, a `mul`
   row, a `mul` row but no `en` row, and neither — using one flag per id number, so
   memory stays at a few bytes per id however many keys a table has.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
METADATA = ROOT / "docs" / "dataset_cards_metadata.json"
LANGUAGE_TABLES = ("labels", "descriptions", "aliases", "claims_labels")
COVERAGE_TABLES = ("labels", "descriptions", "aliases")


def local_keys(table_dir: Path) -> dict[str, list[Path]]:
    keys: dict[str, list[Path]] = {}
    for p in sorted(table_dir.glob("*/*.parquet")):
        keys.setdefault(p.parent.name, []).append(p)
    return keys


def check_copy(hub: Path, metadata: dict) -> None:
    print("== Local copy against the metadata JSON")
    for table, meta in tqdm(metadata.items(), desc="tables", leave=False):
        keys = local_keys(hub / table)
        bad = []
        for key, m in meta.items():
            files = keys.get(key, [])
            got = {
                "files": len(files),
                "bytes": sum(p.stat().st_size for p in files),
                "rows": sum(pq.ParquetFile(p).metadata.num_rows for p in files),
            }
            if got != m:
                bad.append((key, m, got))
        extra = sorted(set(keys) - set(meta))
        status = "ok" if not bad and not extra else "MISMATCH"
        print(f"{table}: {len(keys)} local keys, {len(meta)} in metadata: {status}")
        for key, m, got in bad[:10]:
            print(f"  {key}: metadata {m}, local {got}")
        if extra:
            print(f"  local keys not in metadata: {extra[:10]}")


def regional_codes(metadata: dict) -> None:
    print("\n== Regional language codes (keys with a hyphen)")
    for table in LANGUAGE_TABLES:
        codes = sorted(k for k in metadata.get(table, {}) if "-" in k)
        print(f"{table}: {len(codes)}: {' '.join(codes)}")


def id_flags(files: list[Path], prefix: str, size: int) -> np.ndarray:
    """A bool per id number: True where one of the files has a row for {prefix}{number}."""
    flags = np.zeros(size, dtype=bool)
    nums = (
        pl.scan_parquet(files)
        .select("id")
        .filter(pl.col("id").str.starts_with(prefix))
        .select(pl.col("id").str.slice(1).cast(pl.UInt32))
        .collect(engine="streaming")
        .to_series()
        .to_numpy()
    )
    flags[nums] = True
    return flags


def max_id(files: list[Path], prefix: str) -> int:
    return (
        pl.scan_parquet(files)
        .select("id")
        .filter(pl.col("id").str.starts_with(prefix))
        .select(pl.col("id").str.slice(1).cast(pl.UInt32).max())
        .collect(engine="streaming")
        .item()
        or 0
    )


def coverage(hub: Path, table: str) -> None:
    keys = local_keys(hub / table)
    prefixes = (
        pl.scan_parquet([p for ps in keys.values() for p in ps][:50])
        .select(pl.col("id").str.head(1).unique())
        .collect()
        .to_series()
        .to_list()
    )
    print(f"\n{table}: {len(keys)} keys, id prefixes seen in the first files: {sorted(prefixes)}")
    for prefix in ("Q", "P"):
        t = time.time()
        size = max_id([p for ps in keys.values() for p in ps], prefix) + 1
        any_ = np.zeros(size, dtype=bool)
        for files in tqdm(keys.values(), desc=f"  {table} {prefix}", unit="key"):
            any_ |= id_flags(files, prefix, size)
        en = id_flags(keys["en"], prefix, size) if "en" in keys else np.zeros(size, bool)
        mul = id_flags(keys["mul"], prefix, size) if "mul" in keys else np.zeros(size, bool)
        n_any = int(any_.sum())
        if not n_any:
            continue
        pct = lambda n: f"{n:,} ({100 * n / n_any:.1f}%)"  # noqa: E731
        print(f"  {prefix} ids with a row in any key: {n_any:,}")
        print(f"    with en: {pct(int(en.sum()))}")
        print(f"    with mul: {pct(int(mul.sum()))}")
        print(f"    with mul, no en: {pct(int((mul & ~en).sum()))}")
        print(f"    with en or mul: {pct(int((mul | en).sum()))}")
        print(f"    with neither en nor mul: {pct(int((any_ & ~en & ~mul).sum()))}")
        sample = np.flatnonzero(mul & ~en)[:5]
        print(f"    mul, no en, first ids: {[f'{prefix}{n}' for n in sample]}")
        print(f"    ({time.time() - t:.0f}s)")


def main() -> None:
    hub = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "hub"
    metadata = json.loads(METADATA.read_text())
    check_copy(hub, metadata)
    regional_codes(metadata)
    print("\n== Coverage of en and mul")
    for table in COVERAGE_TABLES:
        coverage(hub, table)


if __name__ == "__main__":
    main()
