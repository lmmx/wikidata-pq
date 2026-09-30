"""Figures and sample rows for the dataset cards, from the local copy of the Hub repos
(HUB_COPY_DIR, as `download-wikidata` and the sort stage leave it).

For each table but claims: the rows of a few fixed ids (SAMPLES), shown in its card. For
each table split by language, also: how many item (`Q`) and property (`P`) ids have a
row in any key, an `en` row, a `mul` row, and a `mul` row but no `en` row, using one flag
per id number, so memory stays at a few bytes per id however many keys a table has.

The figures are written to DATASET_CARDS_STATS with a digest of what they were computed
from: the table's entry in DATASET_CARDS_METADATA and what is computed (its SAMPLES and
STATS_COLUMN entries, PREFIXES); cards.py refuses figures whose digest differs from the
current one, and `finalise` recomputes them.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq
from tqdm import tqdm

from .config import DATASET_CARDS_METADATA, DATASET_CARDS_STATS, HUB_COPY_DIR, Table

# Tables split by language, and the column holding the id each row names
STATS_COLUMN = {
    Table.LABEL: "id",
    Table.DESC: "id",
    Table.ALIAS: "id",
    Table.CLAIMS_LABELS: "ref",
}
PREFIXES = ("Q", "P")
# The rows shown in each card: the id column's values, and the keys to read them from
SAMPLES = {
    Table.LABEL: ("id", ["Q42", "Q5"], ["en", "fr", "de", "mul"]),
    Table.DESC: ("id", ["Q42", "Q5"], ["en"]),
    Table.ALIAS: ("id", ["Q42", "Q5"], ["en", "mul"]),
    Table.LINKS: ("id", ["Q42"], ["enwiki", "frwiki", "dewiki"]),
    Table.CLAIMS_LABELS: ("ref", ["P31", "Q5", "Q11573"], ["en"]),
}


def read_metadata() -> dict[str, dict[str, dict[str, int]]]:
    path = DATASET_CARDS_METADATA
    return json.loads(path.read_text()) if path.exists() else {}


def read_stats() -> dict[str, dict]:
    path = DATASET_CARDS_STATS
    return json.loads(path.read_text()) if path.exists() else {}


def inputs_digest(table: Table, keys: dict[str, dict[str, int]]) -> str:
    """sha256 of a table's metadata entry (files, bytes and rows per key) and of what
    is computed for it."""
    inputs = {
        "metadata": keys,
        "sample": SAMPLES.get(table),
        "column": STATS_COLUMN.get(table),
        "prefixes": PREFIXES,
    }
    return hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()


def current(table: Table, metadata: dict, stats: dict) -> bool:
    """Whether the table's figures were computed from its current metadata and spec."""
    entry = stats.get(str(table))
    keys = metadata.get(str(table))
    return bool(entry and keys) and entry.get("inputs_sha256") == inputs_digest(
        table, keys
    )


def _local_keys(table_dir: Path) -> dict[str, list[Path]]:
    keys: dict[str, list[Path]] = {}
    for p in sorted(table_dir.glob("*/*.parquet")):
        keys.setdefault(p.parent.name, []).append(p)
    return keys


def _check_copy(table: Table, keys: dict[str, list[Path]], meta: dict) -> None:
    """Refuse a local copy whose files, bytes or rows differ from the metadata."""
    got = {
        key: {
            "files": len(files),
            "bytes": sum(p.stat().st_size for p in files),
            "rows": sum(pq.ParquetFile(p).metadata.num_rows for p in files),
        }
        for key, files in keys.items()
    }
    if got != meta:
        bad = sorted(k for k in set(got) | set(meta) if got.get(k) != meta.get(k))
        raise RuntimeError(
            f"[card-stats] {table}: local copy differs from the metadata in "
            f"{len(bad)} keys (e.g. {bad[:5]}): run download-wikidata"
        )


def _id_nums(files: list[Path], col: str, prefix: str) -> np.ndarray:
    return (
        pl.scan_parquet(files)
        .select(col)
        .filter(pl.col(col).str.starts_with(prefix))
        .select(pl.col(col).str.slice(1).cast(pl.UInt32))
        .collect(engine="streaming")
        .to_series()
        .to_numpy()
    )


def _flags(files: list[Path] | None, col: str, prefix: str, size: int) -> np.ndarray:
    """A bool per id number: True where one of the files has a row for {prefix}{number}."""
    flags = np.zeros(size, dtype=bool)
    if files:
        flags[_id_nums(files, col, prefix)] = True
    return flags


def _sample(table: Table, keys: dict[str, list[Path]]) -> list[dict[str, str]]:
    """The rows of the table's SAMPLES ids in its SAMPLES keys, in the order listed."""
    col, ids, sample_keys = SAMPLES[table]
    missing = [k for k in sample_keys if k not in keys]
    if missing:
        raise RuntimeError(f"[card-stats] {table}: no sample keys {missing}")
    rows = []
    for id_ in ids:
        for key in sample_keys:
            lf = pl.scan_parquet(keys[key]).filter(pl.col(col) == id_)
            rows += lf.collect().to_dicts()
    found = {r[col] for r in rows}
    if found != set(ids):
        raise RuntimeError(f"[card-stats] {table}: no rows for {set(ids) - found}")
    return rows


def _coverage(table: Table, keys: dict[str, list[Path]]) -> dict[str, dict[str, int]]:
    col = STATS_COLUMN[table]
    every = [p for files in keys.values() for p in files]
    coverage = {}
    for prefix in PREFIXES:
        top = (
            pl.scan_parquet(every)
            .select(col)
            .filter(pl.col(col).str.starts_with(prefix))
            .select(pl.col(col).str.slice(1).cast(pl.UInt32).max())
            .collect(engine="streaming")
            .item()
        )
        size = (top or 0) + 1
        any_ = np.zeros(size, dtype=bool)
        for files in tqdm(
            keys.values(), desc=f"[card-stats] {table} {prefix}", unit="key"
        ):
            any_ |= _flags(files, col, prefix, size)
        en = _flags(keys.get("en"), col, prefix, size)
        mul = _flags(keys.get("mul"), col, prefix, size)
        coverage[prefix] = {
            "any": int(any_.sum()),
            "en": int(en.sum()),
            "mul": int(mul.sum()),
            "mul_no_en": int((mul & ~en).sum()),
        }
    return coverage


def compute_stats(table: Table, hub_dir: Path = HUB_COPY_DIR) -> dict:
    """The table's sample rows and coverage figures, from hub_dir/{table} checked
    against the metadata."""
    meta = read_metadata().get(str(table))
    if not meta:
        raise RuntimeError(f"[card-stats] {table}: not in {DATASET_CARDS_METADATA}")
    keys = _local_keys(hub_dir / table)
    _check_copy(table, keys, meta)
    entry: dict = {"inputs_sha256": inputs_digest(table, meta)}
    entry["sample"] = _sample(table, keys)
    if table in STATS_COLUMN:
        entry["coverage"] = _coverage(table, keys)
    return entry


def update_stats(tables=tuple(SAMPLES), hub_dir: Path = HUB_COPY_DIR) -> None:
    """Recompute the figures of each table whose figures are not current."""
    metadata = read_metadata()
    for table in tables:
        stats = read_stats()
        if current(table, metadata, stats):
            print(f"[card-stats] {table}: figures current", flush=True)
            continue
        stats[str(table)] = compute_stats(table, hub_dir)
        ordered = {str(t): stats[str(t)] for t in Table if str(t) in stats}
        DATASET_CARDS_STATS.write_text(json.dumps(ordered, indent=2) + "\n")
        print(f"[card-stats] {table}: figures written", flush=True)
