#!/usr/bin/env python3
"""Test push.merge_group on partition files of synthetic chunks (no Hub).

- claims: merged by copying the partition files' encoded row groups (pq_glue.concat).
  The group file has the rows a Polars merge of the same files has (fingerprint), the
  same Arrow schema with its metadata, the partition files' row groups, each with its
  source's statistics, and the partition files are deleted once it is checked
- labels: still merged by Polars' sink, a file per language, the same rows
- a claims partition file with another schema: the merge raises, no group file is left,
  and the partition files are kept for the group's resumption

Runs in a temporary directory.

Usage: python scripts/test_merge.py
"""

import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from test_claims_explode import synthetic_claims  # noqa: E402

import wikidata.push.core as core  # noqa: E402
from wikidata.compact import _fingerprint  # noqa: E402
from wikidata.config import PARTITION_COLS, Table  # noqa: E402
from wikidata.partitioning import partition_parquet, prepare_for_partition  # noqa: E402
from wikidata.push.groups import Group  # noqa: E402

CHUNKS = 3


def check(name: str, ok: bool) -> None:
    print(f"{'ok  ' if ok else 'FAIL'} {name}", flush=True)
    if not ok:
        sys.exit(1)


def labels(i: int) -> pl.DataFrame:
    """Chunk i's labels in the processed table's layout: a map of language to label."""
    return pl.DataFrame(
        {
            "id": [f"Q{i}-{r}" for r in range(50)],
            "labels": [[{"key": lang, "value": f"label {r} {lang}"} for lang in ("en", "de", "fr")[: 1 + r % 3]] for r in range(50)],
        }
    )


def partition(i: int, results: Path, audit: Path) -> None:
    """Chunk i's claims and labels partitioned as partition_chunk does it."""
    name = f"chunk_{i}.parquet"
    for table, frame in ((Table.CLAIMS, synthetic_claims(400, seed=i)), (Table.LABEL, labels(i))):
        src = results / table / name
        src.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(src)
        partition_parquet(
            by=PARTITION_COLS[table],
            lf=prepare_for_partition(src, table),
            source_name=name,
            dst_dir=results / table,
            log_dir=audit / table,
            arrow=table == Table.CLAIMS,
        )
        src.unlink()


def stats(path) -> list:
    md = pq.read_metadata(path)
    return [md.row_group(g).column(c).statistics for g in range(md.num_row_groups) for c in range(md.num_columns)]


def main() -> None:
    with TemporaryDirectory() as tmp:
        os.chdir(tmp)
        results, audit, staging = Path("results").resolve(), Path("audit").resolve(), Path("staging").resolve()
        for i in range(CHUNKS):
            partition(i, results, audit)
        group = Group(0, CHUNKS - 1, CHUNKS - 1)
        claims = sorted((results / Table.CLAIMS / "all").glob("chunk_*.parquet"), key=lambda p: int(p.stem.split("_")[1]))
        want = Path("want.parquet")
        pl.scan_parquet(claims).sink_parquet(want)
        want_stats = [s for p in claims for s in stats(p)]
        want_rgs = sum(pq.read_metadata(p).num_row_groups for p in claims)
        want_schema = pq.read_schema(claims[0])
        langs = {p.parent.name: pl.read_parquet(sorted(p.parent.glob("chunk_*.parquet"))) for p in (results / Table.LABEL).rglob("chunk_0.parquet")}

        core.Table = [Table.CLAIMS, Table.LABEL]  # the tables partitioned here
        core.merge_group(group, audit, staging)
        got = core.staged_path(Table.CLAIMS, "all", group, staging)
        check("claims group file has a Polars merge's rows, in order", _fingerprint(pl.scan_parquet(got)) == _fingerprint(pl.scan_parquet(want)))
        check(
            "claims group file has the partition files' Arrow schema and metadata",
            pq.read_schema(got).equals(want_schema, check_metadata=True),
        )
        check(f"claims group file has the partition files' {want_rgs} row groups", pq.read_metadata(got).num_row_groups == want_rgs)
        check("claims group file has each row group's source statistics", stats(got) == want_stats)
        check("claims partition files deleted once merged", not any((results / Table.CLAIMS / "all").glob("chunk_*.parquet")))
        merged = {p.parent.name: pl.read_parquet(p) for p in (staging / Table.LABEL).rglob("*.parquet")}
        check(
            f"labels merged by Polars, a file per language ({len(merged)})",
            set(merged) == set(langs) and all(merged[k].equals(langs[k]) for k in langs),
        )

        # A claims partition file with another schema: the merge fails, nothing is staged
        for d in (results, audit, staging):
            for p in sorted(d.rglob("*"), reverse=True):
                p.unlink() if p.is_file() else p.rmdir()
        for i in range(CHUNKS):
            partition(i, results, audit)
        odd = results / Table.CLAIMS / "all" / "chunk_1.parquet"
        t = pq.read_table(odd)
        pq.write_table(t.append_column("extra", pa.array([1] * t.num_rows)), odd)
        try:
            core.merge_group(group, audit, staging)
            failed = False
        except ValueError as e:
            failed = "schema" in str(e)
        check("a claims partition file with another schema fails the merge", failed)
        check("no claims group file is left by the failed merge", not any((staging / Table.CLAIMS).rglob("*.parquet")) if (staging / Table.CLAIMS).exists() else True)
        check("the partition files are kept for the group's resumption", len(list((results / Table.CLAIMS / "all").glob("chunk_*.parquet"))) == CHUNKS)
        os.chdir("/")
    print("all passed")


if __name__ == "__main__":
    main()
