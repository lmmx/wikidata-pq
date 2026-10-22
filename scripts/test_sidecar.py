#!/usr/bin/env python3
"""Test the audit sidecar's per-file stats (partitioning/core.py) against a scan of the file.

`file_stats` reads a file's row count and its ids' min and max from the footer, and scans
the file when a statistic may be truncated or is missing. Checked on ids long enough to be
truncated, no rows, null ids, a `ref` column, several row groups, and a partition sink's
own files with their sidecar. Runs in a temporary directory.

Usage: python scripts/test_sidecar.py
"""

import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl

from wikidata.partitioning import partition_parquet
from wikidata.partitioning.core import file_stats


def check(name: str, ok: bool) -> None:
    print(f"{'ok  ' if ok else 'FAIL'} {name}", flush=True)
    if not ok:
        sys.exit(1)


def scanned(f: Path) -> tuple:
    lf = pl.scan_parquet(f)
    col = "id" if "id" in lf.collect_schema() else "ref"
    return lf.select(pl.len(), pl.col(col).min().alias("min"), pl.col(col).max()).collect().row(0)


def main() -> None:
    with TemporaryDirectory() as tmp:
        os.chdir(tmp)
        frames = {
            "ids truncated in the statistics": pl.DataFrame({"id": ["Q" + "9" * 70, "Q1"]}),
            "ids just under the truncation": pl.DataFrame({"id": ["Q" + "9" * 58, "Q" + "1" * 70]}),
            "multi-byte ids": pl.DataFrame({"id": ["é" * 40, "Q1"]}),
            "no rows": pl.DataFrame({"id": pl.Series([], dtype=pl.String)}),
            "some null ids": pl.DataFrame({"id": [None, "Q5", "Q3"]}),
            "only null ids": pl.DataFrame({"id": pl.Series([None, None], dtype=pl.String)}),
            "a ref column": pl.DataFrame({"ref": ["P31", "Q5"], "label": ["a", "b"]}),
            "several row groups": pl.DataFrame({"id": [f"Q{i}" for i in range(250_000)][::-1]}),
        }
        for name, df in frames.items():
            f = Path(f"{len(name)}.parquet")
            df.write_parquet(f, row_group_size=100_000)
            check(f"file_stats matches a scan: {name}", file_stats(f) == scanned(f))

        df = pl.DataFrame(
            {"id": [f"Q{i}" for i in range(5000)], "language": ["en", "de", "fr", "nl", "ja"] * 1000, "value": "x"}
        )
        partition_parquet(by="language", lf=df.lazy(), source_name="chunk_0.parquet", dst_dir=Path("out"), log_dir=Path("audit"))
        sidecar = pl.read_parquet("audit/chunk_0.parquet")
        want = [(p, *scanned(Path(p))) for p in sidecar["path"]]
        got = list(sidecar.select("path", "num_rows", "min_id", "max_id").iter_rows())
        check(f"partition_parquet's sidecar matches scans of its {len(got)} files", got == want)
        os.chdir("/")
    print("all passed")


if __name__ == "__main__":
    main()
