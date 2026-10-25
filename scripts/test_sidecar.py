#!/usr/bin/env python3
"""Test the audit sidecar's per-file stats (partitioning/core.py) against a scan of the file.

`file_stats` reads a file's row count and its ids' min and max from the footer, and scans
the file when a statistic may be truncated or is missing. Checked on ids long enough to be
truncated, no rows, null ids, a `ref` column, several row groups, and a partition sink's
own files with their sidecar. `partition_parquet` takes its sidecar's stats from the
frame it partitions, not the files: checked against `write_sidecar`'s from the files
on keys whose dirs sort differently as paths than as strings, ids long enough to be
truncated, multi-byte and null ids, and a `ref` column. Its one file a table
(`one_file`, a row group per key) against its file per key on the same frames, and on a
key of more rows than a row group: the sidecar the same but for its layout columns
(`path`, `file_size`, `row_groups`), each key's row groups the rows of the key's file, in
order, in as many row groups (one, for a key of a row group's rows or fewer), the
file's Arrow schema the same, and the sizes summing to
the file's. Runs in a temporary directory.

Usage: python scripts/test_sidecar.py
"""

import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from wikidata.partitioning import partition_parquet
from wikidata.partitioning.core import ARROW_ROW_GROUP_ROWS, file_stats, write_sidecar


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

        langs = ["en", "en-gb", "be", "be-tarask", "zh", "zh-hans", "zh_min_nan", "nulls"]
        ids = ["Q" + "9" * 70, "Q1", "é" * 40, None, "Q" + "1" * 58, "P31", "Q10", "Q2"]
        frames = {
            "id": pl.DataFrame({"id": ids * 3, "language": langs * 3, "value": "x"}).with_columns(
                pl.when(pl.col("language") != "nulls").then(pl.col("id")).alias("id")
            ),
            "ref": pl.DataFrame({"field": "property", "ref": ids * 3, "language": langs * 3, "label": "x"}),
        }
        for name, df in frames.items():
            partition_parquet(by="language", lf=df.lazy(), source_name="chunk_1.parquet", dst_dir=Path(name), log_dir=Path("audit"))
            write_sidecar("language", dst_dir=Path(name), source=Path("chunk_1.parquet"), log_dir=Path("footers"))
            got, want = pl.read_parquet("audit/chunk_1.parquet"), pl.read_parquet("footers/chunk_1.parquet")
            check(f"partition_parquet's sidecar is write_sidecar's from its files: {name}", got.equals(want))

        frames["a key of more rows than a row group"] = pl.DataFrame(
            {"id": [f"Q{i}" for i in range(300_000)], "language": ["en"] * 250_000 + ["de"] * 50_000, "value": "x"}
        ).sample(fraction=1.0, shuffle=True, seed=0)
        frames["no rows"] = pl.DataFrame({"id": [], "language": [], "value": []}, schema={"id": pl.String, "language": pl.String, "value": pl.String})
        layout = ["path", "file_size", "row_groups"]
        for name, df in frames.items():
            kw = dict(by="language", lf=df.lazy(), source_name="chunk_2.parquet")
            partition_parquet(**kw, dst_dir=Path("keys") / name, log_dir=Path("keys_audit") / name)
            partition_parquet(**kw, dst_dir=Path("one") / name, log_dir=Path("one_audit") / name, one_file=True)
            per_key = pl.read_parquet(Path("keys_audit") / name / "chunk_2.parquet")
            one = pl.read_parquet(Path("one_audit") / name / "chunk_2.parquet")
            check(f"one file's sidecar is a file per key's but for its layout: {name}",
                  one.drop(layout).equals(per_key.drop("path", "file_size")))
            paths = set(one["path"])
            check(f"one file for the table: {name}", len(paths) == (1 if df.height else 0))
            same = True
            for (key, path, rgs), key_file in zip(one.select("language", "path", "row_groups").iter_rows(), per_key["path"]):
                f, k = pq.ParquetFile(path), pq.ParquetFile(key_file)
                got = pa.concat_tables([f.read_row_group(g) for g in rgs])
                # A key of a row group's rows or fewer (every key of a chunk) is one row
                # group, as the sink wrote it; a larger one is split at ARROW_ROW_GROUP_ROWS,
                # where the sink split at its own morsels
                one_rg = k.metadata.num_rows <= ARROW_ROW_GROUP_ROWS
                same &= got.equals(k.read()) and (len(rgs) == k.metadata.num_row_groups if one_rg else
                                                  all(f.metadata.row_group(g).num_rows <= ARROW_ROW_GROUP_ROWS for g in rgs))
                same &= f.schema_arrow.equals(k.schema_arrow, check_metadata=True)
            check(f"each key's row groups are its file's rows, in order, in as many row groups: {name}", same)
            check(f"the sidecar's sizes sum to the file's: {name}",
                  sum(Path(p).stat().st_size for p in paths) == one["file_size"].sum())
        os.chdir("/")
    print("all passed")


if __name__ == "__main__":
    main()
