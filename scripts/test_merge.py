#!/usr/bin/env python3
"""Test push.merge_group on partition files of synthetic chunks (no Hub).

- claims: merged by copying the partition files' encoded row groups (pq_glue.concat_many).
  The group file has the rows a Polars merge of the same files has (fingerprint), the
  same Arrow schema with its metadata, the partition files' row groups, each with its
  source's statistics, and the partition files are deleted once it is checked
- labels: merged the same way, in the same call, a file per language, with the same rows
  and a row group per partition file
- a merged group merged again: nothing is redone (its partition files are gone)
- a claims partition file with another schema: the merge raises, no group file is left,
  and the partition files are kept for the group's resumption
- a labels partition file gone before its language is merged: the merge raises, naming it
- labels partitioned to one file a chunk, a row group per language (one_file): each
  language's group file has the rows, row groups and Arrow schema of the group file merged
  from a file per language; the chunks' files are deleted once every language is staged; a
  merge interrupted while deleting them resumes, deleting the rest; and a chunk's file gone
  before the merge fails it, naming the file

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


def partition(i: int, results: Path, audit: Path, one_file: bool = False) -> None:
    """Chunk i's claims and labels partitioned as partition_chunk does it (the labels to
    one file, a row group per language, with `one_file`)."""
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
            one_file=one_file and table == Table.LABEL,
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
        lang_rgs = {p.parent.name: sum(pq.read_metadata(q).num_row_groups for q in p.parent.glob("chunk_*.parquet")) for p in (results / Table.LABEL).rglob("chunk_0.parquet")}

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
            f"labels merged, a file per language ({len(merged)}), with the partition files' rows",
            set(merged) == set(langs) and all(merged[k].equals(langs[k]) for k in langs),
        )
        check(
            "each labels group file has its partition files' row groups",
            all(pq.read_metadata(core.staged_path(Table.LABEL, k, group, staging)).num_row_groups == n for k, n in lang_rgs.items()),
        )

        before = {p: p.stat().st_mtime_ns for p in staging.rglob("*.parquet")}
        core.merge_group(group, audit, staging)
        check("a merged group merged again redoes nothing", {p: p.stat().st_mtime_ns for p in staging.rglob("*.parquet")} == before)

        per_key = {k: (pl.read_parquet(p), pq.read_metadata(p).num_row_groups, pq.read_schema(p))
                   for k, p in ((p.parent.name, p) for p in (staging / Table.LABEL).rglob("*.parquet"))}

        def repartition(one_file: bool = False) -> None:
            for d in (results, audit, staging):
                for p in sorted(d.rglob("*"), reverse=True):
                    p.unlink() if p.is_file() else p.rmdir()
            for i in range(CHUNKS):
                partition(i, results, audit, one_file)

        # Labels partitioned to one file a chunk, a row group per language
        repartition(one_file=True)
        files = sorted((results / Table.LABEL).glob("chunk_*.by_language.parquet"))
        check(f"one labels file a chunk ({len(files)}), no language dirs",
              len(files) == CHUNKS and not any(p.is_dir() for p in (results / Table.LABEL).iterdir()))
        core.merge_group(group, audit, staging)
        one = {p.parent.name: p for p in (staging / Table.LABEL).rglob("*.parquet")}
        check("labels merged from one file a chunk: the same languages", set(one) == set(per_key))
        check("each language's group file has the rows a file per language gave, in order",
              all(pl.read_parquet(one[k]).equals(per_key[k][0]) for k in per_key))
        check("each has as many row groups (one a chunk)",
              all(pq.read_metadata(one[k]).num_row_groups == per_key[k][1] for k in per_key))
        check("each has the same Arrow schema and metadata",
              all(pq.read_schema(one[k]).equals(per_key[k][2], check_metadata=True) for k in per_key))
        check("the chunks' labels files deleted once every language is staged",
              not any((results / Table.LABEL).glob("*.parquet")))

        # Interrupted while deleting the chunks' files: the merge resumes, deleting the rest
        repartition(one_file=True)
        core.CLEAN_UP_LOCAL = False
        core.merge_group(group, audit, staging)
        core.CLEAN_UP_LOCAL = True
        files = sorted((results / Table.LABEL).glob("chunk_*.by_language.parquet"))
        files[0].unlink()
        before = {p: p.stat().st_mtime_ns for p in (staging / Table.LABEL).rglob("*.parquet")}
        core.merge_group(group, audit, staging)
        check("a merge interrupted while deleting resumes: nothing glued again",
              {p: p.stat().st_mtime_ns for p in (staging / Table.LABEL).rglob("*.parquet")} == before)
        check("and the chunks' files left are deleted", not any((results / Table.LABEL).glob("*.parquet")))

        # A chunk's labels file gone before the merge: it raises, naming the file
        repartition(one_file=True)
        gone = results / Table.LABEL / "chunk_1.by_language.parquet"
        gone.unlink()
        try:
            core.merge_group(group, audit, staging)
            failed = False
        except RuntimeError as e:
            failed = "partition files missing" in str(e) and str(gone) in str(e)
        check("a chunk's missing labels file fails the merge, naming it", failed)

        # A claims partition file with another schema: the merge fails, nothing is staged
        repartition()
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

        # A labels partition file gone before the merge: it raises, naming the file
        repartition()
        gone = next((results / Table.LABEL).rglob("chunk_1.parquet"))
        gone.unlink()
        try:
            core.merge_group(group, audit, staging)
            failed = False
        except RuntimeError as e:
            failed = "partition files missing" in str(e) and str(gone) in str(e)
        check("a missing labels partition file fails the merge, naming it", failed)
        os.chdir("/")
    print("all passed")


if __name__ == "__main__":
    main()
