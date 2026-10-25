#!/usr/bin/env python3
"""Test push.merge_group on synthetic chunks (no Hub).

- claims (unsplit): each chunk's partition file glued by copying its encoded row groups
  (pq_glue.concat_many). The group file has the rows a Polars merge of the same files has
  (fingerprint), the same Arrow schema with its metadata, the partition files' row groups,
  each with its source's statistics, and the partition files are deleted once checked
- labels (split by key): partitioned once for the group from the chunks' processed files
  (no partition file a chunk). Each language's group file has the rows, in order, and the
  Arrow schema (with its metadata) that the chunks' partition files concatenated had;
  the chunks' sidecars are partition_parquet's but for `file_size`, whose sum is the
  processed file's size; the processed files are deleted once the group files are checked
- claims_labels (deduplicated): each language's distinct rows, first occurrence first,
  as the per-key merge of partition files kept them
- in batches of one chunk (GROUP_SINK_ROWS = 1): the same rows, a row group a batch
- a merged group merged again: nothing is redone
- a merge interrupted in its sink: no marker, the processed files kept, and the rerun's
  group files are an uninterrupted merge's, byte for byte
- a merge interrupted after its marker, deleting the processed files: the rerun deletes
  the rest without partitioning again
- a claims partition file with another schema: the merge raises, no claims group file is
  left, and the partition files are kept for the group's resumption
- a processed labels file gone before the merge, or a sidecar's keys not the rows': the
  merge raises, naming the file or the keys

Runs in a temporary directory.

Usage: python scripts/test_merge.py
"""

import hashlib
import os
import pathlib
import shutil
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
from wikidata.partitioning import partition_parquet, prepare_for_partition, write_stats_sidecar  # noqa: E402
from wikidata.push.groups import Group  # noqa: E402

CHUNKS = 3
LANGS = ("en", "de", "fr", "zh", "zh-hans")
TABLES = [Table.CLAIMS, Table.LABEL, Table.CLAIMS_LABELS]


def check(name: str, ok: bool) -> None:
    print(f"{'ok  ' if ok else 'FAIL'} {name}", flush=True)
    if not ok:
        sys.exit(1)


def labels(i: int) -> pl.DataFrame:
    """Chunk i's labels in the processed table's layout: a map of language to label."""
    return pl.DataFrame(
        {
            "id": [f"Q{i}-{r}" for r in range(50)],
            "labels": [[{"key": lang, "value": f"label {r} {lang}"} for lang in LANGS[: 1 + r % len(LANGS)]] for r in range(50)],
        }
    )


def claims_labels(i: int) -> pl.DataFrame:
    """Chunk i's claims_labels, one row per (field, ref, language, label): the rows of
    refs 0-19 repeat in every chunk, so a group has them once."""
    refs = [f"Q{r}" for r in range(20)] + [f"Q{i}-{r}" for r in range(20)]
    return pl.DataFrame(
        [("labels", ref, lang, f"{ref} in {lang}") for ref in refs for lang in LANGS[: 1 + len(ref) % 3]],
        schema=["field", "ref", "language", "label"],
        orient="row",
    )


def partition(i: int, results: Path, audit: Path, ref: Path) -> None:
    """Chunk i's tables partitioned as main.partition_chunk does it: claims into its one
    partition file, labels and claims_labels kept as processed files with their
    sidecars. Each split table is also partitioned the way it was, a file per key, into
    `ref` (the reference)."""
    name = f"chunk_{i}.parquet"
    for table, frame in ((Table.CLAIMS, synthetic_claims(400, seed=i)), (Table.LABEL, labels(i)),
                         (Table.CLAIMS_LABELS, claims_labels(i))):
        src = results / table / name
        src.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(src)
        if table == Table.CLAIMS:
            partition_parquet(by=PARTITION_COLS[table], lf=prepare_for_partition(src, table), source_name=name,
                              dst_dir=results / table, log_dir=audit / table, arrow=True)
            src.unlink()
            continue
        write_stats_sidecar(PARTITION_COLS[table], prepare_for_partition(src, table), dst_dir=results / table,
                            source=Path(name), log_dir=audit / table, file_bytes=src.stat().st_size)
        partition_parquet(by=PARTITION_COLS[table], lf=prepare_for_partition(src, table), source_name=name,
                          dst_dir=ref / "results" / table, log_dir=ref / "audit" / table)


def stats(path) -> list:
    md = pq.read_metadata(path)
    return [md.row_group(g).column(c).statistics for g in range(md.num_row_groups) for c in range(md.num_columns)]


def reference(ref: Path, table: Table) -> dict[str, tuple[pl.DataFrame, pa.Schema]]:
    """Each key's rows as the per-chunk partition files' merge had them, and their schema."""
    out = {}
    for d in sorted((ref / "results" / table).iterdir()):
        files = sorted(d.glob("chunk_*.parquet"), key=lambda p: int(p.stem.split("_")[1]))
        lf = pl.scan_parquet(files)
        if table in core.DEDUPLICATE:
            lf = lf.unique(maintain_order=True)
        out[d.name] = (lf.collect(), pq.read_schema(files[0]))
    return out


def digests(staging: Path) -> dict[str, str]:
    return {str(p.relative_to(staging)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(staging.rglob("*.parquet"))}


def main() -> None:
    with TemporaryDirectory() as tmp:
        os.chdir(tmp)
        results, audit, staging = Path("results").resolve(), Path("audit").resolve(), Path("staging").resolve()
        ref = Path("ref").resolve()

        def repartition() -> None:
            for d in (results, audit, staging, ref):
                shutil.rmtree(d, ignore_errors=True)
            for i in range(CHUNKS):
                partition(i, results, audit, ref)

        def merge() -> None:
            core.merge_group(group, audit, staging, results)

        repartition()
        group = Group(0, CHUNKS - 1, CHUNKS - 1)
        claims = sorted((results / Table.CLAIMS / "all").glob("chunk_*.parquet"), key=lambda p: int(p.stem.split("_")[1]))
        want = Path("want.parquet")
        pl.scan_parquet(claims).sink_parquet(want)
        want_stats = [s for p in claims for s in stats(p)]
        want_rgs = sum(pq.read_metadata(p).num_row_groups for p in claims)
        want_schema = pq.read_schema(claims[0])
        refs = {t: reference(ref, t) for t in (Table.LABEL, Table.CLAIMS_LABELS)}
        processed = {t: [results / t / f"chunk_{i}.parquet" for i in range(CHUNKS)] for t in refs}
        processed_bytes = {t: [p.stat().st_size for p in ps] for t, ps in processed.items()}

        core.Table = TABLES  # the tables partitioned here
        merge()
        got = core.staged_path(Table.CLAIMS, "all", group, staging)
        check("claims group file has a Polars merge's rows, in order", _fingerprint(pl.scan_parquet(got)) == _fingerprint(pl.scan_parquet(want)))
        check(
            "claims group file has the partition files' Arrow schema and metadata",
            pq.read_schema(got).equals(want_schema, check_metadata=True),
        )
        check(f"claims group file has the partition files' {want_rgs} row groups", pq.read_metadata(got).num_row_groups == want_rgs)
        check("claims group file has each row group's source statistics", stats(got) == want_stats)
        check("claims partition files deleted once merged", not any((results / Table.CLAIMS / "all").glob("chunk_*.parquet")))
        for table, keys in refs.items():
            merged = {p.parent.name: p for p in (staging / table).rglob("*.parquet")}
            check(f"{table}: a group file a key ({len(merged)}), the keys of the per-chunk partition", set(merged) == set(keys))
            check(
                f"{table}: each key's rows, in order, those of its per-chunk partition files"
                + (", each distinct row once" if table in core.DEDUPLICATE else ""),
                all(pl.read_parquet(merged[k]).equals(keys[k][0]) for k in keys),
            )
            check(
                f"{table}: each group file has the partition files' Arrow schema and metadata",
                all(pq.read_schema(merged[k]).equals(keys[k][1], check_metadata=True) for k in keys),
            )
            check(f"{table}: one sink, a row group a key", all(pq.read_metadata(f).num_row_groups == 1 for f in merged.values()))
            check(f"{table}: processed files deleted once merged", not any(p.exists() for p in processed[table]))
            check(f"{table}: the table's marker written", core._marker(table, group, staging).exists())
            for i in range(CHUNKS):
                side = pl.read_parquet(audit / table / f"chunk_{i}.parquet")
                old = pl.read_parquet(ref / "audit" / table / f"chunk_{i}.parquet")
                strip = lambda d: d.drop("file_size").with_columns(pl.col("path").str.replace(r"^.*/results/", "results/"))
                check(f"{table}: chunk {i}'s sidecar is partition_parquet's but for file_size", strip(side).equals(strip(old)))
                check(f"{table}: chunk {i}'s sidecar sizes add up to its processed file",
                      side["file_size"].sum() == processed_bytes[table][i])
        check("no work dirs left", not any((staging / ".tmp" / group.name).rglob("*.parquet")))
        one_batch = digests(staging)

        before = {p: p.stat().st_mtime_ns for p in staging.rglob("*.parquet")}
        merge()
        check("a merged group merged again redoes nothing", {p: p.stat().st_mtime_ns for p in staging.rglob("*.parquet")} == before)

        # In batches of one chunk each: the same rows, a row group a chunk
        repartition()
        core.GROUP_SINK_ROWS = 1
        merge()
        core.GROUP_SINK_ROWS = 10_000_000
        for table, keys in refs.items():
            merged = {p.parent.name: p for p in (staging / table).rglob("*.parquet")}
            check(f"{table}: in batches of a chunk, the same keys and rows, in order",
                  set(merged) == set(keys) and all(pl.read_parquet(merged[k]).equals(keys[k][0]) for k in keys))
        en = core.staged_path(Table.LABEL, "en", group, staging)
        check("labels/en glued from its 3 batches' files, a row group each", pq.read_metadata(en).num_row_groups == CHUNKS)

        # Interrupted in its sink: the second labels batch fails
        repartition()
        sink, calls = core._sink_batch, []

        def failing(table, *args):
            calls.append(table)
            if table == Table.LABEL and calls.count(Table.LABEL) == 2:
                raise KeyboardInterrupt  # as a kill would leave it: the first batch's files written
            return sink(table, *args)

        core._sink_batch, core.GROUP_SINK_ROWS = failing, 60
        try:
            merge()
            interrupted = False
        except KeyboardInterrupt:
            interrupted = True
        core._sink_batch = sink
        check("a sink interrupted mid-table stops the merge", interrupted)
        check("no labels marker, and the processed labels files kept",
              not core._marker(Table.LABEL, group, staging).exists() and all(p.exists() for p in processed[Table.LABEL]))
        merge()
        core.GROUP_SINK_ROWS = 10_000_000
        resumed = digests(staging)
        repartition()
        core.GROUP_SINK_ROWS = 60
        merge()
        core.GROUP_SINK_ROWS = 10_000_000
        check("the resumed merge's group files are an uninterrupted merge's, byte for byte", resumed == digests(staging))
        check("(the batched merge made several batches)", resumed != one_batch)

        # Interrupted after its marker, while deleting the processed labels files
        repartition()
        unlink, deleted = pathlib.Path.unlink, []

        def failing_unlink(self, *a, **k):
            if self in processed[Table.LABEL]:
                deleted.append(self)
                if len(deleted) == 2:
                    raise KeyboardInterrupt
            return unlink(self, *a, **k)

        pathlib.Path.unlink = failing_unlink
        try:
            merge()
        except KeyboardInterrupt:
            pass
        pathlib.Path.unlink = unlink
        check("interrupted deleting: some processed labels files left", any(p.exists() for p in processed[Table.LABEL]))
        before = {p: p.stat().st_mtime_ns for p in (staging / Table.LABEL).rglob("*.parquet")}
        merge()
        check("the rerun deletes the rest without partitioning again",
              not any(p.exists() for p in processed[Table.LABEL])
              and {p: p.stat().st_mtime_ns for p in (staging / Table.LABEL).rglob("*.parquet")} == before)
        check("and its group files are an uninterrupted merge's", digests(staging) == one_batch)

        # A claims partition file with another schema: the merge fails, nothing is staged
        repartition()
        odd = results / Table.CLAIMS / "all" / "chunk_1.parquet"
        t = pq.read_table(odd)
        pq.write_table(t.append_column("extra", pa.array([1] * t.num_rows)), odd)
        try:
            merge()
            failed = False
        except ValueError as e:
            failed = "schema" in str(e)
        check("a claims partition file with another schema fails the merge", failed)
        check("no claims group file is left by the failed merge", not any((staging / Table.CLAIMS).rglob("*.parquet")) if (staging / Table.CLAIMS).exists() else True)
        check("the partition files are kept for the group's resumption", len(list((results / Table.CLAIMS / "all").glob("chunk_*.parquet"))) == CHUNKS)

        # A processed labels file gone before the merge: it raises, naming it
        repartition()
        gone = processed[Table.LABEL][1]
        gone.unlink()
        try:
            merge()
            failed = False
        except RuntimeError as e:
            failed = "processed files missing" in str(e) and str(gone) in str(e)
        check("a missing processed labels file fails the merge, naming it", failed)

        # A sidecar whose keys are not its rows': it raises, naming the keys
        repartition()
        side = audit / Table.LABEL / "chunk_1.parquet"
        pl.read_parquet(side).with_columns(pl.col("language").replace("de", "xx")).write_parquet(side)
        try:
            merge()
            failed = False
        except RuntimeError as e:
            failed = "differ from the audit sidecars" in str(e) and "'xx'" in str(e)
        check("a sidecar's keys not the rows' fail the merge, naming them", failed)
        os.chdir("/")
    print("all passed")


if __name__ == "__main__":
    main()
