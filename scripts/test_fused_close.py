#!/usr/bin/env python3
"""Test a group's close (push.core._partition_table) on the split's built term and
sitelink tables (partitioning.fused, the one-file layout) against the same close on the
pipeline's processed files of the same chunks.

The chunks given are a group, each chunk's four tables made three ways: processed (the
pipeline's normalisation, process.normalise_map_direct and normalise_sitelinks, with
write_stats_sidecar's sidecar), built (fused.write_one_file, as the split writes them,
placed by place_one_file, with write_built_sidecar's sidecar), and mixed (the second
chunk processed, the others built). For each table:

- the built sidecars have the processed ones' rows and columns but `path`, the one file,
  and `file_size`, which add up to the one file's bytes;
- the built group's key files are copied (no sink: partitioning.rowgroups) and each has
  the processed group's rows, in their order, and its Arrow schema, read by Polars and
  pyarrow;
- the mixed group (sunk, as a processed group) has the same rows;
- a close of the built group interrupted in its copy leaves no marker and the built files
  kept, and its rerun's key files are an uninterrupted close's, byte for byte;
- the processed files and indexes are deleted once the group files are checked.

Runs in a temporary directory, as a release (WIKIDATA_RELEASE).

Usage: python scripts/test_fused_close.py CHUNK.parquet CHUNK.parquet [CHUNK.parquet ...]
"""

import hashlib
import os
import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

os.environ.setdefault("WIKIDATA_RELEASE", "20261012")

import polars as pl  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

import wikidata.push.core as core  # noqa: E402
from wikidata.config import PARTITION_COLS  # noqa: E402
from wikidata.partitioning import fused, prepare_for_partition, rowgroups, write_stats_sidecar  # noqa: E402
from wikidata.partitioning.fused import FUSED, large_strings  # noqa: E402
from wikidata.process import atomic_sink_parquet, normalise_map_direct, normalise_sitelinks  # noqa: E402
from wikidata.push.groups import Group  # noqa: E402


def check(name: str, ok: bool) -> None:
    print(f"{'ok  ' if ok else 'FAIL'} {name}", flush=True)
    if not ok:
        sys.exit(1)


def processed(src: Path, results: Path, audit: Path, name: str) -> None:
    """The chunk's four tables as the pipeline processes and partitions them."""
    for table in FUSED:
        dst = results / table / name
        if table == "links":
            atomic_sink_parquet(normalise_sitelinks(pl.read_parquet(src, columns=["id", "sitelinks"])).lazy(), dst)
        else:
            normalise_map_direct(src, dst, key=FUSED[table], lists=table == "aliases")
        write_stats_sidecar(PARTITION_COLS[table], prepare_for_partition(dst, table), dst_dir=results / table,
                            source=Path(name), log_dir=audit / table, file_bytes=dst.stat().st_size)


def built(src: Path, results: Path, audit: Path, name: str, routed: Path) -> None:
    """The chunk's four tables as the split builds them and the pipeline places them."""
    frame = pl.read_parquet(src, columns=["id", *FUSED.values()])
    texts = {t: large_strings(frame.get_column(c)) for t, c in FUSED.items()}
    sizes, why = fused.write_one_file(frame.get_column("id"), texts, None, routed, name)
    check(f"{src.name}: every cell fits ({why})", sizes is not None)
    for table in FUSED:
        fused.place_one_file(routed / table, results / table, name)
        fused.write_built_sidecar(table, results / table, name, audit / table)


def close(table: str, group: Group, root: Path) -> dict[str, Path]:
    core._partition_table(table, group, root / "audit", root / "staging", root / "results")
    return {p.parent.name: p for p in sorted((root / "staging" / table).glob(f"*/{group.name}.parquet"))}


def digest(files: dict[str, Path]) -> dict[str, str]:
    return {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in files.items()}


def main() -> None:
    chunks = [Path(a) for a in sys.argv[1:]]
    if len(chunks) < 2:
        raise SystemExit(__doc__)
    group = Group(0, len(chunks) - 1, len(chunks) - 1)
    with TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        arms = {a: tmp / a for a in ("processed", "built", "mixed", "resumed")}
        for i, src in enumerate(chunks):
            name = f"chunk_{i}.parquet"
            processed(src, arms["processed"] / "results", arms["processed"] / "audit", name)
            built(src, arms["built"] / "results", arms["built"] / "audit", name, tmp / "routed")
        for arm in ("mixed", "resumed"):
            shutil.copytree(arms["built"], arms[arm])
        mid = "chunk_1"
        for table in FUSED:  # the mixed group's second chunk: processed
            for p in (arms["mixed"] / "results" / table).glob(f"{mid}.*"):
                p.unlink()
            shutil.copy(arms["processed"] / "results" / table / f"{mid}.parquet", arms["mixed"] / "results" / table)
            shutil.copy(arms["processed"] / "audit" / table / f"{mid}.parquet", arms["mixed"] / "audit" / table)

        for table in FUSED:
            ok = sums = paths = True
            for i in range(len(chunks)):
                name = f"chunk_{i}.parquet"
                want = pl.read_parquet(arms["processed"] / "audit" / table / name)
                got = pl.read_parquet(arms["built"] / "audit" / table / name)
                ok &= got.drop("path", "file_size").equals(want.drop("path", "file_size"))
                one = arms["built"] / "results" / table / name
                if got.height:
                    sums &= int(got["file_size"].sum()) == one.stat().st_size
                    paths &= set(got["path"]) == {str(one)}
            check(f"{table}: the built sidecars are the processed ones' but path and file_size", ok)
            check(f"{table}: their path is the one file, their file sizes add up to it", sums and paths)

            calls = []
            copy = rowgroups.copy_row_groups

            def count(dst, parts):
                calls.append(dst)
                return copy(dst, parts)

            def interrupt(dst, parts):
                if len(calls) >= 3:
                    raise KeyboardInterrupt("killed")
                return count(dst, parts)

            want = close(table, group, arms["processed"])
            core.rowgroups.copy_row_groups = count
            try:
                got = close(table, group, arms["built"])
            finally:
                core.rowgroups.copy_row_groups = copy
            check(f"{table}: the built group's {len(got)} key files copied, not sunk", len(calls) == len(got) > 0)
            check(f"{table}: the same keys", set(got) == set(want))
            same = schema = gone = True
            for k, f in got.items():
                a, b = pl.read_parquet(f), pl.read_parquet(want[k])
                same &= a.equals(b) and pq.read_table(f).equals(pq.read_table(want[k]))
                schema &= pq.read_schema(f).serialize() == pq.read_schema(want[k]).serialize()
            for i in range(len(chunks)):
                one = arms["built"] / "results" / table / f"chunk_{i}.parquet"
                gone &= not one.exists() and not fused.index_path(one.parent, one.name).exists()
            check(f"{table}: each key's rows, in their order, and its Arrow schema", same and schema)
            check(f"{table}: the built files and indexes deleted once checked", gone)

            mixed = close(table, group, arms["mixed"])
            check(f"{table}: the mixed group (sunk) the same rows",
                  set(mixed) == set(want) and all(pl.read_parquet(mixed[k]).equals(pl.read_parquet(want[k])) for k in want))

            calls.clear()
            core.rowgroups.copy_row_groups = interrupt
            try:
                close(table, group, arms["resumed"])
                check(f"{table}: the interrupted close raised", False)
            except KeyboardInterrupt:
                pass
            finally:
                core.rowgroups.copy_row_groups = copy
            kept = all((arms["resumed"] / "results" / table / f"chunk_{i}.parquet").exists() for i in range(len(chunks)))
            check(f"{table}: interrupted, no marker and the built files kept",
                  kept and not core._marker(table, group, arms["resumed"] / "staging").exists())
            resumed = close(table, group, arms["resumed"])
            check(f"{table}: the rerun's key files the uninterrupted close's, byte for byte", digest(resumed) == digest(got))
            print(f"     {table}: {sum(pl.read_parquet(f).height for f in got.values()):,} rows in {len(got)} keys", flush=True)


if __name__ == "__main__":
    main()
