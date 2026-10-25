#!/usr/bin/env python3
"""Test a group's close (push.core._partition_table) on the split's built term and
sitelink tables (partitioning.fused, the one-file layout) against the same close on the
pipeline's processed files of the same chunks.

The chunks given are a group, each chunk's four tables made three ways: processed (the
pipeline's normalisation, process.normalise_map_direct and normalise_sitelinks, with
write_stats_sidecar's sidecar), built (fused.write_one_file, as the split writes them,
placed by place_one_file, with write_built_sidecar's sidecar), and mixed (the second
chunk processed, the others built). For each table:

- each built table's distinct ids, as the build counts them, are its file's (Polars'
  n_unique), and process's id check (fused.check_built_ids: those and the part's empty
  cells add up to its ids) passes, and halts when an id is lost;
- the built sidecars have the processed ones' rows and columns but `path`, the one file,
  and `file_size`, which add up to the one file's bytes;
- the built group's key files are copied (no sink: partitioning.rowgroups) and each has
  the processed group's rows, in their order, and its Arrow schema, read by Polars and
  pyarrow;
- their row groups are the key's built row groups merged (rowgroups.merge_row_groups):
  none of more than GROUP_ROW_GROUP_ROWS rows but a built one, as few as that allows, and
  each file read the same by pyarrow (ParquetFile.read on one thread, read_table),
  pq_glue and Polars, with the same Arrow schema and footer_size (the sort's plan), as
  the same row groups copied one by one (rowgroups.copy_row_groups); so too merged at
  most 1,000 rows a row group, and one a row group (each built row group alone);
- a footer merge_row_groups does not read (Unsupported) is copied a row group at a time
  by the close, the same rows, as is a key whose merge raises (ValueError); two keys' row
  groups, whose dictionary pages differ, do not merge (ValueError);
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

import pq_glue  # noqa: E402
import wikidata.push.core as core  # noqa: E402
from wikidata.config import PARTITION_COLS  # noqa: E402
from wikidata.partitioning import fused, prepare_for_partition, rowgroups, write_stats_sidecar  # noqa: E402
from wikidata.partitioning.fused import FUSED, large_strings  # noqa: E402
from wikidata.process import atomic_sink_parquet, normalise_map_direct, normalise_sitelinks  # noqa: E402
from wikidata.push.groups import Group  # noqa: E402
from wikidata.sort_by_id import footer_size  # noqa: E402


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
    total = frame.get_column("id").n_unique()
    halts = counted = True
    for table in FUSED:
        fused.place_one_file(routed / table, results / table, name)
        # The build's distinct ids are those Polars counts in the file (check_ids' count)
        one = results / table / name
        n = pl.scan_parquet(one).select(pl.col("id").n_unique()).collect().item() if one.exists() else 0
        counted &= fused._index(results / table, name)["ids"] == n
        fused.check_built_ids(total, table, results / table, name)
        try:  # an entity lost
            fused.check_built_ids(total + 1, table, results / table, name)
            halts = False
        except RuntimeError:
            pass
        fused.write_built_sidecar(table, results / table, name, audit / table)
    check(f"{src.name}: the build's distinct ids are the built files' (Polars' n_unique)", counted)
    check(f"{src.name}: process's id check passes on the built tables, and halts on an id lost", halts)


def close(table: str, group: Group, root: Path) -> dict[str, Path]:
    core._partition_table(table, group, root / "audit", root / "staging", root / "results")
    return {p.parent.name: p for p in sorted((root / "staging" / table).glob(f"*/{group.name}.parquet"))}


def digest(files: dict[str, Path]) -> dict[str, str]:
    return {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in files.items()}


def read_alike(a: Path, b: Path) -> bool:
    """The two files read the same by every reader the pipeline uses, with the same Arrow
    schema and the same footer_size (the sort's plan)."""
    t = pq.read_table(a)
    return (t.equals(pq.read_table(b)) and t.equals(pq.ParquetFile(b).read(use_threads=False))
            and pq_glue.read_parquet(a).equals(pq_glue.read_parquet(b))
            and pl.read_parquet(a).equals(pl.read_parquet(b))
            and pq.read_schema(a).equals(pq.read_schema(b), check_metadata=True)
            and footer_size([a]) == footer_size([b]))


def check_layouts(table: str, built: Path, n_chunks: int, tmp: Path) -> None:
    """The merged layout against the row groups copied one by one, from the built files
    (before the close deletes them)."""
    files = [built / "results" / table / f"chunk_{i}.parquet" for i in range(n_chunks)]
    indexes = [fused.built_index(p.parent, p.name) for p in files]
    chunk = [rowgroups.ChunkFooter(p) if k else None for p, k in zip(files, indexes)]
    tree = [rowgroups.Footer(p) if k else None for p, k in zip(files, indexes)]
    parts: dict[str, list] = {}
    for c, t, keys in zip(chunk, tree, indexes):
        for key, _, _, _, _, first, n in keys:
            parts.setdefault(key, []).append((c, t, first, n))
    out = tmp / "layouts" / table
    out.mkdir(parents=True)
    alike = fewest = capped = single = True
    cap = core.GROUP_ROW_GROUP_ROWS
    for key, ps in parts.items():
        copied = out / f"{key}.copied.parquet"
        rowgroups.copy_row_groups(copied, [(t, first, n) for _, t, first, n in ps])
        built_rows = [pq.read_metadata(copied).row_group(g).num_rows for g in range(pq.read_metadata(copied).num_row_groups)]
        for max_rows, name in ((cap, "merged"), (1000, "merged-1000"), (1, "merged-1")):
            dst = out / f"{key}.{name}.parquet"
            rowgroups.merge_row_groups(dst, [(c, first, n) for c, _, first, n in ps], max_rows)
            alike &= read_alike(copied, dst)
            meta = pq.read_metadata(dst)
            rows = [meta.row_group(g).num_rows for g in range(meta.num_row_groups)]
            capped &= all(r <= max_rows or r in built_rows for r in rows)
            # as few as the cap allows: no two consecutive row groups would fit in one
            fewest &= all(a + b > max_rows for a, b in zip(rows, rows[1:]))
            if max_rows == 1:
                single &= rows == built_rows
    check(f"{table}: {len(parts)} keys merged, each read as its row groups copied one by one "
          "(pyarrow, pq_glue, Polars; Arrow schema; footer_size)", alike)
    check(f"{table}: merged row groups within their cap (or a built one), as few as it allows", capped and fewest)
    check(f"{table}: merged one a row group, the built row groups themselves", single)
    two = [k for k, ps in parts.items() if ps[0][0] is not None][:2]
    if len(two) == 2:
        try:
            rowgroups.merge_row_groups(out / "two.parquet", [(c, f, n) for k in two for c, _, f, n in parts[k][:1]], cap)
            check(f"{table}: two keys' row groups (other dictionary pages) do not merge", False)
        except ValueError:
            check(f"{table}: two keys' row groups (other dictionary pages) do not merge", True)


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

            check_layouts(table, arms["built"], len(chunks), tmp)

            calls = []
            copy = rowgroups.merge_row_groups

            def count(dst, parts, **kw):
                calls.append(dst)
                return copy(dst, parts, **kw)

            def interrupt(dst, parts, **kw):
                if len(calls) >= 3:
                    raise KeyboardInterrupt("killed")
                return count(dst, parts, **kw)

            want = close(table, group, arms["processed"])
            unsupported = tmp / "unsupported"
            shutil.copytree(arms["built"], unsupported)
            core.rowgroups.merge_row_groups = count
            try:
                got = close(table, group, arms["built"])
            finally:
                core.rowgroups.merge_row_groups = copy
            check(f"{table}: the built group's {len(got)} key files merged, not sunk", len(calls) == len(got) > 0)

            def unread(path):
                raise rowgroups.Unsupported("a field")

            chunk_footer, core.rowgroups.ChunkFooter = core.rowgroups.ChunkFooter, unread
            copied = []
            row_copy = rowgroups.copy_row_groups
            core.rowgroups.copy_row_groups = lambda dst, parts: copied.append(dst) or row_copy(dst, parts)
            try:
                fallback = close(table, group, unsupported)
            finally:
                core.rowgroups.ChunkFooter, core.rowgroups.copy_row_groups = chunk_footer, row_copy
            check(f"{table}: a footer not read for merging copied a row group at a time, the same rows",
                  len(copied) == len(got) and set(fallback) == set(got)
                  and all(pl.read_parquet(fallback[k]).equals(pl.read_parquet(got[k])) for k in got))
            shutil.rmtree(unsupported)

            def unmerged(dst, parts, **kw):
                raise ValueError("a dictionary page not the first source's")

            shutil.copytree(arms["mixed"], unsupported)  # (not yet closed: the built files)
            for d in ("staging", "audit", "results"):
                shutil.rmtree(unsupported / d, ignore_errors=True)
            shutil.copytree(arms["resumed"] / "audit", unsupported / "audit")
            shutil.copytree(arms["resumed"] / "results", unsupported / "results")
            copied.clear()
            core.rowgroups.merge_row_groups = unmerged
            core.rowgroups.copy_row_groups = lambda dst, parts: copied.append(dst) or row_copy(dst, parts)
            try:
                fallback = close(table, group, unsupported)
            finally:
                core.rowgroups.merge_row_groups, core.rowgroups.copy_row_groups = copy, row_copy
            check(f"{table}: a key whose row groups do not merge copied a row group at a time, the same rows",
                  len(copied) == len(got) and set(fallback) == set(got)
                  and all(pl.read_parquet(fallback[k]).equals(pl.read_parquet(got[k])) for k in got))
            shutil.rmtree(unsupported)
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
            core.rowgroups.merge_row_groups = interrupt
            try:
                close(table, group, arms["resumed"])
                check(f"{table}: the interrupted close raised", False)
            except KeyboardInterrupt:
                pass
            finally:
                core.rowgroups.merge_row_groups = copy
            kept = all((arms["resumed"] / "results" / table / f"chunk_{i}.parquet").exists() for i in range(len(chunks)))
            check(f"{table}: interrupted, no marker and the built files kept",
                  kept and not core._marker(table, group, arms["resumed"] / "staging").exists())
            resumed = close(table, group, arms["resumed"])
            check(f"{table}: the rerun's key files the uninterrupted close's, byte for byte", digest(resumed) == digest(got))
            print(f"     {table}: {sum(pl.read_parquet(f).height for f in got.values()):,} rows in {len(got)} keys", flush=True)


if __name__ == "__main__":
    main()
