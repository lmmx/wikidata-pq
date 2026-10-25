#!/usr/bin/env python3
"""Test the fused term and sitelink partitions (partitioning/fused.py, reshape/src/fuse.rs)
against the pipeline's: genson's normalisation (process.normalise_map_direct,
normalise_sitelinks), the explode (transforms.prepare_for_partition) and Polars' sink
(core.partition_parquet), run on the same chunk file.

For each of labels, descriptions, aliases and links, and each chunk:
- the same partition files (by relative path), so the same keys;
- each file's rows equal to the pipeline's rows of its key, in their order, and to the
  sink's file read back (Polars, pyarrow and pq_glue each read both the same: values,
  Arrow schema, dtypes);
- the same Parquet schema, the same row groups (count and rows), and the same
  statistics of `id` and the key in each row group (what the sort's planning reads);
- the audit sidecar equal to write_frame_sidecar's but `file_size`, which is each fused
  file's own size;
- each fused file glued (pq_glue.concat, as merge_group glues) after the sink's file of
  its key, read back as the two files' rows;
- in the one-file layout, each key's row groups read back as its key file;
- placed in place (`place_built_terms(in_place=True)`), the sidecar naming the built files,
  each read as the sink's.

Each chunk is also checked as a routed part (every third row, given as positions in the
chunk's columns, as dump.route_parts gives a part's), against the pipeline on a chunk file
of those rows. The fused partitions go through the split's step (`write_terms`, to a
routed directory) and the pipeline's (`place_built_terms`).

The chunks: the files given (the bench's chunks: bench-data's c0 ... s90), and synthetic
ones with escapes, empty strings and maps, badges null, empty and several, multi-byte
keys, keys that sort differently as paths and as strings, a key of 99,000 rows, and a key over a
row group (its rows and schema compared, not its row groups: see BIG).
Cells the build does not reproduce (a null, a number, an empty list of aliases, a link
field other than site, title and badges, a link without a title) are misfits: nothing is
written. Runs in a temporary directory, as a release (WIKIDATA_RELEASE).

Usage: python scripts/test_fused_terms.py [CHUNK.parquet ...]
"""

import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

os.environ.setdefault("WIKIDATA_RELEASE", "20261012")

import orjson  # noqa: E402
import polars as pl  # noqa: E402
import pq_glue  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from wikidata.config import PARTITION_COLS, Table  # noqa: E402
from wikidata.partitioning import partition_parquet, prepare_for_partition  # noqa: E402
from wikidata.partitioning.fused import (  # noqa: E402
    FUSED,
    build_terms,
    large_strings,
    place_built_terms,
    write_fused_partitions,
    write_terms,
)
from wikidata.process import atomic_sink_parquet, normalise_map_direct, normalise_sitelinks  # noqa: E402

COLUMNS = ["id", *FUSED.values()]


def check(name: str, ok: bool) -> None:
    print(f"{'ok  ' if ok else 'FAIL'} {name}", flush=True)
    if not ok:
        sys.exit(1)


def pipeline(src: Path, out: Path, name: str) -> dict[Table, pl.DataFrame]:
    """The pipeline's four tables for `src`, exploded, with its partition files and
    sidecars under `out` (as process() and partition_chunk() make them)."""
    processed = out / "processed"
    for table in (Table.LABEL, Table.DESC, Table.ALIAS):
        normalise_map_direct(src, processed / table / name, key=FUSED[table], lists=table == Table.ALIAS)
    links = normalise_sitelinks(pl.read_parquet(src, columns=["id", "sitelinks"]))
    atomic_sink_parquet(links.lazy(), processed / Table.LINKS / name)
    tables = {}
    for table in FUSED:
        lf = prepare_for_partition(processed / table / name, table)
        tables[table] = lf.collect()
        partition_parquet(
            by=PARTITION_COLS[table], lf=lf, source_name=name, dst_dir=out / "part" / table, log_dir=out / "audit" / table
        )
    return tables


def files(d: Path) -> list[str]:
    return sorted(str(p.relative_to(d)) for p in d.rglob("*.parquet"))


def stats(meta: pq.FileMetaData) -> list:
    """Each row group's rows, and its id and key columns' statistics"""
    out = []
    for g in range(meta.num_row_groups):
        rg = meta.row_group(g)
        cols = []
        for c in range(2):
            s = rg.column(c).statistics
            cols.append((s.has_min_max, s.min, s.max, s.null_count) if s is not None else None)
        out.append((rg.num_rows, cols))
    return out


def compare(label: str, src: Path, tmp: Path, *, rows: list[int] | None = None, layout: bool = True) -> None:
    """The fused partitions of chunk `src` (of its `rows` only, a routed part's, if given)
    against the pipeline's on the same rows. With `layout`, the row groups too."""
    name = "chunk_0.parquet"
    ref_dir, fused_dir = tmp / f"{label}-ref", tmp / f"{label}-fused"
    frame = pl.read_parquet(src, columns=COLUMNS)
    if rows is None:
        tables = pipeline(src, ref_dir, name)
        why = write_fused_partitions(frame, name, fused_dir / "part", fused_dir / "audit", fused_dir / "routed")
    else:
        part = tmp / f"{label}-part.parquet"
        pl.read_parquet(src)[rows].write_parquet(part)
        tables = pipeline(part, ref_dir, name)
        texts = {t: large_strings(frame.get_column(c)) for t, c in FUSED.items()}
        why = write_terms(frame.get_column("id"), texts, pa.array(rows, pa.int64()), fused_dir / "routed", name)
        if why is None:
            place_built_terms(fused_dir / "routed", name, fused_dir / "part", fused_dir / "audit")
    check(f"{label}: every cell fits ({why})", why is None)
    check(f"{label}: nothing is left of the built tables once placed", not any(p.is_file() or p.parent.name == "chunk_0" for p in (fused_dir / "routed").rglob("*")))
    for table in FUSED:
        key = PARTITION_COLS[table]
        ref_part, fused_part = ref_dir / "part" / table, fused_dir / "part" / table
        ref_files, fused_files = files(ref_part), files(fused_part)
        check(f"{label} {table}: the same {len(ref_files)} partition files", ref_files == fused_files)
        want = tables[table]
        same_rows = same_read = same_layout = glued = small_groups = True
        for rel in ref_files:
            ref_f, fused_f = ref_part / rel, fused_part / rel
            k = Path(rel).parent.name
            mine = pl.read_parquet(fused_f)
            same_rows &= mine.equals(want.filter(pl.col(key) == k)) and mine.schema == want.schema
            ref_df = pl.read_parquet(ref_f)
            a, b = pq.read_table(ref_f), pq.read_table(fused_f)
            g1, g2 = pq_glue.read_parquet(str(ref_f)), pq_glue.read_parquet(str(fused_f))
            same_read &= (
                mine.equals(ref_df)
                and a.schema.remove_metadata().equals(b.schema.remove_metadata())
                and a.equals(b)
                and g1.schema.remove_metadata().equals(g2.schema.remove_metadata())
                and g1.equals(g2)
            )
            m1, m2 = pq.read_metadata(ref_f), pq.read_metadata(fused_f)
            same_layout &= m1.schema.equals(m2.schema) and (stats(m1) == stats(m2) if layout else True)
            small_groups &= all(m2.row_group(g).num_rows <= 122_880 for g in range(m2.num_row_groups))
            dst = tmp / "glued.parquet"
            n = pq_glue.concat([str(ref_f), str(fused_f)], str(dst))
            glued &= n == 2 * ref_df.height and pl.read_parquet(dst).equals(pl.concat([ref_df, mine]))
        check(f"{label} {table}: each file holds its key's rows, in order", same_rows)
        check(f"{label} {table}: each file reads as the sink's (Polars, pyarrow, pq_glue)", same_read)
        what = "row groups and id/key statistics" if layout else "(row groups of at most 122,880 rows)"
        check(f"{label} {table}: the same Parquet schema, {what}", same_layout and small_groups)
        check(f"{label} {table}: each file glues after the sink's", glued)
        ref_sc = pl.read_parquet(ref_dir / "audit" / table / name)
        fused_sc = pl.read_parquet(fused_dir / "audit" / table / name)
        relative = pl.col("path").str.replace(r"^.*/part/", "")
        check(
            f"{label} {table}: the sidecar is write_frame_sidecar's but file_size",
            ref_sc.schema == fused_sc.schema
            and ref_sc.drop("file_size").with_columns(relative).equals(fused_sc.drop("file_size").with_columns(relative)),
        )
        sizes = [os.path.getsize(p) for p in fused_sc["path"]]
        check(f"{label} {table}: the sidecar's file_size is each fused file's size", sizes == fused_sc["file_size"].to_list())
    # Placed in place: the sidecars name the built files, which read as the sink's
    inplace = tmp / f"{label}-inplace"
    texts = {t: large_strings(frame.get_column(c)) for t, c in FUSED.items()}
    positions = None if rows is None else pa.array(rows, pa.int64())
    assert write_terms(frame.get_column("id"), texts, positions, inplace / "routed", name) is None
    place_built_terms(inplace / "routed", name, inplace / "part", inplace / "audit", in_place=True)
    ok = not (inplace / "part").exists()
    for table in FUSED:
        sc = pl.read_parquet(inplace / "audit" / table / name)
        ref_sc = pl.read_parquet(ref_dir / "audit" / table / name)
        ok &= sc.drop("path", "file_size").equals(ref_sc.drop("path", "file_size"))
        for path, k in sc.select("path", PARTITION_COLS[table]).iter_rows():
            ok &= Path(path).parent == inplace / "routed" / table / "chunk_0"
            ok &= pl.read_parquet(path).equals(pl.read_parquet(ref_dir / "part" / table / k / name))
    check(f"{label}: placed in place, the sidecars name the built files, each read as the sink's", ok)
    # The one-file layout: each key's row groups read back as its file
    one = tmp / f"{label}-one"
    built, why = build_terms(large_strings(frame.get_column("id")), texts, positions, {t: one / f"{t}.parquet" for t in FUSED}, one_file=True)
    ok = built is not None
    for table, keys in (built or {}).items():
        f = pq.ParquetFile(one / f"{table}.parquet") if keys else None
        for k, n, _, _, _, first, count in keys:
            got = pl.from_arrow(f.read_row_groups(list(range(first, first + count))))
            ok &= got.height == n and got.equals(pl.read_parquet(fused_dir / "part" / table / k / name))
    check(f"{label}: the one-file layout's row groups by key read back as the key files", ok)
    print(f"     {label}: {sum(t.height for t in tables.values()):,} rows", flush=True)


def link(site: str, title: str, badges=()) -> dict:
    return {"site": site, "title": title, "badges": list(badges)}


def synthetic(path: Path, rows: list[dict]) -> Path:
    """A chunk file of JSON columns, as the split writes them (orjson)"""
    cols = {c: [orjson.dumps(r.get(c, {})).decode() if c != "id" else r["id"] for r in rows] for c in COLUMNS}
    cols |= {"claims": ["{}"] * len(rows), "entity": ['{"type":"item"}'] * len(rows)}
    pl.DataFrame(cols).write_parquet(path)
    return path


EDGES = [
    {
        "id": "Q1",
        "labels": {"en": 'quote " backslash \\ slash / tab \t nl \n', "de": "", "be-tarask": "x", "be": "y"},
        "descriptions": {"zh": "\u0001 control \u001f", "zh-hans": "emoji \U0001f98f", "zh_min_nan": "é"},
        "aliases": {"en": ["a", "", "a"], "fr": ["b"]},
        "sitelinks": {"enwiki": link("enwiki", "T", ["Q17437796", "Q70894304"]), "dewiki": link("dewiki", "")},
    },
    {"id": "Q2"},
    {
        "id": "Q10",
        "labels": {"en": "second", "ru": "Россия"},
        "aliases": {"en": ["c"]},
        "sitelinks": {"enwiki": {"site": "enwiki", "title": "U", "badges": None}, "frwiki": {"title": "V", "site": "frwiki"}},
    },
    {"id": "Q3", "labels": {}, "descriptions": {}, "aliases": {}, "sitelinks": {}},
    {"id": "P31", "labels": {"ét": "keyed by a multi-byte key"}, "sitelinks": {"a": link("enwiki", "W", ["Q1"])}},
]
# A key with more rows than a row group (Polars' sink splits a key's rows past about
# 100,000 by its own morsels, so only the rows and schema are compared)
BIG = [{"id": f"Q{i}", "labels": {"en": f"v{i}"}, "aliases": {"en": ["a", "b"]}} for i in range(70_000)]
# A key with fewer: one row group, as the sink's
NEAR = [{"id": f"Q{i}", "labels": {"en": f"v{i}"}, "aliases": {"en": ["a"]}} for i in range(99_000)]

MISFITS = {
    "a null label": {"labels": {"en": None}},
    "a label that is a number": {"labels": {"en": 1}},
    "an empty list of aliases": {"aliases": {"en": []}},
    "aliases that are a string": {"aliases": {"en": "a"}},
    "a null alias": {"aliases": {"en": [None]}},
    "a link field other than site, title and badges": {"sitelinks": {"enwiki": {**link("enwiki", "T"), "url": "u"}}},
    "a link without a title": {"sitelinks": {"enwiki": {"site": "enwiki", "badges": []}}},
    "a badge that is not a string": {"sitelinks": {"enwiki": link("enwiki", "T", [1])}},
    "a key with a slash": {"labels": {"a/b": "x"}},
}


def main() -> None:
    chunks = [Path(p).resolve() for p in sys.argv[1:]]
    with TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        compare("edge cases", synthetic(tmp / "edges.parquet", EDGES), tmp)
        compare("a key over a row group", synthetic(tmp / "big.parquet", BIG), tmp, layout=False)
        compare("a key of 99,000 rows", synthetic(tmp / "near.parquet", NEAR), tmp)
        compare("a part (rows 0, 2 and 4)", synthetic(tmp / "edges.parquet", EDGES), tmp, rows=[0, 2, 4])
        for why, cells in MISFITS.items():
            src = synthetic(tmp / "misfit.parquet", [EDGES[2], {"id": "Q9", **cells}])
            out = tmp / "misfit"
            result = write_fused_partitions(pl.read_parquet(src, columns=COLUMNS), "chunk_0.parquet", out / "part", out / "audit", out / "routed")
            check(f"misfit, nothing written: {why} ({result})", result is not None and "row 1:" in result and not any(out.rglob("*.parquet")))
        for src in chunks:
            compare(src.stem, src, tmp)
            n = pl.scan_parquet(src).select(pl.len()).collect().item()
            compare(f"{src.stem}, every third row", src, tmp, rows=list(range(0, n, 3)))


if __name__ == "__main__":
    main()
