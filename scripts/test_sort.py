#!/usr/bin/env python3
"""Test the bucketed sort (sort_by_id.sort_bucketed) on a small sample of claims: rows
taken from a key's files and shuffled across a few source files, sorted with tiny
buckets, part files and row groups so every step runs with several workers.

Checks that the part files hold the sample's rows in stable id order, that bucketing
resumes from `bucketed.jsonl`, and that the checks catch what they are for: a fragment
changed after bucketing, a sort that is no permutation, an id out of order, an id's rows
reordered (not a stable sort), a value changed in a written file or a row group missing
from it, and a bucket's rows miscounted. Runs in a temporary directory; reads the
claims files only.

Usage: python scripts/test_sort.py [claims key dir] [rows per slice]
(default releases/20260928/hub/claims/all, 50,000)
"""

import json
import os
import shutil
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "releases/20260928/hub/claims/all").resolve()
ROWS = int(sys.argv[2]) if len(sys.argv) > 2 else 50_000
N_SOURCES = 4
SAMPLE_RG_ROWS = 2_000  # small row groups, as each bucket gets a piece of every one


def make_sample(dst: Path) -> list[Path]:
    """N_SOURCES files of slices from files spread through the key, rows shuffled (fixed
    seed), so every row group, like a real source row group, has rows for every bucket;
    in row groups of SAMPLE_RG_ROWS."""
    files = sorted(SRC.glob("*.parquet"))
    picks = [files[round(k * (len(files) - 1) / (2 * N_SOURCES - 1))] for k in range(2 * N_SOURCES)]
    rows = pa.concat_tables(pq.ParquetFile(f).read_row_group(0).slice(0, ROWS) for f in picks)
    rows = rows.take(pa.array(np.random.default_rng(0).permutation(rows.num_rows)))
    per_file = rows.num_rows // N_SOURCES
    dst.mkdir(parents=True)
    out = []
    for j in range(N_SOURCES):
        path = dst / f"chunks-{j:02d}-{j:02d}.parquet"
        pq.write_table(rows.slice(j * per_file, per_file), path, row_group_size=SAMPLE_RG_ROWS)
        out.append(path)
    return out


def fails(name: str, f, words: str) -> None:
    """`f` raises a RuntimeError saying `words`."""
    try:
        f()
    except RuntimeError as e:
        check(f"{name} ({e})", words in str(e))
        return
    check(f"{name}: no error raised", False)


def streams(path: Path) -> list[pa.Table]:
    """A fragment's IPC streams, one per flush."""
    data, out = path.read_bytes(), []
    reader = pa.BufferReader(data)
    while reader.tell() < len(data):
        out.append(ipc.open_stream(reader).read_all())
    return out


def read_all(paths: list[Path]) -> pl.DataFrame:
    return pl.concat([pl.read_parquet(p) for p in paths])


def check(name: str, ok: bool) -> None:
    print(f"{'ok  ' if ok else 'FAIL'} {name}", flush=True)
    if not ok:
        sys.exit(1)


def main() -> None:
    with TemporaryDirectory() as tmp:
        os.chdir(tmp)
        os.environ["WIKIDATA_RELEASE"] = "sorttest"
        os.environ["WIKIDATA_FINALISE_WORKERS"] = "3"
        from wikidata import compact
        from wikidata import sort_by_id as sbi
        from wikidata.config import SORT_DIR, Table

        sources = make_sample(sbi._src_dir(Table.CLAIMS) / "all")
        total = sum(p.stat().st_size for p in sources)
        sbi.SORT_BUCKET_BYTES = total // 20
        sbi.COMPACT_FILE_BYTES = total // 4
        compact.COMPACT_ROW_GROUP_BYTES = sbi.COMPACT_ROW_GROUP_BYTES = 4 * 1024**2
        sbi.SORT_BUCKET_WRITE_BYTES = 1024**2
        rows = sum(pq.ParquetFile(p).metadata.num_rows for p in sources)
        print(f"sample: {N_SOURCES} files, {rows:,} rows, {total / 1e6:.0f} MB", flush=True)
        expected = read_all(sources).sort("id", maintain_order=True)

        t0 = time.time()
        files = sbi.sort_bucketed(Table.CLAIMS, "all", sources)
        print(f"sorted in {time.time() - t0:.0f} s, {len(files)} part files", flush=True)
        outputs = [sbi._out_dir(Table.CLAIMS) / "all" / f["name"] for f in files]
        check("part files hold the sample's rows in stable id order", read_all(outputs).equals(expected))
        bdir = sbi._bucket_dir(Table.CLAIMS, "all")
        buckets = json.loads((bdir / "buckets.json").read_text())
        check("buckets.json has the fragments' format", buckets["format"] == sbi.FRAGMENT_FORMAT
              and buckets["fragments"] == N_SOURCES)
        check("bucket fragments removed once sorted", not list(bdir.glob("bucket-*")))
        lines = (bdir / "sorted.jsonl").read_text().splitlines()
        sorted_buckets = sorted((json.loads(line) for line in lines), key=lambda b: b["name"])
        sbi._check_whole("all", sources, buckets, sorted_buckets)
        check("end check passes on the sorted buckets", True)
        fewer = [dict(sorted_buckets[0], rows=sorted_buckets[0]["rows"] - 1), *sorted_buckets[1:]]
        fails("end check catches a bucket's rows miscounted", lambda: sbi._check_whole("all", sources, buckets, fewer),
              "differ from those bucketed")

        # The sort and the written file's checks, on a bucket's worth of rows
        t = pq.read_table(sources[0]).replace_schema_metadata()
        idx = pa.compute.sort_indices(t, [("id", "ascending")])
        st = sbi._sorted(t, "id", "sample")
        check("a correct sort passes", st.num_rows == t.num_rows)
        order = idx.to_numpy().copy()
        dup = order.copy()
        dup[1] = dup[0]
        fails("sort check catches a row taken twice", lambda: sbi._check_sort("s", t, pa.array(dup), t.take(pa.array(dup)), "id"),
              "not a permutation")
        ids = st.column("id").to_numpy(zero_copy_only=False)
        j = int(np.flatnonzero(ids[1:] != ids[:-1])[0])  # two rows of different ids, in order
        swapped = order.copy()
        swapped[[j, j + 1]] = swapped[[j + 1, j]]
        fails("sort check catches ids out of order", lambda: sbi._check_sort("s", t, pa.array(swapped), t.take(pa.array(swapped)), "id"),
              "not sorted")
        k = int(np.flatnonzero(ids[1:] == ids[:-1])[0])  # two rows of one id
        unstable = order.copy()
        unstable[[k, k + 1]] = unstable[[k + 1, k]]
        fails("sort check catches an id's rows reordered", lambda: sbi._check_sort("s", t, pa.array(unstable), t.take(pa.array(unstable)), "id"),
              "not a stable sort")
        other = st.set_column(0, st.schema.field(0), st.column(0).take(pa.array(np.roll(np.arange(st.num_rows), 1))))
        fails("sort check catches ids that are not the rows' own", lambda: sbi._check_sort("s", t, idx, other, "id"),
              "not the rows' own")

        schema = st.schema
        out = Path("check.parquet")
        rg = sbi._row_group_rows(sources)
        compact._write_file_parallel(out, schema, st.to_batches(max_chunksize=rg), rg)
        sbi._check_written("w", [out], st, schema)
        check("written file check passes on the sorted rows", True)
        i = schema.get_field_index("property")
        prop = st.column("property").to_pylist()
        prop[st.num_rows // 2] = "P0"
        changed = st.set_column(i, schema.field(i), pa.array(prop, schema.field(i).type))
        compact._write_file_parallel(out, schema, changed.to_batches(max_chunksize=rg), rg)
        fails("written file check catches a value changed", lambda: sbi._check_written("w", [out], st, schema), "differs")
        compact._write_file_parallel(out, schema, st.slice(0, st.num_rows - 1).to_batches(max_chunksize=rg), rg)
        fails("written file check catches a row missing", lambda: sbi._check_written("w", [out], st, schema), "rows")

        shutil.rmtree(SORT_DIR)
        first = sbi.bucket_key(Table.CLAIMS, "all", sources)
        frag = max(bdir.glob(f"bucket-*.{sbi.FRAGMENT_FORMAT}"), key=lambda p: p.stat().st_size)
        n_rg = len(streams(frag))
        src_rgs = pq.ParquetFile(sources[0]).metadata.num_row_groups
        flush = sbi._row_group_rows(sources) * sbi.SORT_BUCKET_WRITE_BYTES // sbi.COMPACT_ROW_GROUP_BYTES
        want = -(-sum(s.num_rows for s in streams(frag)) // flush)
        check(
            f"buffered writes: {frag.name} has {n_rg} record batches, {want} expected"
            f" (unbuffered: one per source row group, {src_rgs})",
            n_rg <= want + 1 < src_rgs,
        )
        (bdir / "buckets.json").unlink()
        log = bdir / "bucketed.jsonl"
        lines = log.read_text().splitlines()
        log.write_text("\n".join(lines[:-1]) + "\n")
        again = sbi.bucket_key(Table.CLAIMS, "all", sources)
        check("bucketing resumes from bucketed.jsonl, same buckets", again == first)
        check("bucketing redid only the missing file", len(log.read_text().splitlines()) == N_SOURCES)

        t = pa.concat_tables(streams(frag))
        prop = t.column("property").to_pylist()
        prop[0] = "P0"
        i = t.schema.get_field_index("property")
        frag.write_bytes(sbi._ipc_stream(t.set_column(i, t.schema.field(i), pa.array(prop, t.schema.field(i).type)), t.schema).to_pybytes())
        fails("sorting catches a fragment changed after bucketing",
              lambda: sbi.sort_buckets(Table.CLAIMS, "all", sources, again), "differs from the bytes bucketed")
        os.chdir("/")
    print("all passed")


if __name__ == "__main__":
    main()
