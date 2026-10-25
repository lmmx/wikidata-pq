#!/usr/bin/env python3
"""Test the bucketed sort (sort_by_id.sort_bucketed) on a small sample of claims: rows
taken from row groups spread through a key's files, kept in their order (the dump's,
nearly by id) in all but the last of a few source files and shuffled in that one, sorted with
tiny buckets, part files and row groups so every step runs with several workers. So the
sources have row groups of every kind: read directly by each of their buckets, read
directly with a few rows spilled (a property's among items), and spilled whole (shuffled,
or ids of several lengths).

Checks that each bucket's rows, read as planned (directly from the row groups and from
the fragments), are exactly its rows of the sources whole, every column, in source order;
that the part files hold the sample's rows in stable id order; that bucketing resumes from
`bucketed.jsonl` and buckets of the format before are made again; that a sort interrupted
resumes from `sorted.jsonl`; and that the checks catch what they are for: a fragment
changed after bucketing, a bucket's rows counted otherwise than its ids, a row group's
rows of a bucket or a fragment's rows other than planned, a sort that is no permutation,
an id out of order, an id's rows reordered (not a stable sort), a value changed in a
written file or a row group missing from it, and a bucket's rows miscounted. Runs in a
temporary directory; reads the claims files only.

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
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "releases/20260928/hub/claims/all").resolve()
ROWS = int(sys.argv[2]) if len(sys.argv) > 2 else 50_000
N_SOURCES = 4
SAMPLE_RG_ROWS = 2_000  # small row groups, as each bucket gets a piece of every one


def make_sample(dst: Path) -> list[Path]:
    """N_SOURCES files of slices of 2 * N_SOURCES row groups spread through the key, in
    their order but in the last file, whose rows are shuffled (fixed seed) so each of its
    row groups has rows for every bucket; in row groups of SAMPLE_RG_ROWS."""
    groups = [(f, g) for f in sorted(SRC.glob("*.parquet")) for g in range(pq.ParquetFile(f).metadata.num_row_groups)]
    picks = [groups[round(k * (len(groups) - 1) / (2 * N_SOURCES - 1))] for k in range(2 * N_SOURCES)]
    rows = pa.concat_tables(pq.ParquetFile(f).read_row_group(g).slice(0, ROWS) for f, g in picks)
    per_file = rows.num_rows // N_SOURCES
    dst.mkdir(parents=True)
    out = []
    for j in range(N_SOURCES):
        t = rows.slice(j * per_file, per_file)
        if j == N_SOURCES - 1:
            t = t.take(pa.array(np.random.default_rng(0).permutation(t.num_rows)))
        path = dst / f"chunks-{j:02d}-{j:02d}.parquet"
        pq.write_table(t, path, row_group_size=SAMPLE_RG_ROWS)
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


def read_all(paths: list[Path]) -> pa.Table:
    return pa.concat_tables(pq.read_table(p).replace_schema_metadata() for p in paths)


def stable_sorted(t: pa.Table) -> pa.Table:
    """`t` stably sorted by id, by numpy (not Arrow's sort, which the sort uses; Polars'
    sort of the sample's claims took over 7 GB)."""
    ids = np.array(t.column("id").to_pylist(), dtype=object)
    return t.take(pa.array(np.argsort(ids, kind="stable")))


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
        expected = stable_sorted(read_all(sources))

        t0 = time.time()
        buckets = sbi.bucket_key(Table.CLAIMS, "all", sources)
        bdir = sbi._bucket_dir(Table.CLAIMS, "all")
        check("buckets.json has the format", buckets["format"] == sbi.BUCKETS_FORMAT and buckets["fragments"] == N_SOURCES)
        bucketed = {e["source"]: e for e in sbi._read_jsonl(bdir / "bucketed.jsonl")}
        kinds = {"direct": 0, "partly spilled": 0, "spilled": 0}
        for g in (g for e in bucketed.values() for g in e["row_groups"]):
            kinds["direct" if g["direct"] == g["buckets"] else "partly spilled" if g["direct"] else "spilled"] += 1
        print(f"bucketed in {time.time() - t0:.0f} s, {len(buckets['rows'])} buckets, row groups: {kinds}", flush=True)
        check("the sources have row groups of every kind", all(kinds.values()))
        lengths = [
            (f.name, g, sorted(set(pc.utf8_length(pq.ParquetFile(f).read_row_group(g, columns=["id"]).column("id")).to_pylist())), e["row_groups"][g])
            for f in sources[:-1]
            for e in [bucketed[f.name]]
            for g in range(pq.ParquetFile(f).metadata.num_row_groups)
        ]
        mixed = [(name, g, n, rg["direct"]) for name, g, n, rg in lengths if len(n) > 1]
        check(f"the sources have row groups of ids of several lengths, in order: {len(mixed)}, e.g. {mixed[:2]}", bool(mixed))
        whole = read_all(sources)
        ids, bounds, schema = whole.column("id"), buckets["bounds"], sbi._schema(sources)
        plan = sbi._bucket_parts(bdir, sources, buckets, bucketed)

        def bucket_range(i: int) -> tuple:
            return bounds[i - 1] if i else None, bounds[i] if i < len(bounds) else None

        def in_range(ids, lo, hi):
            keep = pc.is_valid(ids)
            keep = keep if lo is None else pc.and_(keep, pc.greater_equal(ids, lo))
            return keep if hi is None else pc.and_(keep, pc.less(ids, hi))

        read = {"whole": 0, "one run": 0, "several runs": 0}  # direct row groups, by the rows kept
        for i in range(len(buckets["rows"])):
            lo, hi = bucket_range(i)
            for src, _, _, groups in plan[i]:
                for g, _, direct in groups:
                    if direct:
                        rg_ids = pq.ParquetFile(src).read_row_group(g, columns=["id"]).column("id")
                        kept = np.flatnonzero(in_range(rg_ids, lo, hi).to_numpy())
                        one = kept[-1] - kept[0] + 1 == len(kept)
                        read["whole" if len(kept) == len(rg_ids) else "one run" if one else "several runs"] += 1
            got = sbi._read_bucket(f"bucket {i}", plan[i], schema, "id", lo, hi)
            if not got.equals(whole.filter(in_range(ids, lo, hi))):
                check(f"bucket {i} read as planned holds its rows of the sources, in source order", False)
        # (several runs: a row group of ids of several lengths read directly, below if not here)
        check(f"the buckets read row groups directly wholly theirs and partly, by the rows kept: {read}",
              read["whole"] and read["one run"])
        check("each bucket read as planned holds exactly its rows of the sources, every column, in source order",
              sum(buckets["rows"]) == whole.num_rows)

        # A row group partly a bucket's, its rows one run (sliced) or several (filtered)
        small = pa.table({"id": ["Q9", "Q10", "Q11", "Q12", "Q2", "Q1", "Q3", "Q13"], "v": list(range(8))})
        pq.write_table(small, "small.parquet")
        one_schema = sbi._schema([Path("small.parquet")])
        for name, lo, hi, want in (("one run", "Q10", "Q13", [1, 2, 3]), ("several runs", "Q12", "Q3", [3, 4, 7])):
            parts = [(Path("small.parquet"), None, None, [(0, len(want), True)])]
            got = sbi._read_bucket("small", parts, one_schema, "id", lo, hi)
            check(f"a row group partly a bucket's, its rows {name}: those rows, in order",
                  got.equals(small.take(pa.array(want))))
            if name == "one run":  # (a slice of the row group as read: its buffers whole)
                check("a row group's one run is a slice of it, not a copy",
                      got.get_total_buffer_size() > small.take(pa.array(want)).get_total_buffer_size())
            for delta in (1, -1):
                parts = [(Path("small.parquet"), None, None, [(0, len(want) + delta, True)])]
                fails(f"reading catches a row group's rows of a bucket miscounted ({name}, {delta:+d})",
                      lambda: sbi._read_bucket("small", parts, one_schema, "id", lo, hi), "planned")

        # A row group's rows of a bucket miscounted, read directly or spilled
        for direct, delta, words in ((True, 1, "planned"), (False, -1, "spilled")):
            i, j, k = next(
                (i, j, k)
                for i in range(len(buckets["rows"]))
                for j, part in enumerate(plan[i])
                for k, group in enumerate(part[3])
                if group[2] == direct and k == len(part[3]) - 1
            )
            parts = [list(part) for part in plan[i]]
            parts[j][3] = list(parts[j][3])
            g, n, _ = parts[j][3][k]
            parts[j][3][k] = (g, n + delta, direct)
            fails(f"reading catches a row group's rows of a bucket miscounted ({'direct' if direct else 'spilled'})",
                  lambda: sbi._read_bucket("b", parts, schema, "id", *bucket_range(i)), words)

        # A sort interrupted (a fragment changed, so its bucket fails), then resumed
        frag = sorted(bdir.glob(f"bucket-*.{sbi.FRAGMENT_FORMAT}"))[-1]
        kept = frag.read_bytes()
        t = sbi._read_streams(kept, schema)
        prop = t.column("property").to_pylist()
        prop[0] = "P0"
        f = t.schema.get_field_index("property")
        frag.write_bytes(sbi._ipc_stream(t.set_column(f, t.schema.field(f), pa.array(prop, t.schema.field(f).type)), t.schema).to_pybytes())
        fails("sorting catches a fragment changed after bucketing",
              lambda: sbi.sort_buckets(Table.CLAIMS, "all", sources, buckets), "differs from the bytes bucketed")
        before = len(sbi._read_jsonl(bdir / "sorted.jsonl"))
        frag.write_bytes(kept)
        sorted_buckets = sbi.sort_buckets(Table.CLAIMS, "all", sources, buckets)
        after = len(sbi._read_jsonl(bdir / "sorted.jsonl"))
        check(f"an interrupted sort resumes: {before} buckets kept, {after - before} sorted",
              before > 0 and after == len(buckets["rows"]))
        files = sbi.pack_key(Table.CLAIMS, "all", sources, buckets, sorted_buckets)
        print(f"sorted in {time.time() - t0:.0f} s, {len(files)} part files", flush=True)
        outputs = [sbi._out_dir(Table.CLAIMS) / "all" / f["name"] for f in files]
        check("part files hold the sample's rows in stable id order", read_all(outputs).equals(expected))
        check("bucket fragments removed once sorted", not list(bdir.glob("bucket-*")))
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
        src_rgs = pq.ParquetFile(sources[-1]).metadata.num_row_groups
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

        (bdir / "buckets.json").write_text(json.dumps(dict(again, format=sbi.FRAGMENT_FORMAT)))
        made = sbi.bucket_key(Table.CLAIMS, "all", sources)
        check("buckets of the format before are made again",
              made == first and len(log.read_text().splitlines()) == N_SOURCES)

        (bdir / "buckets.json").unlink()
        log.unlink()
        counted = json.loads((bdir / "bounds.json").read_text())
        counted["rows"][0] += 1
        counted["rows"][1] -= 1
        (bdir / "bounds.json").write_text(json.dumps(counted))
        fails("bucketing catches a bucket's rows other than its ids counted",
              lambda: sbi.bucket_key(Table.CLAIMS, "all", sources), "differ from the ids counted")
        os.chdir("/")
    print("all passed")


if __name__ == "__main__":
    main()
