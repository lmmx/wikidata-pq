#!/usr/bin/env python3
"""Test the bucketed sort (sort_by_id.sort_bucketed) on a small sample of claims: rows
taken from a key's files and shuffled across a few source files, sorted with tiny
buckets, part files and row groups so every step runs with several workers.

Checks that the part files hold the sample's rows in stable id order, that bucketing
resumes from `bucketed.jsonl`, that the end check catches a row changed in a bucket
(the step no other check covers), and the end check's fallback for buckets made before
2026-10-05. Runs in a temporary directory; reads the claims files only.

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

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "releases/20260928/hub/claims/all").resolve()
ROWS = int(sys.argv[2]) if len(sys.argv) > 2 else 50_000
N_SOURCES = 4
SAMPLE_RG_ROWS = 2_000  # small row groups, as each bucket gets a piece of every one


def make_sample(dst: Path) -> list[Path]:
    """N_SOURCES files of two slices each, from files spread through the key, the later
    slice first, so each file's ids run down and every file spans the key's id range;
    in row groups of SAMPLE_RG_ROWS."""
    files = sorted(SRC.glob("*.parquet"))
    picks = [files[round(k * (len(files) - 1) / (2 * N_SOURCES - 1))] for k in range(2 * N_SOURCES)]
    slices = [pq.ParquetFile(f).read_row_group(0).slice(0, ROWS) for f in picks]
    schema = slices[0].schema
    dst.mkdir(parents=True)
    out = []
    for j in range(N_SOURCES):
        path = dst / f"chunks-{j:02d}-{j:02d}.parquet"
        with pq.ParquetWriter(path, schema, compression="zstd") as w:
            for t in (slices[N_SOURCES + j], slices[j]):
                w.write_table(t, row_group_size=SAMPLE_RG_ROWS)
        out.append(path)
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
        os.environ["WIKIDATA_SORT_WORKERS"] = "3"
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
        check("buckets.json has the sources' sums", "sums" in buckets and buckets["fragments"] == N_SOURCES)
        check("bucket fragments removed once sorted", not list(bdir.glob("bucket-*.parquet")))

        sbi._check_whole("fallback", sources, {}, [{}], outputs)
        check("fallback end check passes on the part files", True)
        try:
            sbi._check_whole("fallback", sources, {}, [{}], outputs[1:])
            check("fallback end check fails without a part file", False)
        except RuntimeError:
            check("fallback end check fails without a part file", True)

        shutil.rmtree(SORT_DIR)
        first = sbi.bucket_key(Table.CLAIMS, "all", sources)
        frag = max(bdir.glob("bucket-*.parquet"), key=lambda p: p.stat().st_size)
        n_rg = pq.ParquetFile(frag).metadata.num_row_groups
        src_rgs = pq.ParquetFile(sources[0]).metadata.num_row_groups
        check(f"buffered writes: {frag.name} has {n_rg} row groups (source {src_rgs})", n_rg <= src_rgs // 10)
        (bdir / "buckets.json").unlink()
        log = bdir / "bucketed.jsonl"
        lines = log.read_text().splitlines()
        log.write_text("\n".join(lines[:-1]) + "\n")
        again = sbi.bucket_key(Table.CLAIMS, "all", sources)
        check("bucketing resumes from bucketed.jsonl, same buckets", again == first)
        check("bucketing redid only the missing file", len(log.read_text().splitlines()) == N_SOURCES)

        t = pq.read_table(frag)
        prop = t.column("property").to_pylist()
        prop[0] = "P0"
        i = t.schema.get_field_index("property")
        pq.write_table(t.set_column(i, t.schema.field(i), pa.array(prop, t.schema.field(i).type)), frag)
        try:
            sbi.pack_key(Table.CLAIMS, "all", sources, again, sbi.sort_buckets(Table.CLAIMS, "all", sources, again))
            check("end check catches a row changed in a bucket", False)
        except RuntimeError as e:
            check(f"end check catches a row changed in a bucket ({e})", "rows differ" in str(e))
        os.chdir("/")
    print("all passed")


if __name__ == "__main__":
    main()
