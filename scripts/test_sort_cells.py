#!/usr/bin/env python3
"""Test a bucketed key planned from its sources' rows by cell (sort_by_id.read_cells, the
sidecar a writer can record as it writes: stage 3 of
docs/journal/2026-10-25-decoupled-thresholds.md), against the same key planned from its
ids: the same boundaries, the same buckets' rows and the same part files, byte for byte.
Then the same rows reordered in each source stably by cell (as the split would write
them): the same part files again, with fewer rows spilled and fewer row groups read to
plan. And the checks: a sidecar that does not fit its file is refused, and one whose
counts put rows in another bucket than their ids is caught as the rows are read.

Usage: python scripts/test_sort_cells.py [claims key dir] [rows per slice]
(default releases/20260928/hub/claims/all, 50,000)
"""

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "releases/20260928/hub/claims/all").resolve()
ROWS = int(sys.argv[2]) if len(sys.argv) > 2 else 50_000
N_SOURCES = 4
SAMPLE_RG_ROWS = 2_000


def check(name: str, ok: bool) -> None:
    print(f"{'ok  ' if ok else 'FAIL'} {name}", flush=True)
    if not ok:
        sys.exit(1)


def fails(name: str, f, words: str) -> None:
    try:
        f()
    except RuntimeError as e:
        check(f"{name} ({e})", any(w in str(e) for w in ([words] if isinstance(words, str) else words)))
        return
    check(f"{name}: no error raised", False)


def sample(dst: Path, by_cell: bool, cell_chars: int) -> list[Path]:
    """test_sort's sample: N_SOURCES files of row groups spread through the key, the last
    shuffled; with `by_cell`, each file's rows reordered stably by cell."""
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
        if by_cell:
            cells = pc.utf8_slice_codeunits(t.column("id"), 0, cell_chars)
            t = t.take(pc.sort_indices(pa.table({"c": cells}), [("c", "ascending")]))  # (stable)
        path = dst / f"chunks-{j:02d}-{j:02d}.parquet"
        pq.write_table(t, path, row_group_size=SAMPLE_RG_ROWS)
        out.append(path)
    return out


def run(sbi, Table, sources: list[Path]) -> dict:
    """The key sorted through buckets (its plan's, at the test's sizes), from a clean
    sort directory: its buckets, row groups planned and read, and part files' sha256."""
    shutil.rmtree(sbi.SORT_DIR, ignore_errors=True)
    files = sbi.sort_bucketed(Table.CLAIMS, "all", sources)
    bdir = sbi._bucket_dir(Table.CLAIMS, "all")
    made = json.loads((bdir / "buckets.json").read_text())
    bucketed = sbi._read_jsonl(bdir / "bucketed.jsonl")
    groups = [g for e in bucketed for g in e["row_groups"]]
    out = sbi._out_dir(Table.CLAIMS) / "all"
    return {
        "bounds": made["bounds"],
        "rows": made["rows"],
        "spilled": sum(c for g in groups for b, c in zip(g["buckets"], g["rows"]) if b not in g["direct"]),
        "spilling_groups": sum(1 for g in groups if g["direct"] != g["buckets"]),
        "groups": len(groups),
        "files": [(f["name"], hashlib.sha256((out / f["name"]).read_bytes()).hexdigest()) for f in files],
    }


def main() -> None:
    with TemporaryDirectory() as tmp:
        os.chdir(tmp)
        os.environ["WIKIDATA_RELEASE"] = "cellstest"
        os.environ["WIKIDATA_FINALISE_WORKERS"] = "3"
        from wikidata import compact
        from wikidata import sort_by_id as sbi
        from wikidata.config import Table

        key_dir = sbi._src_dir(Table.CLAIMS) / "all"
        sources = sample(key_dir, False, sbi.SORT_CELL_CHARS)
        est = sbi.plan_key(Table.CLAIMS, sources).decoded
        sbi.SORT_BUCKET_BYTES = est // 20
        sbi.COMPACT_FILE_BYTES = int(est * sbi.SORT_HUB_BYTES_PER_DECODED["claims"]) // 4
        compact.COMPACT_ROW_GROUP_BYTES = sbi.COMPACT_ROW_GROUP_BYTES = 4 * 1024**2
        sbi.SORT_BUCKET_WRITE_BYTES = 1024**2

        by_ids = run(sbi, Table, sources)
        for s in sources:
            sbi.write_cells(s)
        by_cells = run(sbi, Table, sources)
        print(f"{len(by_ids['rows'])} buckets, {len(by_ids['files'])} part files; {by_ids['groups']} row groups,"
              f" {by_ids['spilling_groups']} spilling", flush=True)
        check("planned from the cells: the same boundaries and buckets' rows as from the ids",
              (by_cells["bounds"], by_cells["rows"]) == (by_ids["bounds"], by_ids["rows"]))
        check("planned from the cells: the same row groups spilled, the same rows",
              (by_cells["spilled"], by_cells["spilling_groups"]) == (by_ids["spilled"], by_ids["spilling_groups"]))
        check("planned from the cells: the same part files, byte for byte", by_cells["files"] == by_ids["files"])

        # The same rows, each source reordered stably by cell, as the split would write them
        shutil.rmtree(key_dir)
        reordered = sample(key_dir, True, sbi.SORT_CELL_CHARS)
        for s in reordered:
            sbi.write_cells(s)
        by_cell_order = run(sbi, Table, reordered)
        check(f"sources reordered by cell: the same part files, byte for byte"
              f" (rows spilled {by_ids['spilled']:,} -> {by_cell_order['spilled']:,},"
              f" row groups spilling {by_ids['spilling_groups']} -> {by_cell_order['spilling_groups']} of {by_cell_order['groups']})",
              by_cell_order["files"] == by_ids["files"] and by_cell_order["rows"] == by_ids["rows"]
              and by_cell_order["spilled"] < by_ids["spilled"])

        # A sidecar that does not fit its file is refused
        path = sbi._cells_path(reordered[0])
        good = path.read_text()
        made = json.loads(good)
        made["row_groups"][0][0][1] += 1
        path.write_text(json.dumps(made))
        fails("a sidecar whose row group's rows differ from the footer's is refused",
              lambda: sbi.read_cells(reordered[0]), "does not fit")
        path.write_text(json.dumps(dict(json.loads(good), cell_chars=sbi.SORT_CELL_CHARS + 1)))
        fails("a sidecar of another cell width is refused", lambda: sbi.read_cells(reordered[0]), "does not fit")

        # One whose counts put rows in another bucket than their ids: caught as read
        made = json.loads(good)
        bounds = by_cell_order["bounds"]
        g, runs = next((g, r) for g, r in enumerate(made["row_groups"]) if len({sbi.bisect.bisect_right(bounds, c) for c, _ in r}) == 1)
        b = sbi.bisect.bisect_right(bounds, runs[0][0])
        other = bounds[b] if b < len(bounds) else bounds[b - 1][:-1] if len(bounds[b - 1]) > 1 else "P"
        moved = sorted([[c, k] for c, k in runs] + [[other, 1]])
        moved[[c for c, _ in moved].index(runs[0][0])][1] -= 1
        made["row_groups"][g] = [[c, k] for c, k in moved if k > 0]
        path.write_text(json.dumps(made))
        fails("a sidecar putting a row in another bucket than its id is caught as the rows are read",
              lambda: run(sbi, Table, reordered), ("planned", "other buckets than its cells"))
        path.write_text(good)
        os.chdir("/")
    print("all passed")


if __name__ == "__main__":
    main()
