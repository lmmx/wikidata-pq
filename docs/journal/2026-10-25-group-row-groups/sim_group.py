"""A release-sized main-set group (N chunks: the bench's routed main chunks, built in the
one-file layout, cycled and hard linked) closed by push.core._copy_built, as #148 does.
Writes OUT/copy/{table}/{key}/g.parquet and prints the layout per table."""
import json, os, sys, time, statistics as st
from pathlib import Path
import pyarrow.parquet as pq
import wikidata.push.core as core
from wikidata.partitioning import fused
from wikidata.partitioning.fused import FUSED

built, out, n = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
chunks = sys.argv[4].split(",")
for table in FUSED:
    res = out / "results" / str(table)
    res.mkdir(parents=True, exist_ok=True)
    processed = []
    for i in range(n):
        c = chunks[i % len(chunks)]
        src = built / c / str(table) / "chunk.parquet"
        dst = res / f"chunk_{i}.parquet"
        if src.exists() and not dst.exists():
            os.link(src, dst)
        idx = fused.index_path(res, dst.name)
        if not idx.exists():
            os.link(fused.index_path(built / c / str(table), "chunk.parquet"), idx)
        processed.append(dst)
    indexes = [fused.built_index(p.parent, p.name) for p in processed]
    work = out / "copy" / str(table)
    t0 = time.process_time(); w0 = time.time()
    merged = core._copy_built(processed, indexes, work, "g.parquet")
    cpu, wall = time.process_time() - t0, time.time() - w0
    rgs, foot, size, rows = [], [], [], []
    for k, f in merged.items():
        m = pq.read_metadata(f)
        rgs.append(m.num_row_groups); rows.append(m.num_rows)
        with open(f, "rb") as fh:
            fh.seek(-8, 2); foot.append(int.from_bytes(fh.read(4), "little"))
        size.append(f.stat().st_size)
    q = sorted(rgs)
    print(json.dumps({"table": str(table), "files": len(merged), "copy_cpu": round(cpu, 2), "copy_wall": round(wall, 2),
        "rgs": sum(rgs), "rgs_median": q[len(q)//2], "rgs_max": q[-1], "rows": sum(rows),
        "rows_per_rg_median": sorted(r / g for r, g in zip(rows, rgs))[len(q)//2],
        "footer_bytes": sum(foot), "file_bytes": sum(size), "footer_per_rg": round(sum(foot) / sum(rgs), 1)}), flush=True)
