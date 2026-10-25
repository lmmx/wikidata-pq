"""Close a simulated group's four tables three ways, timing each (CPU, one thread):
copy (push.core._copy_built, as #148), and re-encode (each chunk file read once, each
key's row groups appended to its own pyarrow writer, one row group up to RG rows)."""
import sys, time, shutil, json, os
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import pq_glue
import wikidata.push.core as core
from wikidata.config import PARTITION_COLS, Table
from wikidata.partitioning import fused

RG = 122_880

def reencode(table, processed, indexes, work, name, reader):
    key_col = PARTITION_COLS[Table(table)]
    writers = {}
    for p, keys in zip(processed, indexes):
        if not keys:
            continue
        if reader == "glue":
            t = pq_glue.read_parquet(p)
        else:
            t = pq.ParquetFile(p).read(use_threads=False)
        bounds = []  # each row group's first row, from the table's chunks (one a row group)
        at = 0
        # row offsets of row groups: from the index's rows per key (row groups per key)
        for key, n, lo, hi, size, first, nrg in keys:
            part = t.slice(at, n)
            at += n
            w = writers.get(key)
            if w is None:
                dst = work / key / name
                dst.parent.mkdir(parents=True, exist_ok=True)
                w = writers[key] = pq.ParquetWriter(dst, t.schema, compression="zstd", compression_level=3,
                                                    use_dictionary=[key_col], write_statistics=True)
            for b in part.to_batches():
                w.write_batch(b, row_group_size=RG)
    for w in writers.values():
        w.close()
    return {k: work / k / name for k in writers}

root = Path(sys.argv[1])
modes = sys.argv[2].split(",")
for table in ["labels", "descriptions", "aliases", "links"]:
    res = root / "results" / table
    processed = sorted(res.glob("chunk_*.parquet"), key=lambda p: int(p.stem.split("_")[1]))
    indexes = [fused.built_index(p.parent, p.name) for p in processed]
    out = {"table": table, "chunks": len(processed)}
    for mode in modes:
        work = root / "proto" / mode / table
        shutil.rmtree(work, ignore_errors=True)
        core.GROUP_COPY_THREADS = 1
        t0 = time.process_time(); w0 = time.time()
        if mode == "copy":
            merged = core._copy_built(processed, indexes, work, "g.parquet")
        else:
            merged = reencode(table, processed, indexes, work, "g.parquet", mode)
        out[mode + "_cpu"] = round(time.process_time() - t0, 3)
        out[mode + "_wall"] = round(time.time() - w0, 3)
        out[mode + "_bytes"] = sum(f.stat().st_size for f in merged.values())
        out[mode + "_rgs"] = sum(pq.read_metadata(f).num_row_groups for f in merged.values())
        out[mode + "_files"] = len(merged)
    # same rows, order and schema
    a, b = root / "proto" / modes[0] / table, root / "proto" / modes[-1] / table
    same = True
    for f in sorted(a.glob("*/g.parquet")):
        g = b / f.parent.name / f.name
        same &= pq.read_table(f).equals(pq.read_table(g)) and pq.read_schema(f).equals(pq.read_schema(g), check_metadata=True)
    out["same"] = same
    print(json.dumps(out), flush=True)
