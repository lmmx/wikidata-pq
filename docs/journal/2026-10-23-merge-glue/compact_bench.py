"""Compaction (compact.rewrite_table) of KEEP'd group files: wall, CPU (with workers),
and the output files' row groups, rows and bytes per table.

Usage: compact_bench.py <src> <staging> <workdir> [tables...]"""
import os, resource, shutil, sys, time
from pathlib import Path

os.environ["WIKIDATA_RELEASE"] = "20261012"
src, staging, work = sys.argv[1], Path(sys.argv[2]).resolve(), Path(sys.argv[3]).resolve()
sys.path.insert(0, str(Path(src).resolve()))
import pyarrow.parquet as pq  # noqa: E402
import wikidata.compact as compact  # noqa: E402

def cpu():
    s, c = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    return s.ru_utime + s.ru_stime + c.ru_utime + c.ru_stime


if __name__ == "__main__":
    shutil.rmtree(work, ignore_errors=True)
    (work / "src").mkdir(parents=True)
    compact.COMPACT_DIR = work
    tables = sys.argv[4:] or ["labels", "descriptions", "aliases", "links", "entities"]


    for t in tables:
        (work / "src" / t).symlink_to(staging / t)
        table = compact.Table(t)
        w, c = time.perf_counter(), cpu()
        compact.rewrite_table(table, 65)
        w, c = time.perf_counter() - w, cpu() - c
        outs = sorted((work / "out" / t).rglob("*.parquet"))
        rgs = sum(pq.ParquetFile(p).metadata.num_row_groups for p in outs)
        rows = sum(pq.ParquetFile(p).metadata.num_rows for p in outs)
        mb = sum(p.stat().st_size for p in outs) / 2**20
        print(f"{t:13s} {w:6.2f}s wall {c:6.2f}s cpu  {len(outs)} files {rgs} row groups {rows} rows {mb:.1f} MB", flush=True)
