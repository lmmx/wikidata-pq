"""Chunk 0's claims through claims_base + partition_parquet in row slices the base code can
finish, base and new each in a fresh process (run per slice with: python c0_slices.py run
<slice file> <src> <run dir>). `make` writes the slices; `join` checks the new whole-chunk
claims partition equals the slices' new outputs concatenated."""
import os, sys
from pathlib import Path
os.environ["WIKIDATA_RELEASE"] = "20261012"
cmd = sys.argv[1]
if cmd == "make":
    import polars as pl
    df = pl.read_parquet(sys.argv[2]); n = int(sys.argv[3]); step = -(-df.height // n)
    Path("slices").mkdir(exist_ok=True)
    for i in range(n):
        df.slice(i * step, step).write_parquet(f"slices/chunk_0_{i}.parquet")
    print("slices", [min(step, df.height - i * step) for i in range(n)])
elif cmd == "run":
    src, run = Path(sys.argv[2]).resolve(), Path(sys.argv[4]).resolve()
    sys.path.insert(0, str(Path(sys.argv[3]).resolve()))
    import resource, shutil, time
    import wikidata.partitioning.claims as c
    from wikidata.config import PARTITION_COLS, Table
    from wikidata.partitioning import partition_parquet, prepare_for_partition
    assert c.__file__.startswith(str(Path(sys.argv[3]).resolve()))
    shutil.rmtree(run, ignore_errors=True); run.mkdir(parents=True); os.chdir(run)
    t = time.perf_counter()
    partition_parquet(by=PARTITION_COLS[Table.CLAIMS], lf=prepare_for_partition(src, Table.CLAIMS), source_name="chunk_0.parquet", dst_dir=Path("out"), log_dir=Path("audit"))
    print(f"{src.name} {sys.argv[3]}: {time.perf_counter() - t:.1f} s, peak {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024:,} MB")
elif cmd == "join":
    import polars as pl
    whole = pl.read_parquet(sys.argv[2])
    parts = pl.concat([pl.read_parquet(p) for p in sys.argv[3:]])
    print("whole chunk equals slices concatenated:", whole.schema == parts.schema and whole.equals(parts, null_equal=True), whole.shape)
