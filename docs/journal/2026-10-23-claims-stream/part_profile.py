"""Partitioning's cost per table on processed chunks: for each chunk, process() it, then for
each table time (CPU and wall, this process) reading the processed file (with drop_nulls),
the pre-partition transform collected (read, explode, unnest), and the whole
partition_parquet (transform, sink, sidecar) to a scratch directory. Also counts each
table's rows, null and empty lists, and rows out.

Usage: part_profile.py <src> <out.jsonl> <rounds> <chunk files...>"""
import json, os, resource, shutil, sys, time
from pathlib import Path

os.environ["WIKIDATA_RELEASE"] = "20261012"
src, out, rounds = sys.argv[1], Path(sys.argv[2]).resolve(), int(sys.argv[3])
files = [Path(f).resolve() for f in sys.argv[4:]]
run = out.with_suffix(".run")
shutil.rmtree(run, ignore_errors=True)
work = run / "releases" / "20261012"
data, state, results = work / "data", work / "state", work / "results"
for d in (data, state, results):
    d.mkdir(parents=True)
os.chdir(run)
sys.path.insert(0, str(Path(src).resolve()))
import polars as pl  # noqa: E402
import wikidata.main as main  # noqa: E402
from wikidata.config import PARTITION_COLS, Table  # noqa: E402
from wikidata.partitioning import partition_parquet, prepare_for_partition  # noqa: E402
from wikidata.partitioning.transforms import TABLE_COLS  # noqa: E402


def cpu():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def timed(f):
    w, c = time.perf_counter(), cpu()
    v = f()
    return v, time.perf_counter() - w, cpu() - c


for i, f in enumerate(files):
    (data / f"chunk_{i}.parquet").symlink_to(f)
    main.process(chunk_idx=i, data_dir=data.resolve(), output_dir=results.resolve(), state_dir=state.resolve(), repo_id="bench/none")
    name = f"chunk_{i}.parquet"
    for t in Table:
        p = results / t / name
        rec = {"chunk": f.stem, "table": t.value}
        if t in TABLE_COLS:
            col = TABLE_COLS[t]
            s = pl.scan_parquet(p).select(
                pl.len().alias("rows"),
                pl.col(col).is_null().sum().alias("null_lists"),
                (pl.col(col).list.len() == 0).sum().alias("empty_lists"),
            ).collect().row(0, named=True)
            rec |= s
        for r in range(rounds):
            _, rec[f"read_w{r}"], rec[f"read_c{r}"] = timed(lambda: pl.scan_parquet(p).drop_nulls().collect())
            df, rec[f"prep_w{r}"], rec[f"prep_c{r}"] = timed(lambda: prepare_for_partition(p, t).collect())
            rec["rows_out"] = df.height
            if "language" in df.columns:
                rec["null_language_rows"] = df["language"].null_count()
            del df
            dst = run / "part" / t
            shutil.rmtree(run / "part", ignore_errors=True)
            _, rec[f"part_w{r}"], rec[f"part_c{r}"] = timed(lambda: partition_parquet(
                by=PARTITION_COLS[t], lf=prepare_for_partition(p, t), source_name=name,
                dst_dir=dst, log_dir=run / "part_audit" / t, arrow=t == Table.CLAIMS))
            shutil.rmtree(run / "part", ignore_errors=True)
        with open(out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
    for t in Table:
        (results / t / name).unlink(missing_ok=True)
shutil.rmtree(run, ignore_errors=True)
