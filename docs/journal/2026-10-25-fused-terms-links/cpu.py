#!/usr/bin/env python3
"""CPU time (user + sys, every thread) of the four tables' path, the pipeline's against
the fused build, on one chunk file: the median of REPS runs after one warm-up, in one
process (as a pipeline worker runs many chunks).

Pipeline: normalise_map_direct for labels, descriptions and aliases (genson reads the
JSON columns from the chunk file itself); the sitelinks column read; normalise_sitelinks
and its processed file's sink; prepare_for_partition + partition_parquet for each table.
Fused: write_terms on the columns in memory (the split's step: the columns exported to
Arrow large strings, the build, the files), then place_built_terms (the pipeline's: the
files renamed into place, the key dirs already made as after a release's first chunk,
or left in place; and the sidecars). Split:
the part file's write (dump.route_parts' zstd) as design B writes it (no claims column)
and without the four JSON columns too.

Usage: python cpu.py CHUNK.parquet [REPS]  (prints one JSON line)
"""

import contextlib
import io
import json
import os
import resource
import shutil
import statistics
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("WIKIDATA_RELEASE", "20261012")

import orjson  # noqa: E402
import polars as pl  # noqa: E402

from wikidata.config import PARTITION_COLS, Table  # noqa: E402
from wikidata.dump import ZSTD_LEVEL  # noqa: E402
from wikidata.partitioning import partition_parquet, prepare_for_partition  # noqa: E402
from wikidata.partitioning.fused import FUSED, INDEX, large_strings, place_built_terms, write_terms  # noqa: E402
from wikidata.process import atomic_sink_parquet, normalise_map_direct, normalise_sitelinks  # noqa: E402


def cpu() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def pipeline(src: Path, out: Path, name: str) -> dict[str, float]:
    t, c = {}, cpu()

    def lap(k):
        nonlocal c
        now = cpu()
        t[k] = now - c
        c = now

    with contextlib.redirect_stdout(io.StringIO()):
        for table in (Table.LABEL, Table.DESC, Table.ALIAS):
            normalise_map_direct(src, out / "processed" / table / name, key=FUSED[table], lists=table == Table.ALIAS)
            lap(f"normalise {table}")
        df = pl.read_parquet(src, columns=["id", "sitelinks"])
        lap("read sitelinks")
        atomic_sink_parquet(normalise_sitelinks(df).lazy(), out / "processed" / Table.LINKS / name)
        lap("normalise links")
        for table in FUSED:
            lf = prepare_for_partition(out / "processed" / table / name, table)
            partition_parquet(by=PARTITION_COLS[table], lf=lf, source_name=name, dst_dir=out / "part" / table, log_dir=out / "audit" / table)
            lap(f"partition {table}")
    return t


def main() -> None:
    src = Path(sys.argv[1])
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    name = "chunk_0.parquet"
    frame = pl.read_parquet(src)
    runs: dict[str, list[float]] = {}
    with tempfile.TemporaryDirectory(dir=os.environ.get("CPU_TMP")) as tmp:
        tmp = Path(tmp)
        for rep in range(reps + 1):
            out = tmp / "run"
            got = pipeline(src, out, name)
            shutil.rmtree(out)
            c = cpu()
            texts = {t: large_strings(frame.get_column(col)) for t, col in FUSED.items()}
            why = write_terms(frame.get_column("id"), texts, None, out / "routed", name)
            got["fused: write_terms (split)"] = cpu() - c
            assert why is None, why
            # The key dirs made first, as a release's first chunk leaves them
            for t in FUSED:
                for key, *_ in orjson.loads((out / "routed" / t / "chunk_0" / INDEX).read_bytes()):
                    (out / "part" / t / key).mkdir(parents=True, exist_ok=True)
            c = cpu()
            place_built_terms(out / "routed", name, out / "part", out / "audit")
            got["fused: place_built_terms, renamed (pipeline)"] = cpu() - c
            shutil.rmtree(out)
            write_terms(frame.get_column("id"), texts, None, out / "routed", name)
            c = cpu()
            place_built_terms(out / "routed", name, out / "part", out / "audit", in_place=True)
            got["fused: place_built_terms, in place (pipeline)"] = cpu() - c
            shutil.rmtree(out)
            for k, cols in {"chunk write, design B (no claims)": [c for c in frame.columns if c != "claims"], "chunk write, fused (id, entity)": ["id", "entity"]}.items():
                c = cpu()
                frame.select(cols).write_parquet(tmp / "chunk.parquet", compression="zstd", compression_level=ZSTD_LEVEL)
                got[k] = cpu() - c
            if rep:
                for k, v in got.items():
                    runs.setdefault(k, []).append(v)
    print(json.dumps({"chunk": src.stem, "reps": reps, **{k: round(statistics.median(v), 4) for k, v in runs.items()}}))


if __name__ == "__main__":
    main()
