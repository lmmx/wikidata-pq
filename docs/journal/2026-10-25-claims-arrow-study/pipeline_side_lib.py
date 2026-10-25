"""Pipeline side: the claims call now, the read of the claims column, and the typed
claims written into / read back from a chunk file instead of the JSON text."""
import json, os, resource, statistics, sys, tempfile, time
from pathlib import Path
import polars as pl, pyarrow as pa, pyarrow.parquet as pq
import wikidata.process as P
from wikidata.partitioning.claims import statements

REPS = int(os.environ.get("REPS", 3))

def cpu():
    r = resource.getrusage(resource.RUSAGE_SELF); return r.ru_utime, r.ru_stime

def timed(fn):
    u0, s0 = cpu(); t = time.perf_counter(); x = fn(); w = time.perf_counter() - t
    u1, s1 = cpu(); return x, {"user": u1 - u0, "sys": s1 - s0, "wall": w}

def med(rs):
    return {k: round(statistics.median(r[k] for r in rs), 3) for k in rs[0]}

def claims_call(path):
    with tempfile.TemporaryDirectory() as d:
        stream = P.normalise_in_schema("claims", path, column="claims", output_path=None,
            output_format="arrow", output_column="claims", wrap_root="claims",
            **P.CLAIMS_INFERENCE_OPTIONS, profile=False, max_builders=100, typed=True,
            keep_columns=["id"], extract_invariants=P.LABEL_INVARIANTS,
            lookup_output_path=Path(d) / "l.parquet", prune=P.QUARANTINE_FIELDS,
            prune_output_path=Path(d) / "p.parquet", avro_schema=P.CLAIMS_AVRO,
            map_encoding="kv", schema=P.claims_schema, unnest=True)
        return pa.RecordBatchReader.from_stream(stream).read_all()

