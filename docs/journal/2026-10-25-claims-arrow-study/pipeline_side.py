"""Pipeline side: the claims call now, the read of the claims column, and the typed
claims written into / read back from a chunk file instead of the JSON text."""
import json, os, resource, statistics, sys, tempfile, time
from pathlib import Path
import polars as pl, pyarrow as pa, pyarrow.parquet as pq
import wikidata.process as P
from wikidata.partitioning.claims import statements

DATA = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(exist_ok=True)
chunks = sys.argv[3].split(",")
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

res = {}
for c in chunks:
    path = DATA / f"{c}.parquet"
    r = {}
    rs = []
    for i in range(REPS + 1):
        claims, t = timed(lambda: claims_call(path))
        if i: rs.append(t)
    r["claims_call"] = med(rs)
    rs = [timed(lambda: pq.read_table(path, columns=["claims"]))[1] for _ in range(REPS)]
    r["read_claims_json_pyarrow"] = med(rs)
    rs = [timed(lambda: pl.read_parquet(path, columns=["claims"]))[1] for _ in range(REPS)]
    r["read_claims_json_polars"] = med(rs)
    src = pl.read_parquet(path)
    r["rows"] = src.height
    r["claims_json_bytes_uncompressed"] = int(src["claims"].str.len_bytes().sum())
    # the chunk write as dump.route_parts writes it: all 7 columns, and without claims
    with tempfile.TemporaryDirectory(dir=OUT) as d:
        f_all, f_no, f_typed = (Path(d) / n for n in ("all.parquet", "noclaims.parquet", "typed.parquet"))
        rs = [timed(lambda: src.write_parquet(f_all, compression="zstd", compression_level=3))[1] for _ in range(REPS)]
        r["write_chunk_json"] = med(rs)
        rs = [timed(lambda: src.drop("claims").write_parquet(f_no, compression="zstd", compression_level=3))[1] for _ in range(REPS)]
        r["write_chunk_no_claims"] = med(rs)
        typed = pl.concat([src.drop("claims"), pl.from_arrow(claims).select("claims")], how="horizontal")
        assert (typed["id"] == pl.from_arrow(claims)["id"]).all()
        rs = [timed(lambda: typed.write_parquet(f_typed, compression="zstd", compression_level=3))[1] for _ in range(REPS)]
        r["write_chunk_typed_polars"] = med(rs)
        f_typed_pa = Path(d) / "typed_pa.parquet"
        tt = typed.to_arrow()
        rs = [timed(lambda: pq.write_table(tt, f_typed_pa, compression="zstd", compression_level=3))[1] for _ in range(REPS)]
        r["write_chunk_typed_pyarrow"] = med(rs)
        r["bytes_chunk_json"] = f_all.stat().st_size
        r["bytes_chunk_no_claims"] = f_no.stat().st_size
        r["bytes_chunk_typed_polars"] = f_typed.stat().st_size
        r["bytes_chunk_typed_pyarrow"] = f_typed_pa.stat().st_size
        meta = pq.read_metadata(f_all)
        rg = [meta.row_group(g) for g in range(meta.num_row_groups)]
        r["claims_json_col_bytes"] = sum(g.column(i).total_compressed_size for g in rg for i in range(g.num_columns) if g.column(i).path_in_schema == "claims")
        rs = [timed(lambda: pq.read_table(f_typed_pa, columns=["id", "claims"]))[1] for _ in range(REPS)]
        r["read_claims_typed_pyarrow"] = med(rs)
        rs = [timed(lambda: pl.read_parquet(f_typed, columns=["id", "claims"]).to_arrow())[1] for _ in range(REPS)]
        r["read_claims_typed_polars"] = med(rs)
        # identity: the statements from typed claims read back, against the call's
        back = pq.read_table(f_typed_pa, columns=["id", "claims"])
        a, b = statements(claims), statements(back)
        r["statements_identical_pyarrow_roundtrip"] = a.equals(b) and a.schema.equals(b.schema)
        back2 = pl.read_parquet(f_typed, columns=["id", "claims"]).to_arrow()
        b2 = statements(back2)
        r["statements_identical_polars_roundtrip"] = a.equals(b2) and a.schema.equals(b2.schema)
        r["call_output_types"] = str(claims.schema.field("claims").type)[:120]
        r["pyarrow_roundtrip_types"] = str(back.schema.field("claims").type)[:120]
        rs = [timed(lambda: statements(claims))[1] for _ in range(REPS)]
        r["statements"] = med(rs)
    res[c] = r
    print(c, json.dumps(r), flush=True)
(OUT / "pipeline_side.json").write_text(json.dumps(res, indent=1))
