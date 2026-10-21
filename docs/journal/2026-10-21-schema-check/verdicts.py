"""Run at feat/genson-target-schema, before is_acceptable_diff was removed.

Differential check: wikidata-pq's is_acceptable_diff (DeepDiff of schema_to_dict) against
polars-genson's check_schema_compatibility, on schemas derived from chunk 0's real inferred
claims schema and the maps' expected schemas. Each mutation: drop a field, add a field,
change a scalar dtype, make a field Null, at every path of the schema (sampled)."""
import functools, os, random, sys
os.environ["WIKIDATA_RELEASE"] = "20261012"
sys.path.insert(0, "/home/user/lmmx/wikidata-pq/src")
import polars as pl
from deepdiff import DeepDiff
import polars_genson as G
from wikidata import process as P

random.seed(0)

def old(stored, inferred):
    d1, d2 = G.schema_to_dict(stored), G.schema_to_dict(inferred)
    return d1 == d2 or P.is_acceptable_diff(DeepDiff(d1, d2, ignore_order=True))

def new(stored, inferred):
    return G.check_schema_compatibility(inferred, stored) == []

def paths(dt, prefix=()):
    """Every (path to a struct, field name) in a dtype, through lists."""
    if isinstance(dt, pl.Struct):
        for f in dt.fields:
            yield prefix, f.name
            yield from paths(f.dtype, prefix + (f.name,))
    elif isinstance(dt, (pl.List, pl.Array)):
        yield from paths(dt.inner, prefix + ("[]",))

def edit(dt, path, fn):
    if not path:
        return fn(dt)
    head, rest = path[0], path[1:]
    if head == "[]":
        return pl.List(edit(dt.inner, rest, fn))
    return pl.Struct([pl.Field(f.name, edit(f.dtype, rest, fn) if f.name == head else f.dtype) for f in dt.fields])

def mutations(schema):
    root = pl.Struct(schema)
    for parent, name in paths(root):
        def drop(s, name=name): return pl.Struct([f for f in s.fields if f.name != name])
        def add(s): return pl.Struct([*s.fields, pl.Field("zz_new", pl.String)])
        def null(s, name=name): return pl.Struct([pl.Field(f.name, pl.Null if f.name == name else f.dtype) for f in s.fields])
        def retype(s, name=name):
            swap = {pl.String: pl.Int64, pl.Int64: pl.Float64, pl.Float64: pl.Int64, pl.Boolean: pl.String}
            return pl.Struct([pl.Field(f.name, swap.get(f.dtype.base_type(), pl.String) if f.name == name and not f.dtype.is_nested() else f.dtype) for f in s.fields])
        for kind, fn in (("drop", drop), ("add", add), ("null", null), ("retype", retype)):
            yield kind, parent, name, pl.Schema(edit(root, parent, fn).to_schema())

chunk = sys.argv[1]
tmp = sys.argv[2]
nfp = functools.partial(G.normalise_from_parquet, map_encoding="kv")
nfp(input_path=chunk, column="claims", output_path=f"{tmp}/c.parquet", output_column="claims", wrap_root="claims",
    **P.CLAIMS_INFERENCE_OPTIONS, max_builders=100, typed=True, keep_columns=["id"],
    extract_invariants=P.LABEL_INVARIANTS, lookup_output_path=f"{tmp}/lk.parquet", prune=P.QUARANTINE_FIELDS,
    prune_output_path=f"{tmp}/pr.parquet")
inferred = pl.Schema(G.avro_to_polars_schema(G.read_parquet_metadata(f"{tmp}/c.parquet")["genson_avro_schema"], map_encoding="kv"))
cases = [("claims: real inferred vs stored", P.claims_schema, inferred)]
for kind, parent, name, mutated in mutations(inferred):
    cases.append((f"claims inferred, {kind} {'/'.join(parent + (name,))}", P.claims_schema, mutated))
for kind, parent, name, mutated in mutations(P.claims_schema):
    cases.append((f"claims stored as inferred, {kind} {'/'.join(parent + (name,))}", P.claims_schema, mutated))
for key, lists in (("labels", False), ("descriptions", False), ("aliases", True)):
    exp = P._map_schema(key, lists=lists)
    for kind, parent, name, mutated in mutations(exp):
        cases.append((f"{key}, {kind} {'/'.join(parent + (name,))}", exp, mutated))
agree = disagree = 0
by = {}
for label, stored, inf in cases:
    a, b = old(stored, inf), new(stored, inf)
    k = (a, b)
    by[k] = by.get(k, 0) + 1
    if a != b:
        disagree += 1
        if disagree <= 15:
            print(f"DISAGREE old={a} new={b}: {label}")
    else:
        agree += 1
print(f"{len(cases)} cases: {agree} agree, {disagree} disagree; (old, new) counts: {by}")
