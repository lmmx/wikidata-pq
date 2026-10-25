"""A part's claims partition file written by the split (dump.route_parts), so the chunk
file holds no claims column and the pipeline makes no claims call.

The pipeline's claims step (process.normalise_claims_direct) reads a chunk's claims JSON
back from its file and has polars-genson decode it against `claims.avsc`, then lays it out
as the claims partition (partitioning.claims.statements) and writes it (write_table). The
split has that JSON in memory as it writes the chunk. Here it is decoded straight into the
same Arrow arrays by the reshape (reshape/src/claims_arrow.rs: genson's strict decode,
stricter), then laid out and written by the same code. The file is the pipeline's, byte
for byte, at the same thread count (see `batch_rows`); scripts/test_claims_build.py checks
it on the bench's chunks.

A part whose claims the reshape does not decode exactly as genson would (`claims_arrays`
gives None) keeps its claims as JSON, and the pipeline decodes them as before. So do all
parts when WIKIDATA_SPLIT_CLAIMS=json, or without claims.avsc.
"""

import os
import threading
from contextlib import contextmanager
from pathlib import Path

import polars as pl
import pyarrow as pa
from wikidata_reshape import claims_batches, rayon_threads

from .claims_schema import CLAIMS_AVRO, check_ids, claims_schema
from .partitioning.claims import statements
from .partitioning.core import write_table

# Where a part's claims partition file is, beside the part: `claims/` in its directory
CLAIMS_DIR = "claims"

# The split decodes the claims it can (built), or leaves them all as JSON
SPLIT_CLAIMS = os.environ.get("WIKIDATA_SPLIT_CLAIMS", "built")
if SPLIT_CLAIMS not in ("built", "json"):
    raise SystemExit(f"WIKIDATA_SPLIT_CLAIMS is built or json, not {SPLIT_CLAIMS!r}")


def _branches(avro) -> tuple[bool, object | None]:
    """An Avro type as (nullable, its one other branch, or None for ["null"])."""
    if isinstance(avro, list):
        others = [b for b in avro if b != "null"]
        if len(others) > 1:
            raise ValueError(f"a union genson's strict decode does not compile: {avro}")
        return "null" in avro, (others[0] if others else None)
    return False, avro


_SCALARS = {
    "string": ("str", pl.String),
    "int": ("int", pl.Int64),
    "long": ("int", pl.Int64),
    "float": ("float", pl.Float64),
    "double": ("float", pl.Float64),
}


def _phantom(dtype: pl.DataType) -> tuple:
    """A target field claims.avsc lacks: null in every row (genson's conform), and a
    member for it a misfit."""
    return ("phantom", True, _shape(dtype))


def _shape(dtype: pl.DataType) -> tuple:
    """A target dtype's plan with no Avro type: the column a phantom field is built as."""
    scalars = {pl.String: "str", pl.Int64: "int", pl.Float64: "float", pl.Null: "null"}
    if dtype in scalars:
        return (scalars[dtype], True)
    if isinstance(dtype, pl.Struct):
        return ("record", True, [(f.name, _phantom(f.dtype)) for f in dtype.fields])
    if isinstance(dtype, pl.List):
        entry = dtype.inner
        if isinstance(entry, pl.Struct) and [f.name for f in entry.fields] == ["key", "value"]:
            return ("map", True, _shape(entry.fields[1].dtype))
        return ("array", True, _shape(entry))
    raise ValueError(f"a target field of {dtype} that claims.avsc lacks")


def _plan(dtype: pl.DataType, avro) -> tuple:
    """The decode plan (claims_arrow.rs plan_of) of a target dtype and its Avro type."""
    nullable, inner = _branches(avro)
    if inner is None:
        if dtype != pl.Null:
            raise ValueError(f'["null"] in claims.avsc at a target {dtype}')
        return ("null", True)
    if isinstance(inner, str):
        kind, want = _SCALARS[inner]
        if dtype != want:
            raise ValueError(f"claims.avsc's {inner} at a target {dtype}")
        return (kind, nullable)
    kind = inner["type"]
    if kind == "record":
        if not isinstance(dtype, pl.Struct):
            raise ValueError(f"a record at a target {dtype}")
        avro_fields = {f["name"]: f["type"] for f in inner["fields"]}
        target = {f.name: f.dtype for f in dtype.fields}
        if extra := set(avro_fields) - set(target):
            raise ValueError(f"claims.avsc fields the target lacks: {sorted(extra)}")
        fields = [
            (name, _plan(t, avro_fields[name]) if name in avro_fields else _phantom(t))
            for name, t in target.items()
        ]
        return ("record", nullable, fields)
    if not isinstance(dtype, pl.List):
        raise ValueError(f"an {kind} at a target {dtype}")
    if kind == "array":
        return ("array", nullable, _plan(dtype.inner, inner["items"]))
    if kind == "map":
        entry = dtype.inner
        if not (isinstance(entry, pl.Struct) and [f.name for f in entry.fields] == ["key", "value"]):
            raise ValueError(f"a map at a target {dtype}")
        return ("map", nullable, _plan(entry.fields[1].dtype, inner["values"]))
    raise ValueError(f"an Avro {kind} in claims.avsc")


def _compile():
    """The plan of the claims column, and the Arrow schema of genson's output (id, claims)."""
    if CLAIMS_AVRO is None or SPLIT_CLAIMS == "json":
        return None, None
    import json

    root = json.loads(CLAIMS_AVRO)
    (claims_field,) = [f for f in root["fields"] if f["name"] == "claims"]
    plan = _plan(claims_schema["claims"], claims_field["type"])
    schema = pl.DataFrame(schema={"id": pl.String, **claims_schema}).to_arrow(
        compat_level=pl.CompatLevel.oldest()
    ).schema
    return plan, schema


PLAN, SCHEMA = _compile()


def batch_rows(rows: int) -> int:
    """The rows of each of genson's batches of a chunk's claims (polars-genson's
    parquet_io.rs: `rows.div_ceil(rayon threads * 4)`). The claims partition file's bytes
    depend on where its batches end, so they depend on the threads: the pipeline's worker's
    rayon pool and this process's have the same default (RAYON_NUM_THREADS, else the CPUs
    available), so the file is the pipeline's on the same machine. The published files are
    rewritten by finalise, so they do not depend on it."""
    return max(1, -(-rows // (rayon_threads() * 4)))


def _arrays(plan: tuple, t: pa.DataType, nodes) -> pa.Array:
    """A column of `t` from the reshape's nodes (claims_arrow.rs Col::export), depth first."""
    length, nulls, valid, offsets, values = next(nodes)
    vb = None if valid is None else pa.py_buffer(valid)
    kind = plan[0]
    if kind == "phantom":
        plan = plan[2]
        kind = plan[0]
    if kind == "null":
        return pa.nulls(length)
    if kind == "str":
        return pa.Array.from_buffers(t, length, [vb, pa.py_buffer(offsets), pa.py_buffer(values)], nulls)
    if kind in ("int", "float"):
        return pa.Array.from_buffers(t, length, [vb, pa.py_buffer(values)], nulls)
    if kind == "record":
        children = [_arrays(p, f.type, nodes) for (_, p), f in zip(plan[2], t)]
        return pa.Array.from_buffers(t, length, [vb], nulls, children=children)
    entry = t.value_type
    if kind == "map":
        keys = _arrays(("str", False), entry.field(0).type, nodes)
        vals = _arrays(plan[2], entry.field(1).type, nodes)
        items = pa.Array.from_buffers(entry, len(keys), [None], 0, children=[keys, vals])
    else:
        items = _arrays(plan[2], entry, nodes)
    return pa.Array.from_buffers(t, length, [vb, pa.py_buffer(offsets)], nulls, children=[items])


def claims_arrays(
    ids: pl.Series, text: pa.Array, rows: pa.Array | None = None
) -> tuple[pa.Table | None, str | None]:
    """A part's claims as the claims call gives them (`id`, `claims`, in genson's
    batches), or None and why they do not all fit: `ids` the part's, `text` the claims
    column (large strings) its rows are at (`rows`, int64 positions in it, or all of it)."""
    if PLAN is None:
        return None, "not built (WIKIDATA_SPLIT_CLAIMS=json, or no claims.avsc)"
    if isinstance(text, pa.ChunkedArray):
        text = text.combine_chunks()
    if text.null_count:
        return None, "a null claims cell"
    _, offsets, data = text.buffers()
    index = None
    if rows is not None:
        if rows.offset or rows.null_count:
            rows = pa.array(rows.to_numpy(zero_copy_only=False), pa.int64())
        index = rows.buffers()[1]
    n = len(text) if rows is None else len(rows)
    first = text.offset
    size = batch_rows(n)
    batches, why = claims_batches(PLAN, data, offsets, first, n, size, index)
    if batches is None:
        return None, why
    ids = ids.to_arrow(compat_level=pl.CompatLevel.oldest())
    claims_type = SCHEMA.field("claims").type
    out = []
    for k, nodes in enumerate(batches):
        claims = _arrays(PLAN, claims_type, iter(nodes))
        out.append(pa.RecordBatch.from_arrays([ids.slice(k * size, len(claims)), claims], schema=SCHEMA))
    return pa.Table.from_batches(out, schema=SCHEMA), None


def text_of(part: pl.DataFrame) -> pa.Array:
    """A frame's claims column as large strings (copied out of Polars' string views)."""
    return part.get_column("claims").to_arrow(compat_level=pl.CompatLevel.oldest())


def write_claims(ids: pl.Series, text: pa.Array, rows: pa.Array | None, dst: Path) -> str | None:
    """Write a part's claims partition file to `dst` as process.normalise_claims_direct
    writes it (its ids checked first, as there), by way of a temporary file: the part's
    `ids`, and its claims text (`text` at `rows`, see claims_arrays). None once written,
    else why its claims stay as JSON (and nothing is written)."""
    with BUDGET.hold(_cost(text, rows)):
        claims, why = claims_arrays(ids, text, rows)
        if claims is None:
            return why
        check_ids(ids.n_unique(), pl.from_arrow(claims.select(["id"])), table="claims")
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".tmp")
        write_table(statements(claims), tmp)
        del claims
        tmp.replace(dst)
    return None


def _cost(text: pa.Array, rows: pa.Array | None) -> int:
    """A part's claims' bytes in memory while they are built and written, estimated from
    their text: the arrays hold about 1.45 times the text, and the write about as much
    again (the pipeline's claims step on c0 peaked at about 3 times its text)."""
    n = text.nbytes if rows is None else text.nbytes * len(rows) // max(1, len(text))
    return 3 * n


class Budget:
    """At most `limit` bytes of claims built at once across the split's workers: a part
    waits until its cost fits beside those being built, and one larger than the whole
    budget is built alone. So the split's claims are bounded however many workers run."""

    def __init__(self, limit: int):
        self.limit, self.used, self.cond = limit, 0, threading.Condition()

    @contextmanager
    def hold(self, cost: int):
        with self.cond:
            self.cond.wait_for(lambda: self.used == 0 or self.used + cost <= self.limit)
            self.used += cost
        try:
            yield
        finally:
            with self.cond:
                self.used -= cost
                self.cond.notify_all()


# The split's claims in memory at once (WIKIDATA_SPLIT_CLAIMS_MEMORY_GB, 4 GB by default:
# a pipeline worker's claims step on the dump's first chunk peaks at about 1.5 GB)
BUDGET = Budget(int(float(os.environ.get("WIKIDATA_SPLIT_CLAIMS_MEMORY_GB", 4)) * 1024**3))
