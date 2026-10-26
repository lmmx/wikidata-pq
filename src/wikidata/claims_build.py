"""A part's claims partition file written by the split (dump.route_parts), so the chunk
file holds no claims column and the pipeline makes no claims call.

The pipeline's claims step (process.normalise_claims_direct) reads a chunk's claims JSON
back from its file and has polars-genson decode it against `claims.avsc`, then lays it out
as the claims partition (partitioning.claims.statements) and writes it (write_table). The
split has that JSON in memory as it writes the chunk. Here it is decoded straight into the
same Arrow arrays by the reshape (reshape/src/claims_arrow.rs: genson's strict decode,
stricter), then given the same globe-coordinate numbers (claims_numeric), laid out and
written by the same code, a batch at a time (see write_claims), so the split never holds
a part's claims whole. The file is the
pipeline's in its rows, their order, its schema and its row groups (each row group's
encoding is its own, not the whole file's); scripts/test_claims_build.py checks it on the
bench's chunks.

A part whose claims the reshape does not decode exactly as genson would (`claims_arrays`
gives None) keeps its claims as JSON, and the pipeline decodes them as before. So do all
parts when WIKIDATA_SPLIT_CLAIMS=json, or without claims.avsc.

The split first builds a chunk's claims from its dump lines in the reshape's own scan
(`reshape_built`, claims_arrow.rs RawBuild): no claims text is written or decoded again,
and each row's set is taken from the same scan, as scholarly.is_scholarly's regex would
route it. The arrays are those the text would decode to. Any line that path does not
mirror exactly sends the whole chunk down the text path above, from its start.
"""

import os
import shutil
import threading
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import polars as pl
import pq_glue
import pyarrow as pa
from typing import NamedTuple

from wikidata_reshape import claims_batches, entity_rows_built

from . import release_schema
from .claims_numeric import canonical_coordinates
from .claims_refs import with_refs
from .claims_schema import CLAIMS_AVRO, check_ids, claims_schema
from .partitioning.claims import statements
from .partitioning.core import ARROW_ROW_GROUP_ROWS, write_table
from .scholarly import SCHOLARLY_CLASSES

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


def _compile(avro: str | None = None, target: pl.Schema | None = None):
    """The plan of the claims column, and the Arrow schema of genson's output (id, claims):
    of claims.avsc and claims_schema (the release's version 0), or of `avro` and `target`
    (a later version's, release_schema.py)."""
    avro = CLAIMS_AVRO if avro is None else avro
    target = claims_schema if target is None else target
    if avro is None or SPLIT_CLAIMS == "json":
        return None, None
    import json

    root = json.loads(avro)
    (claims_field,) = [f for f in root["fields"] if f["name"] == "claims"]
    plan = _plan(target["claims"], claims_field["type"])
    schema = pl.DataFrame(schema={"id": pl.String, **target}).to_arrow(
        compat_level=pl.CompatLevel.oldest()
    ).schema
    return plan, schema


PLAN, SCHEMA = _compile()
# Each later version's plan and schema, compiled once each (current)
_VERSIONS: dict[int, tuple] = {}


def current() -> tuple[int, tuple | None, pa.Schema | None]:
    """The release schema's latest version (release_schema.py), with its plan and the
    Arrow schema of genson's output in it: the split follows the release schema as the
    pipeline's chunks add to it, so a part holding a field a chunk added is built, not
    left as JSON. Version 0's are PLAN and SCHEMA (read here, as tests replace them)."""
    version = release_schema.CLAIMS.latest() if PLAN is not None else 0
    if version == 0:
        return 0, PLAN, SCHEMA
    if version not in _VERSIONS:
        rs = release_schema.CLAIMS
        _VERSIONS[version] = _compile(rs.avro(version), rs.schema(version))
    return version, *_VERSIONS[version]


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
    ids: pl.Series, text: pa.Array, rows: pa.Array | None = None, at: tuple | None = None
) -> tuple[pa.Table | None, str | None]:
    """Claims as the claims call gives them (`id`, `claims`), in one batch, or None and why
    they do not all fit: `ids` theirs, `text` the claims column (large strings) they are
    at (`rows`, int64 positions in it, or all of it). Built in the release schema's
    latest version, or in `at` (a version, its plan and schema: current())."""
    _, plan, schema = at or current()
    if plan is None:
        return None, "not built (WIKIDATA_SPLIT_CLAIMS=json, or no claims.avsc)"
    if isinstance(text, pa.ChunkedArray):
        text = text.combine_chunks()
    if text.null_count:
        return None, "a null claims cell"
    _, offsets, data = text.buffers()
    # (the positions as exactly their bytes: an Arrow buffer of a slice holds all its array's)
    index = None if rows is None else np.ascontiguousarray(rows.to_numpy(zero_copy_only=False), np.int64).view(np.int8)
    n = len(text) if rows is None else len(rows)
    batches, why = claims_batches(plan, data, offsets, text.offset, n, max(1, n), index)
    if batches is None:
        return None, why
    claims = [_arrays(plan, schema.field("claims").type, iter(nodes)) for nodes in batches]
    ids = ids.to_arrow(compat_level=pl.CompatLevel.oldest())
    return pa.Table.from_arrays([ids, pa.chunked_array(claims, schema.field("claims").type)], schema=schema), None


# Where a chunk's claims built from its lines are, among its arrays (dump._reshape)
BUILT = "built claims"


class Built(NamedTuple):
    """A chunk's claims built from its lines: each row's set (True: the scholarly set's,
    as is_scholarly routes it) and each set's claims column, its rows in order, in the
    release schema's `version` (and its genson output `schema`)."""

    scholarly: pl.Series
    claims: dict[str, pa.ChunkedArray]
    version: int = 0
    schema: pa.Schema | None = None


def reshape_built(lines: list[bytes]) -> tuple[tuple, Built] | None:
    """A chunk's rows as `entity_rows` gives them (its claims column's cells empty), and
    its claims built from its lines; or None for the text path (not built, a line left
    to dump.entity_row, or a misfit: see claims_arrow.rs RawBuild)."""
    version, plan, schema = current()
    if plan is None:
        return None
    rows, built, _why = entity_rows_built(lines, plan, list(SCHOLARLY_CLASSES), BATCH_TEXT_BYTES)
    if rows is None:
        return None
    flags, sets = built
    t = schema.field("claims").type
    claims = {name: pa.chunked_array([_arrays(plan, t, iter(nodes)) for nodes in batches], t)
              for name, batches in zip(("main", "scholar"), sets)}
    scholarly = pl.Series("claims", np.frombuffer(flags, dtype=np.uint8).astype(bool))
    return rows, Built(scholarly, claims, version, schema)


def write_built(ids: pl.Series, claims: pa.ChunkedArray, dst: Path, version: int = 0,
                schema: pa.Schema | None = None) -> None:
    """Write a part's claims partition file to `dst` from its claims built from its lines
    (reshape_built), as write_claims writes it from their text: its `ids` checked, its
    statements laid out and written a row group at a time, with their refs sidecar, in
    the release schema's `version` (Built.version) and tagged with it."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_arrays([ids.to_arrow(compat_level=pl.CompatLevel.oldest()), claims], schema=schema or SCHEMA)
    out = with_refs(RowGroups(dst.with_suffix(".rowgroups")))
    try:
        out.add(release_schema.tagged(statements(canonical_coordinates(table)), version))
        check_ids(ids.n_unique(), pl.from_arrow(table.select(["id"])), table="claims")
        out.close(dst)
    except BaseException:
        shutil.rmtree(out.work, ignore_errors=True)
        raise


def text_of(part: pl.DataFrame) -> pa.Array:
    """A frame's claims column as large strings (copied out of Polars' string views)."""
    return part.get_column("claims").to_arrow(compat_level=pl.CompatLevel.oldest())


# The claims text each batch of a part is built from, about: a batch's arrays are about
# 1.45 times its text, and are let go once its statements are written (see write_claims)
BATCH_TEXT_BYTES = 16 * 1024**2


def _batches(text: pa.Array, rows: pa.Array | None) -> list[tuple[int, int]]:
    """A part's rows (all of `text`, or its `rows`) cut into runs of about BATCH_TEXT_BYTES
    of claims text: each (start, end), positions in the part."""
    offsets = np.frombuffer(text.buffers()[1], dtype=np.int64)[text.offset : text.offset + len(text) + 1]
    sizes = np.diff(offsets)
    if rows is not None:
        sizes = sizes[rows.to_numpy(zero_copy_only=False)]
    ends = np.searchsorted(np.cumsum(sizes), np.arange(1, len(sizes) + 1) * BATCH_TEXT_BYTES, side="right")
    cuts = sorted({0, *(int(e) for e in ends if 0 < e < len(sizes)), len(sizes)})
    return list(zip(cuts, cuts[1:])) or [(0, 0)]


class RowGroups:
    """A claims partition file written a row group at a time: statements added in order,
    each full row group of ARROW_ROW_GROUP_ROWS rows written (write_table) to a file of its
    own as soon as it is full and let go, and the files glued into one at the end
    (pq_glue.concat, which copies their encoded row groups). So no more than a row group
    and a batch are held at once, and the file has the row groups write_table gives the
    whole table, its rows and schema the same."""

    def __init__(self, work: Path):
        self.work, self.files, self.pending, self.held, self.schema = work, [], [], 0, None
        work.mkdir(parents=True, exist_ok=True)

    def add(self, table: pa.Table) -> None:
        self.schema = self.schema or table.schema
        for batch in table.to_batches():
            self.pending.append(batch)
            self.held += batch.num_rows
            while self.held >= ARROW_ROW_GROUP_ROWS:
                self._flush(ARROW_ROW_GROUP_ROWS)

    def _flush(self, rows: int) -> None:
        take, left = [], rows
        while left:
            batch = self.pending.pop(0)
            if batch.num_rows > left:
                self.pending.insert(0, batch.slice(left))
                batch = batch.slice(0, left)
            take.append(batch)
            left -= batch.num_rows
        self.held -= rows
        f = self.work / f"{len(self.files)}.parquet"
        write_table(pa.Table.from_batches(take, schema=self.schema), f)
        self.files.append(f)

    def close(self, dst: Path) -> None:
        """The file at `dst` (by way of a temporary file), and the row groups' files gone."""
        if self.held:
            self._flush(self.held)
        elif not self.files:  # no statements: a file with no rows, as write_table writes one
            self.files.append(self.work / "0.parquet")
            write_table(pa.Table.from_batches([], schema=self.schema), self.files[0])
        tmp = dst.with_suffix(".tmp")
        if len(self.files) == 1:
            self.files[0].replace(tmp)
        else:
            pq_glue.concat([str(f) for f in self.files], str(tmp))
        shutil.rmtree(self.work)
        tmp.replace(dst)


def write_claims(ids: pl.Series, text: pa.Array, rows: pa.Array | None, dst: Path) -> str | None:
    """Write a part's claims partition file to `dst` as process.normalise_claims_direct
    writes it, the same rows, order and schema (its ids checked first, as there): the
    part's `ids`, and its claims text (`text` at `rows`, see claims_arrays). Built and
    written a batch at a time (see _batches, RowGroups), each batch's arrays let go once its
    statements are written, so a part is never held whole. None once written, else why its
    claims stay as JSON (and nothing is left written). Every batch is built in the
    release schema's version when the part began (current), and the file tagged with it."""
    at = current()
    if at[1] is None:
        return "not built (WIKIDATA_SPLIT_CLAIMS=json, or no claims.avsc)"
    dst.parent.mkdir(parents=True, exist_ok=True)
    out = with_refs(RowGroups(dst.with_suffix(".rowgroups")))
    built = []
    try:
        for start, end in _batches(text, rows):
            batch_rows = None if rows is None else rows.slice(start, end - start)
            if rows is None:
                # positions in the part, as rows of the whole column
                batch_rows = pa.array(np.arange(start, end, dtype=np.int64))
            with BUDGET.hold(_cost(text, batch_rows)):
                claims, why = claims_arrays(ids.slice(start, end - start), text, batch_rows, at)
                if claims is None:
                    shutil.rmtree(out.work)
                    return why
                built.append(claims.column("id"))
                out.add(release_schema.tagged(statements(canonical_coordinates(claims)), at[0]))
                del claims
        check_ids(ids.n_unique(), pl.from_arrow(pa.table({"id": pa.chunked_array(
            [c for col in built for c in col.chunks], pa.large_string())})), table="claims")
        out.close(dst)
    except BaseException:
        shutil.rmtree(out.work, ignore_errors=True)
        raise
    return None


def _cost(text: pa.Array, rows: pa.Array | None) -> int:
    """A batch's claims' bytes in memory while they are built and written, estimated from
    their text: the arrays hold about 1.45 times the text, and a row group's statements
    are held until written (about as much again)."""
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


# The split's claims in memory at once, across its workers (WIKIDATA_SPLIT_CLAIMS_MEMORY_GB,
# 4 GB by default: each batch is costed at 3 times its BATCH_TEXT_BYTES)
BUDGET = Budget(int(float(os.environ.get("WIKIDATA_SPLIT_CLAIMS_MEMORY_GB", 4)) * 1024**3))
