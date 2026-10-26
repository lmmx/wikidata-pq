"""The algebra of a stored schema that grows as a release runs (release_schema.py keeps
the release's state): no state here, only functions of schemas.

- **Specs.** A Polars dtype as JSON (`spec`, `dtype_of`), for the release's log.
- **Paths.** A field's place in a dtype: its struct fields' names from the root, `[]` for
  a list's items (`claims [] value [] mainsnak datavalue time`).
- **Changes.** A version of the release schema is the stored schema with every change
  of the versions up to it applied (`apply`): `add` (a field a struct lacked, at any
  depth, of any dtype) and `type` (a field added during the release as Null, every value
  seen null, given the dtype a later chunk shows). Nothing else: a field is never
  removed, renamed, reordered or given another type once it has one.
- **Order.** Each struct of a version has the stored schema's fields of that struct
  first, in their stored order, then the fields added to it, sorted by name
  (`canonical`), so a version's schema is a function of its fields alone: not of the
  order the chunks that showed them were processed or finished in, nor of restarts.
- **Diff.** What a chunk's inferred schema adds to a version (`diff`): its changes, and
  what is not additive, each with its path and both dtypes (a conflict).
- **Conform.** Arrow data of a version read as a later version (`conform_table`): each
  field the data lacks inserted as nulls, at any depth, and each Null column given its
  later dtype, the data's own buffers kept (no cast: pyarrow's cast of the claims'
  nested structs corrupts them, compact._source_batches). A field the target lacks, or
  any other dtype, raises.
- **Avro.** The Avro schema a version's claims are decoded against (`complete_avro`): the
  stored claims.avsc with each target field it lacks added, nullable at every level, as
  scripts/avro_schemas.py completes it (#172).
"""

from __future__ import annotations

import json

import polars as pl
import pyarrow as pa
import pyarrow.compute as pc

ITEMS = "[]"

_SCALARS = {
    "String": pl.String,
    "Int64": pl.Int64,
    "Float64": pl.Float64,
    "Boolean": pl.Boolean,
    "Null": pl.Null,
}


def spec(dtype: pl.DataType):
    """A dtype as JSON: a scalar's name, {"list": spec} or {"struct": [[name, spec], ...]}."""
    for name, t in _SCALARS.items():
        if dtype == t:
            return name
    if isinstance(dtype, pl.Struct):
        return {"struct": [[f.name, spec(f.dtype)] for f in dtype.fields]}
    if isinstance(dtype, pl.List):
        return {"list": spec(dtype.inner)}
    raise ValueError(f"a dtype a release schema does not hold: {dtype}")


def dtype_of(s) -> pl.DataType:
    if isinstance(s, str):
        return _SCALARS[s]
    if "list" in s:
        return pl.List(dtype_of(s["list"]))
    return pl.Struct({name: dtype_of(f) for name, f in s["struct"]})


def show(path) -> str:
    """A path as genson's diagnostics name one (`claims[].value[].mainsnak.x`)."""
    out = ""
    for step in path:
        out += step if step == ITEMS else (f".{step}" if out else step)
    return out


def at(dtype: pl.DataType | None, path) -> pl.DataType | None:
    """The dtype at `path`, or None if there is no field there."""
    for step in path:
        if dtype is None:
            return None
        if step == ITEMS:
            dtype = dtype.inner if isinstance(dtype, pl.List) else None
        elif isinstance(dtype, pl.Struct):
            dtype = next((f.dtype for f in dtype.fields if f.name == step), None)
        else:
            return None
    return dtype


def _with(dtype: pl.DataType, path, fn) -> pl.DataType:
    """`dtype` with the node at `path` replaced by fn(node)."""
    if not path:
        return fn(dtype)
    step, rest = path[0], path[1:]
    if step == ITEMS:
        if not isinstance(dtype, pl.List):
            raise ValueError(f"no list at {step}")
        return pl.List(_with(dtype.inner, rest, fn))
    if not isinstance(dtype, pl.Struct):
        raise ValueError(f"no struct at {step}")
    names = [f.name for f in dtype.fields]
    if step not in names:
        raise ValueError(f"no field {step}")
    return pl.Struct({f.name: _with(f.dtype, rest, fn) if f.name == step else f.dtype for f in dtype.fields})


def canonical(dtype: pl.DataType, base: pl.DataType | None) -> pl.DataType:
    """`dtype` in the release's order: each struct's fields that `base` (the stored
    schema's node at the same place, or None) has, in base's order, then the others
    sorted by name, recursively."""
    if isinstance(dtype, pl.List):
        return pl.List(canonical(dtype.inner, base.inner if isinstance(base, pl.List) else None))
    if not isinstance(dtype, pl.Struct):
        return dtype
    fields = {f.name: f.dtype for f in dtype.fields}
    stored = [f.name for f in base.fields] if isinstance(base, pl.Struct) else []
    order = [n for n in stored if n in fields] + sorted(n for n in fields if n not in stored)
    sub = {f.name: f.dtype for f in base.fields} if isinstance(base, pl.Struct) else {}
    return pl.Struct({n: canonical(fields[n], sub.get(n)) for n in order})


def apply(base: pl.Struct, changes: list[dict]) -> pl.Struct:
    """The stored schema (a struct of the root's fields) with `changes` applied, in the
    release's order (`canonical`)."""
    dtype = base
    for c in changes:
        path, new = tuple(c["path"]), dtype_of(c["dtype"])
        if c["op"] == "add":
            def add(node, name=path[-1], new=new):
                if not isinstance(node, pl.Struct) or name in [f.name for f in node.fields]:
                    raise ValueError(f"cannot add {show(path)}")
                return pl.Struct({**{f.name: f.dtype for f in node.fields}, name: new})
            dtype = _with(dtype, path[:-1], add)
        elif c["op"] == "type":
            def typed(node, new=new):
                if node != pl.Null:
                    raise ValueError(f"cannot type {show(path)}: it is {node}, not Null")
                return new
            dtype = _with(dtype, path, typed)
        else:
            raise ValueError(f"an unknown change {c['op']!r}")
    return canonical(dtype, base)


def diff(current: pl.DataType, inferred: pl.DataType, base: pl.DataType, path=()) -> tuple[list, list]:
    """What `inferred` (a chunk's schema) adds to `current` (a version of the release
    schema), and what of it is not additive. `base` is the stored schema: a field typed
    Null there keeps that type (its column published as null), so a value in it is a
    conflict; a field the release added as Null may be given a type, as no value of it
    has been written. Returns (changes, conflicts), each conflict a dict of its path and
    both dtypes."""
    changes: list[dict] = []
    conflicts: list[dict] = []

    def conflict(old, new, why):
        conflicts.append({"path": list(path), "old": _brief(old), "new": _brief(new), "why": why})

    if inferred == pl.Null:  # no value seen: fits any dtype
        return changes, conflicts
    if current == pl.Null:
        if at(base, path) is not None:
            conflict(current, inferred, "a field the stored schema has as null in every row now holds values")
        else:
            changes.append({"op": "type", "path": list(path), "dtype": spec(inferred)})
        return changes, conflicts
    if isinstance(current, pl.Struct) and isinstance(inferred, pl.Struct):
        have = {f.name: f.dtype for f in current.fields}
        for f in inferred.fields:
            if f.name in have:
                c, x = diff(have[f.name], f.dtype, base, (*path, f.name))
                changes += c
                conflicts += x
            else:
                changes.append({"op": "add", "path": [*path, f.name], "dtype": spec(f.dtype)})
        return changes, conflicts
    if isinstance(current, pl.List) and isinstance(inferred, pl.List):
        return diff(current.inner, inferred.inner, base, (*path, ITEMS))
    if _is_map(current) and isinstance(inferred, pl.Struct):
        # A map inferred as a record of its keys: genson unifies a map's values unless
        # one does not fit the others, so the change is in a value; each key's value
        # against the map's, to name it (and the run halts whatever they show)
        value = current.inner.fields[1].dtype
        for f in inferred.fields:
            _, x = diff(value, f.dtype, base, (*path, ITEMS, "value"))
            conflicts += [{**c, "why": f"{c['why']} (in the value of key {f.name!r})"} for c in x]
        if not conflicts:
            conflict(current, inferred, "a map became a record")
        return changes[:0], conflicts
    if current != inferred:
        kinds = lambda t: "a record" if isinstance(t, pl.Struct) else "a list" if isinstance(t, pl.List) else "a scalar"  # noqa: E731
        if kinds(current) != kinds(inferred):
            conflict(current, inferred, f"{kinds(current)} became {kinds(inferred)}")
        else:
            conflict(current, inferred, "another type: an existing column is never given another")
    return changes, conflicts


def _brief(dtype: pl.DataType, limit: int = 160) -> str:
    text = str(dtype)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def merge_changes(changes: list[dict]) -> list[dict]:
    """Changes in a canonical order (adds before types, then by path), each path once:
    two chunks showing the same field add it once. The same path with two dtypes is not
    merged (diff of the second against the first finds the conflict)."""
    seen, out = set(), []
    for c in sorted(changes, key=lambda c: (c["op"] != "add", c["path"], json.dumps(c["dtype"]))):
        k = (c["op"], tuple(c["path"]), json.dumps(c["dtype"], sort_keys=True))
        if k not in seen:
            seen.add(k)
            out.append(c)
    return out


# Arrow: data of a version read as a later one


def _same(a: pa.DataType, b: pa.DataType) -> bool:
    """The same type, its list item fields' names too (which `equals` does not compare,
    but a file's footer keeps)."""
    return a.equals(b) and str(a) == str(b)


def conform_array(arr: pa.Array, t: pa.DataType) -> pa.Array:
    """`arr` as type `t`: the same values, each field `t` has and `arr` lacks as nulls,
    a Null array as nulls of its type, the other buffers kept as they are (a list's item
    field taking `t`'s name). Raises for a field `t` lacks, or any other difference of
    type."""
    if _same(arr.type, t):
        return arr
    if pa.types.is_null(arr.type):
        return pa.nulls(len(arr), t)
    mask = arr.is_null() if arr.null_count else None
    if pa.types.is_struct(t) and pa.types.is_struct(arr.type):
        have = [arr.type.field(i).name for i in range(arr.type.num_fields)]
        if extra := set(have) - {f.name for f in t}:
            raise ValueError(f"fields the target lacks: {sorted(extra)}")
        children = [conform_array(arr.field(f.name), f.type) if f.name in have else pa.nulls(len(arr), f.type)
                    for f in t]
        return pa.StructArray.from_arrays(children, fields=list(t), mask=mask)
    lists = {pa.types.is_large_list: pa.LargeListArray, pa.types.is_list: pa.ListArray}
    for is_list, cls in lists.items():
        if is_list(t) and is_list(arr.type):
            offsets, values = arr.offsets, arr.values
            if arr.offset:  # (a slice: its offsets rebased to its values, which from_arrays needs with a mask)
                start, end = offsets[0], offsets[len(offsets) - 1]
                values = values.slice(start.as_py(), end.as_py() - start.as_py())
                offsets = pc.subtract(offsets, start)
            return cls.from_arrays(offsets, conform_array(values, t.value_type), type=t, mask=mask)
    raise ValueError(f"{arr.type} is not an earlier version of {t}")


def conform_table(table: pa.Table, schema: pa.Schema) -> pa.Table:
    """`table` (of an earlier version's schema) in `schema` (conform_array, a column at a
    time), its metadata dropped; checked to have exactly `schema` and its rows."""
    t = table.replace_schema_metadata()
    if t.schema.equals(schema) and all(_same(a.type, b.type) for a, b in zip(t.schema, schema)):
        return t
    names = t.column_names
    if extra := set(names) - set(schema.names):
        raise ValueError(f"columns the target lacks: {sorted(extra)}")
    columns = []
    for f in schema:
        if f.name in names:
            col = t.column(f.name)
            columns.append(pa.chunked_array([conform_array(c, f.type) for c in col.chunks], f.type))
        else:
            columns.append(pa.chunked_array([pa.nulls(len(c), f.type) for c in t.column(0).chunks] or [], f.type))
    out = pa.Table.from_arrays(columns, schema=schema)
    if out.num_rows != table.num_rows or not out.schema.equals(schema):
        raise ValueError("conformed rows differ in number or schema")
    return out


# Avro: the schema a version's claims are decoded against

_AVRO_SCALARS = {pl.String: "string", pl.Int64: "int", pl.Float64: "float"}
_INT_SUFFIXES = ("int", "integer", "long")
_FLOAT_SUFFIXES = ("float", "double", "number")


def _suffix(name: str) -> str:
    """A promoted field's type suffix (`latitude__number`: number), as genson reads it."""
    return name.split("__")[-1] if "__" in name else ""


def _is_map(dtype: pl.DataType) -> bool:
    entry = dtype.inner if isinstance(dtype, pl.List) else None
    return isinstance(entry, pl.Struct) and [f.name for f in entry.fields] == ["key", "value"]


def _new(dtype: pl.DataType, name: str, namespace: str):
    """The Avro type of a target field the inferred schema lacks: nullable at every level
    (a field genson never saw), its records named as genson names them."""
    if dtype == pl.Null:
        return ["null"]
    if dtype in _AVRO_SCALARS:
        return ["null", _AVRO_SCALARS[dtype]]
    if isinstance(dtype, pl.Struct):
        record = {"type": "record", "name": name, "namespace": namespace, "fields": []}
        return ["null", _completed(record, dtype)]
    if _is_map(dtype):
        values = _new(dtype.inner.fields[1].dtype, f"{name}_values", f"{namespace}.{name}_types")
        return ["null", {"type": "map", "values": values}]
    if isinstance(dtype, pl.List):
        return ["null", {"type": "array", "items": _new(dtype.inner, name, f"{namespace}.{name}_types")}]
    raise ValueError(f"a target field {name} of {dtype}, with no Avro type")


def _completed(record: dict, dtype: pl.Struct) -> dict:
    """An Avro record with the target struct's fields it lacks added (see `complete_avro`)."""
    have = {f["name"]: f for f in record["fields"]}
    target = {f.name: f.dtype for f in dtype.fields}
    namespace = f"{record['namespace']}.{record['name']}_types"
    for name, t in target.items():
        if name in have:
            have[name]["type"] = _complete(have[name]["type"], t, name, namespace)
            continue
        # Never a promoted integer field beside a float one: genson's strict decode gives an
        # integer to the float field (`__number`) only where its record has no integer
        # field, so adding one would move every integer the schema already took
        siblings = [n for n in [*have, *target] if n.split("__")[0] == name.split("__")[0]]
        if _suffix(name) in _INT_SUFFIXES and any(_suffix(n) in _FLOAT_SUFFIXES for n in siblings):
            continue
        record["fields"].append({"name": name, "type": _new(t, name, namespace)})
        have[name] = record["fields"][-1]
    return record


def _complete(avro, dtype: pl.DataType, name: str, namespace: str):
    """An inferred Avro type completed against its target dtype: each record's missing
    fields added, recursively. Its nullability and its types are kept as inferred, but
    a ["null"] the target has since typed (a field the release added as null), which
    takes the target's type, nullable."""
    if avro == ["null"] or avro == "null":
        return avro if dtype == pl.Null else _new(dtype, name, namespace)
    if isinstance(avro, list):
        return [b if b == "null" else _complete(b, dtype, name, namespace) for b in avro]
    if isinstance(avro, str):
        return avro
    kind = avro["type"]
    if kind == "record":
        return _completed(avro, dtype)
    if kind == "map":
        values = dtype.inner.fields[1].dtype
        avro["values"] = _complete(avro["values"], values, f"{name}_values", namespace)
    elif kind == "array":
        avro["items"] = _complete(avro["items"], dtype.inner, name, namespace)
    return avro


def complete_avro(avsc: str, target: pl.DataType, column: str = "claims") -> str:
    """`avsc` (the stored claims.avsc) with every field of `target` (the dtype of its
    `column`) it lacks added, nullable: a claim holding such a field fits it, and is
    decoded as genson's inference would give it (a field absent is null either way).
    Every field of the target is then in it, but for a promoted integer field beside a
    float one (an integer latitude goes to `latitude__number`, as it does today)."""
    root = json.loads(avsc)
    (field,) = [f for f in root["fields"] if f["name"] == column]
    field["type"] = _complete(field["type"], target, column, root["namespace"])
    return json.dumps(root, separators=(",", ":"))
