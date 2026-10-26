"""A globe-coordinate's numbers in one place, whatever the path its claims took.

The published claims keep a promoted field for each type a value's JSON can have: a
datavalue's `latitude`, `longitude` and `precision` are each a record of a float field
(`__number`) and an integer field (`__integer`). Which one a whole number (`47`, not
`47.0`) went to depended on the path its part took, not on the value:

- **the split's build** (claims_build, reshape/src/claims_arrow.rs) and **genson's strict
  decode** against claims.avsc: a latitude or longitude to `__number` (claims.avsc has no
  `__integer` field there), a precision to `__integer` (it has both there, as a time's
  precision is an integer);
- **genson's inference** (the fallback, for a part that misfits claims.avsc): to
  `__integer` wherever the part's inferred record had an integer field, as its own values
  had it, else to `__number`. And a null precision was a record of nulls where every
  other precision at its place was a number, not a null record.

The rule (`canonical_coordinates`): a globe-coordinate's (`datavalue_type`
"globecoordinate") latitude, longitude and precision are always in `__number`, as a
Float64, and the `__integer` field beside each is null; and a null one is a null
record, as the build gives it. Applied to the claims as each path hands them over, before
they are laid out as statements (partitioning.claims.statements): the split's built
claims (claims_build.write_claims, write_built) and genson's (process.normalise_claims_direct),
at a statement's mainsnak, its qualifiers' snaks and its references' snaks. Nothing
else changes: a time's precision (an integer, 0 to 14) stays in `precision__integer`, and
every other value, field and type is as it was, the arrays not copied where nothing
moves.

An integer is moved only if a Float64 holds it exactly (|v| <= 2^53, Arrow's checked
cast); any other halts, as does a whole number with a float already beside it, which no
path writes.
"""

import pyarrow as pa
import pyarrow.compute as pc

GLOBE = "globecoordinate"
FIELDS = ("latitude", "longitude", "precision")


class NotExact(ValueError):
    """A globe-coordinate's integer that a Float64 does not hold exactly, or one with a
    float beside it: not moved, and the claims not written."""


def canonical_coordinates(table: pa.Table) -> pa.Table:
    """`table` (`id`, `claims`, as genson or the split's build gives them) with every
    globe-coordinate's latitude, longitude and precision in `__number` (see the module's
    docstring): the same table where none is elsewhere."""
    i = table.schema.get_field_index("claims")
    column = table.column(i)
    old = column.chunks  # (each access wraps them anew)
    chunks = [_claims(c) for c in old]
    if all(n is c for n, c in zip(chunks, old)):
        return table
    return table.set_column(i, table.schema.field(i), pa.chunked_array(chunks, column.type))


def _claims(claims: pa.Array) -> pa.Array:
    """A claims column: a list of {key, value: [statement]}."""
    return _list(claims, lambda groups: _struct(groups, {"value": lambda s: _list(s, _statements)}))


def _statements(statements: pa.Array) -> pa.Array:
    """Statements: the mainsnak, the qualifiers' snaks and the references' snaks."""
    by_property = lambda groups: _list(groups, lambda g: _struct(g, {"value": lambda s: _list(s, _snaks)}))  # noqa: E731
    return _struct(statements, {
        "mainsnak": _snaks,
        "qualifiers": by_property,
        "references": lambda refs: _list(refs, lambda r: _struct(r, {"snaks": by_property})),
    })


def _snaks(snaks: pa.Array) -> pa.Array:
    """Snaks: each globe-coordinate's latitude, longitude and precision in `__number`."""
    names = snaks.type.names
    if "datavalue_type" not in names or "datavalue" not in names:
        return snaks
    globe = None

    def number(xs: pa.Array, name: str) -> pa.Array:
        nonlocal globe
        ints, nums = xs.field(f"{name}__integer"), xs.field(f"{name}__number")
        # (what could move: an integer, or a record of nulls; nothing in most chunks)
        if ints.null_count == len(ints) and not pc.any(pc.and_(xs.is_valid(), nums.is_null())).as_py():
            return xs
        if globe is None:
            globe = pc.fill_null(pc.equal(snaks.field("datavalue_type"), pa.scalar(GLOBE, snaks.type["datavalue_type"].type)), False)
            globe = pc.and_(globe, pc.and_(snaks.is_valid(), snaks.field("datavalue").is_valid()))
        at = pc.and_(globe, xs.is_valid())
        move = pc.and_(at, ints.is_valid())
        empty = pc.and_(at, pc.and_(ints.is_null(), nums.is_null()))
        n_move, n_empty = pc.sum(move).as_py() or 0, pc.sum(empty).as_py() or 0
        if not n_move and not n_empty:
            return xs
        children = {f.name: xs.field(f.name) for f in xs.type}
        if n_move:
            if pc.any(pc.and_(move, nums.is_valid())).as_py():
                raise NotExact(f"a globe-coordinate's {name} with both an integer and a float")
            moved = pc.if_else(move, ints, pa.nulls(len(ints), ints.type))
            try:  # (checked: an integer past 2^53 in magnitude raises)
                as_float = pc.cast(moved, nums.type, safe=True)
            except pa.ArrowInvalid as e:
                raise NotExact(f"a globe-coordinate's {name} a Float64 does not hold exactly: {e}") from None
            children[f"{name}__number"] = pc.if_else(move, as_float, nums)
            children[f"{name}__integer"] = pc.if_else(move, pa.nulls(len(ints), ints.type), ints)
        null = pc.or_(xs.is_null(), empty) if n_empty else xs.is_null()
        return pa.StructArray.from_arrays(list(children.values()), fields=list(xs.type), mask=null)

    dv = snaks.field("datavalue")
    present = [x for x in FIELDS if x in dv.type.names]
    return _struct(snaks, {"datavalue": lambda d: _struct(d, {x: (lambda xs, x=x: number(xs, x)) for x in present})})


def _list(lists: pa.Array, f) -> pa.Array:
    """A list array with `f` applied to its values (the same array if `f` changes none)."""
    values = lists.values
    new = f(values)
    if new is values:
        return lists
    validity, offsets = lists.buffers()[:2]
    return pa.Array.from_buffers(lists.type, len(lists), [validity, offsets], lists.null_count, lists.offset,
                                 children=[new])


def _struct(records: pa.Array, fs: dict) -> pa.Array:
    """A struct array with each `fs[name]` applied to its field `name` (the same array if
    they change none)."""
    children = [records.field(i) for i in range(records.type.num_fields)]
    new = [fs[f.name](c) if f.name in fs else c for f, c in zip(records.type, children)]
    if all(n is c for n, c in zip(new, children)):
        return records
    if records.offset == 0:  # (its children as field() gives them: from the struct's start)
        return pa.Array.from_buffers(records.type, len(records), records.buffers()[:1], records.null_count, 0,
                                     children=new)
    return pa.StructArray.from_arrays(new, fields=list(records.type), mask=records.is_null())
