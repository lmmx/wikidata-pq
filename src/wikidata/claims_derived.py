"""The claims' derived leaves: each entity value's `entity-type` and `numeric-id`, left out
of the claims partition files (and so of the group files glued from them, on the build
branch) wherever they are exactly a function of their sibling `id`, and put back as
finalise reads them, before its sort and its checks (docs/journal/2026-10-26-derived-leaves.md).

In a release's claims every datavalue has an `id`, `entity-type` and `numeric-id` (a
main snak's, each qualifier's and each reference snak's: the three `SITES`). An entity
value's type and number are spelled out by its id:

- `entity-type`: `Q…` item, `P…` property, `L…` lexeme, `L…-F…` form, `L…-S…` sense,
  `E…` entity-schema (`derive`'s ENTITY_TYPES), and null for a null id (any other value);
- `numeric-id`: the digits of a `Q…`, `P…` or `L…` id, and null for any other (forms,
  senses and entity schemas have none) or a null id.

They were 6.3% of the claims' group-file bytes (5.0% `numeric-id`, 1.3% `entity-type`:
docs/journal/2026-10-26-group-file-bytes.md), and the group files are the bytes a
release uploads to its build branch at its uplink's rate.

- **Checked, not assumed.** A writer drops a leaf only where every value of it in the
  rows written is the function of its sibling `id` (`reduce`): each leaf's values as
  Parquet stores them (under every non-null parent, null or not) compared with those
  derived from the ids, exactly. A leaf with any other value is kept, in that file.
- **Recorded.** A file without some derived leaves names them (their Parquet paths) in
  its Arrow schema's metadata (KEY), so each file is read back as what it is: one with
  none dropped is untagged, the bytes written before. A key's files of either form, or
  of several, meet across a restart or a code upgrade: they are made one form where they
  are glued (push.core._harmonise, the rows put back where needed) and read each as its
  own form in finalise.
- **Put back exactly.** `restore` derives the leaves from each `id` read (null for a null
  one, so null under a null datavalue as a Parquet read gives) and puts them in their
  place in the full schema: the same arrays, value for value and type for type, as the
  full file read back. Finalise restores the rows it reads (sort_by_id: the in-memory
  sort's reads, a bucket's direct reads and spilled rows) before its sort and every
  check, which compare the written files with the restored rows as before; the rows'
  plan is the same (sort_by_id.footer_size counts a dropped leaf as its sibling `id`'s
  values), so are the published files.
- **Schemas, explicitly.** A file's schema must be exactly its full schema (its version's,
  release_schema) less the leaves it names, and the leaves must be derived leaves of
  that schema (`reduced_schema`, `check_form`): nothing else is let through.
"""

from __future__ import annotations

import json
import os

import pyarrow as pa
import pyarrow.compute as pc

# WIKIDATA_CLAIMS_DERIVED=keep writes every leaf, as before (a file of either form is read)
MODE = os.environ.get("WIKIDATA_CLAIMS_DERIVED") or "drop"
if MODE not in ("drop", "keep"):
    raise SystemExit(f"WIKIDATA_CLAIMS_DERIVED is drop or keep, not {MODE!r}")

# The leaves dropped and their file's Arrow schema metadata key
KEY = b"wikidata.derived_leaves"
LEAVES = ("entity-type", "numeric-id")
# A datavalue struct holding an `id` and the derived leaves
_SIBLING = "id"

# An entity id's prefix (a letter and its digits) and its entity type, and a lexeme's
# form (-F and its digits) or sense (-S)
ENTITY_TYPES = {"Q": "item", "P": "property", "L": "lexeme", "E": "entity-schema"}
LEXEME_PARTS = {"F": "form", "S": "sense"}
# The entity types with a numeric-id: the id's digits
NUMBERED = ("Q", "P", "L")
# Digits a numeric-id takes at most (an int64 holds any 18)
_DIGITS = 18
_LEXEME_PART = rf"^L[0-9]{{1,{_DIGITS}}}-(?P<part>[FS])[0-9]+$"
_LETTERS, _TYPES = pa.array(list(ENTITY_TYPES), pa.large_string()), pa.array(list(ENTITY_TYPES.values()), pa.large_string())
_PARTS, _PART_TYPES = pa.array(list(LEXEME_PARTS), pa.large_string()), pa.array(list(LEXEME_PARTS.values()), pa.large_string())
_NUMBERED = pa.array(list(NUMBERED), pa.large_string())
_NO_INDEX, _NO_NUMBER = pa.scalar(None, pa.int32()), pa.scalar(None, pa.int64())
_ZERO, _MAX_DIGITS = pa.scalar("0", pa.large_string()), pa.scalar(_DIGITS, pa.int64())


def _is_list(t: pa.DataType) -> bool:
    return pa.types.is_list(t) or pa.types.is_large_list(t)


def sites(schema: pa.Schema | pa.DataType, prefix: str = "") -> list[str]:
    """The Parquet paths (list items as `list.element`) of the structs holding an `id`
    and both derived leaves, in schema order."""
    out = []
    fields = list(schema) if isinstance(schema, pa.Schema) else [schema.field(i) for i in range(schema.num_fields)]
    for f in fields:
        path = f"{prefix}{f.name}"
        t = f.type
        while _is_list(t):
            path, t = f"{path}.list.element", t.value_type
        if pa.types.is_struct(t):
            names = {t.field(i).name for i in range(t.num_fields)}
            if {_SIBLING, *LEAVES} <= names:
                out.append(path)
            out += sites(t, f"{path}.")
    return out


def derivable(schema: pa.Schema) -> list[str]:
    """The Parquet paths of the derived leaves of a full schema."""
    return [f"{s}.{leaf}" for s in sites(schema) for leaf in LEAVES]


def dropped(schema: pa.Schema) -> frozenset[str]:
    """The leaves a file's Arrow schema (with its metadata) names as dropped."""
    tag = (schema.metadata or {}).get(KEY)
    return frozenset(json.loads(tag)) if tag else frozenset()


def tagged(table: pa.Table, leaves) -> pa.Table:
    """`table` with its metadata naming `leaves` as dropped (none: the key removed)."""
    meta = {k: v for k, v in (table.schema.metadata or {}).items() if k != KEY}
    if leaves:
        meta[KEY] = json.dumps(sorted(leaves)).encode()
    return table.replace_schema_metadata(meta or None)


# Derivation


def derive(ids: pa.Array) -> tuple[pa.Array, pa.Array]:
    """The `entity-type` (large_string) and `numeric-id` (int64) of each id (see the
    module's docstring), null for a null id or one of no known form: a letter of
    ENTITY_TYPES and 1 to 18 ASCII digits (`numeric-id` those digits, for NUMBERED), or a
    lexeme's form or sense (`L…-F…`, `L…-S…`, with no `numeric-id`). Without a regex for
    the ids of a letter and digits (nearly all): a regex is matched only on the rest."""
    if not pa.types.is_large_string(ids.type):
        ids = ids.cast(pa.large_string())
    first, rest = pc.utf8_slice_codeunits(ids, 0, 1), pc.utf8_slice_codeunits(ids, 1)
    plain = pc.and_(pc.ascii_is_decimal(rest), pc.less_equal(pc.binary_length(rest), _MAX_DIGITS))
    plain = pc.fill_null(plain, False)
    letter = pc.index_in(first, value_set=_LETTERS)
    entity_type = _TYPES.take(pc.if_else(plain, letter, _NO_INDEX))
    other = pc.and_(pc.invert(plain), pc.is_valid(ids))
    if other.true_count:
        part = pc.extract_regex(ids.filter(other), _LEXEME_PART).flatten()[0]
        entity_type = pc.replace_with_mask(entity_type, other, _PART_TYPES.take(pc.index_in(part, value_set=_PARTS)))
    numbered = pc.and_(plain, pc.fill_null(pc.is_in(first, value_set=_NUMBERED), False))
    numeric = pc.if_else(numbered, pc.cast(pc.if_else(numbered, rest, _ZERO), pa.int64()), _NO_NUMBER)
    return entity_type, numeric


# Walking the nested arrays


def _children(arr: pa.StructArray) -> list[pa.Array]:
    """A struct array's children, sliced to it (no parent validity merged)."""
    return [arr.field(i) for i in range(arr.type.num_fields)]


def _mask(arr: pa.Array) -> pa.Array | None:
    return pc.is_null(arr) if arr.null_count else None


def _list_parts(arr: pa.Array) -> tuple[pa.Array, pa.Array]:
    """A list array's offsets and values, the values only those its lists hold and the
    offsets from 0 into them (as from_arrays takes a sliced list with nulls)."""
    offsets, values = arr.offsets, arr.values
    start, end = offsets[0].as_py(), offsets[-1].as_py()
    if arr.offset == 0 and start == 0 and end == len(values):
        return offsets, values
    return pc.subtract(offsets, offsets[0]), values.slice(start, end - start)


def _rebuild(arr: pa.Array, target: pa.DataType, path: str, leaves: frozenset[str], derive_into: bool) -> pa.Array:
    """`arr` laid out as `target`, which differs from its type only in `leaves` (Parquet
    paths): with `derive_into`, each is put back (derived from its sibling `id`), else
    each is left out. Every other buffer is reused."""
    if _is_list(arr.type):
        offsets, values = _list_parts(arr)
        values = _rebuild(values, target.value_type, f"{path}.list.element", leaves, derive_into)
        cls = pa.LargeListArray if pa.types.is_large_list(arr.type) else pa.ListArray
        return cls.from_arrays(offsets, values, type=target, mask=_mask(arr))
    if not pa.types.is_struct(arr.type):
        if arr.type != target:
            raise ValueError(f"{path}: {arr.type}, not {target}")
        return arr
    have = {arr.type.field(i).name: c for i, c in enumerate(_children(arr))}
    children = []
    derived = None
    for i in range(target.num_fields):
        f = target.field(i)
        sub = f"{path}.{f.name}" if path else f.name
        if sub in leaves:
            if not derive_into:
                raise ValueError(f"{sub}: a leaf to drop in the target")
            if f.name in have:
                raise ValueError(f"{sub}: put back where it is already")
            if derived is None:
                derived = dict(zip(LEAVES, derive(have[_SIBLING])))
            if f.type != derived[f.name].type:
                raise ValueError(f"{sub}: {f.type}, not the derived {derived[f.name].type}")
            children.append(derived[f.name])
        elif f.name not in have:
            raise ValueError(f"{sub}: in the target, not in the rows")
        else:
            children.append(_rebuild(have[f.name], f.type, sub, leaves, derive_into))
    if extra := set(have) - {target.field(i).name for i in range(target.num_fields)}:
        if derive_into or any((f"{path}.{n}" if path else n) not in leaves for n in extra):
            raise ValueError(f"{path}: fields in the rows that are not in the target: {sorted(extra)}")
    return pa.StructArray.from_arrays(children, fields=list(target), mask=_mask(arr))


def _drop_type(t: pa.DataType, path: str, leaves: frozenset[str]) -> pa.DataType:
    if _is_list(t):
        vf = t.value_field
        inner = vf.with_type(_drop_type(vf.type, f"{path}.list.element", leaves))
        return pa.large_list(inner) if pa.types.is_large_list(t) else pa.list_(inner)
    if not pa.types.is_struct(t):
        return t
    fields = []
    for i in range(t.num_fields):
        f = t.field(i)
        sub = f"{path}.{f.name}" if path else f.name
        if sub not in leaves:
            fields.append(f.with_type(_drop_type(f.type, sub, leaves)))
    return pa.struct(fields)


def reduced_schema(full: pa.Schema, leaves) -> pa.Schema:
    """`full` (no metadata kept) less `leaves`, each a derived leaf of it."""
    leaves = frozenset(leaves)
    if extra := leaves - set(derivable(full)):
        raise ValueError(f"not derived leaves of the schema: {sorted(extra)}")
    return pa.schema([f.with_type(_drop_type(f.type, f.name, leaves)) for f in full if f.name not in leaves])


def check_form(schema: pa.Schema, full: pa.Schema, what: str) -> frozenset[str]:
    """A file's leaves dropped (from its metadata), its schema checked to be exactly
    `full` less them; returns them."""
    leaves = dropped(schema)
    try:
        want = reduced_schema(full.remove_metadata(), leaves)
    except ValueError as e:
        raise RuntimeError(f"[derived] {what}: {e}") from None
    if not schema.remove_metadata().equals(want):
        raise RuntimeError(f"[derived] {what}: its schema is not its full schema less the leaves it names ({sorted(leaves)})")
    return leaves


# Writing and reading


def _logical(arr: pa.Array, path: str, want: str) -> pa.Array | None:
    """The values Parquet stores for the leaf at `want` (a path below `path`): under every
    non-null parent, in order (struct validity merged down, list values flattened)."""
    if path == want:
        return arr
    if _is_list(arr.type):
        return _logical(arr.flatten(), f"{path}.list.element", want)
    if pa.types.is_struct(arr.type):
        for i, child in enumerate(arr.flatten()):
            name = arr.type.field(i).name
            sub = f"{path}.{name}" if path else name
            if want == sub or want.startswith(sub + "."):
                return _logical(child, sub, want)
    return None


def _raw(arr: pa.Array, path: str, want: str) -> pa.Array | None:
    """The array at `want` (a path below `path`) as it is laid out, sliced to its parents
    but with no parent validity merged and every list's values whole: no copy. Its values
    hold the ones Parquet stores (`_logical`) and maybe others, under null parents."""
    if path == want:
        return arr
    if _is_list(arr.type):
        return _raw(arr.values, f"{path}.list.element", want)
    if pa.types.is_struct(arr.type):
        for i in range(arr.type.num_fields):
            name = arr.type.field(i).name
            sub = f"{path}.{name}" if path else name
            if want == sub or want.startswith(sub + "."):
                return _raw(arr.field(i), sub, want)
    return None


def _matches(arr: pa.Array, name: str, site: str) -> set[str]:
    """The derived leaves of `site` whose values in `arr` (a chunk of column `name`) are
    the ones derived from their sibling ids. Compared first as the arrays are laid out,
    with no copy (equal so, the values stored are equal), and only where they differ as
    Parquet stores them (`_logical`), under non-null parents alone."""
    ok = set()
    ids = _raw(arr, name, f"{site}.{_SIBLING}")
    for leaf, want in zip(LEAVES, derive(ids)):
        have = _raw(arr, name, f"{site}.{leaf}")
        if have.type == want.type and have.equals(want):
            ok.add(f"{site}.{leaf}")
            continue
        ids_l, have_l = _logical(arr, name, f"{site}.{_SIBLING}"), _logical(arr, name, f"{site}.{leaf}")
        want_l = derive(ids_l)[LEAVES.index(leaf)]
        if have_l.equals(want_l.cast(have_l.type) if have_l.type != want_l.type else want_l):
            ok.add(f"{site}.{leaf}")
    return ok


def matching(table: pa.Table) -> frozenset[str]:
    """The derived leaves of `table` whose every value Parquet would store is the one its
    sibling `id` gives (`derive`), exactly (nulls included), in every chunk."""
    ok = set()
    for site in sites(table.schema):
        name, here = site.split(".", 1)[0], {f"{site}.{leaf}" for leaf in LEAVES}
        for chunk in table.column(name).chunks:
            here &= _matches(chunk, name, site)
        ok |= here
    return frozenset(ok)


def drop(table: pa.Table, leaves) -> pa.Table:
    """`table` without `leaves` (derived leaves it has), tagged with them."""
    leaves = frozenset(leaves)
    if not leaves:
        return table
    target = reduced_schema(table.schema, leaves)
    columns = []
    for f in target:
        col = table.column(f.name)
        columns.append(pa.chunked_array([_rebuild(c, f.type, f.name, leaves, False) for c in col.chunks], f.type))
    out = pa.Table.from_arrays(columns, schema=target.with_metadata(table.schema.metadata))
    return tagged(out, dropped(table.schema) | leaves)


def reduce(table: pa.Table) -> pa.Table:
    """`table` without the derived leaves its values allow (`matching`), tagged with them
    (as it is with WIKIDATA_CLAIMS_DERIVED=keep, or a schema with none: not a release's)."""
    if MODE == "keep" or not sites(table.schema):
        return table
    return drop(table, matching(table))


def _leaves(schema: pa.Schema | pa.DataType, prefix: str = "") -> frozenset[str]:
    """A schema's leaf paths (list items as `list.element`, whatever their name)."""
    out = set()
    fields = list(schema) if isinstance(schema, pa.Schema) else [schema.field(i) for i in range(schema.num_fields)]
    for f in fields:
        path, t = f"{prefix}{f.name}", f.type
        while _is_list(t):
            path, t = f"{path}.list.element", t.value_type
        out |= _leaves(t, f"{path}.") if pa.types.is_struct(t) else {path}
    return frozenset(out)


def restore(table: pa.Table, full: pa.Schema, keep=frozenset()) -> pa.Table:
    """`table`'s rows (of a file whose leaves dropped its schema's metadata names) with
    those leaves put back but `keep`, derived from their ids, in `full`'s layout (the
    full schema, its metadata not kept) less `keep`: the rows the file of all its leaves
    reads back as (with `keep`, still without those, and tagged with them)."""
    have, keep = dropped(table.schema), frozenset(keep)
    if not keep <= have:
        raise ValueError(f"leaves to keep dropped that the rows have: {sorted(keep - have)}")
    # (rows whose metadata was lost, or names other leaves than they lack, are refused, not
    # passed through: their leaves are exactly the full schema's less those named)
    if _leaves(table.schema) != _leaves(full) - have:
        raise ValueError(f"rows not of the full schema less the leaves they name ({sorted(have)})")
    if not (put := have - keep):
        return table
    target = reduced_schema(full.remove_metadata(), keep)
    columns = []
    for f in target:
        col = table.column(f.name)
        columns.append(pa.chunked_array([_rebuild(c, f.type, f.name, put, True) for c in col.chunks], f.type))
    meta = {k: v for k, v in (table.schema.metadata or {}).items() if k != KEY} or None
    return tagged(pa.Table.from_arrays(columns, schema=target.with_metadata(meta)), keep)
