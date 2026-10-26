#!/usr/bin/env python3
"""The release schema's algebra and log (schema_evolution.py, release_schema.py), on small
schemas, in a few seconds (scripts/test_schema_evolution.py runs the pipeline end to end):

- **order:** a version's fields are the stored ones in their order, then the added ones
  sorted by name, at every depth, whichever order the changes came in;
- **diff:** what is additive (a new field at any depth, of any dtype; a field the release
  added as null given a type) and what is not (another type, a scalar become a record or a
  list, a value in a field the stored schema has as null, a map become a record), each
  conflict naming its path and both types;
- **log:** proposals merged under the lock in any order give one schema; versions only
  grow; strict mode records proposals as pending and accept merges them as one version,
  whatever their order; a frozen set takes no new field; a log begun from another stored
  schema halts;
- **check:** a file's schema must be exactly its tagged version's (an untagged file is
  version 0): a file of another schema, tagged otherwise than its schema, or of a version
  after the limit fails;
- **conform:** an earlier version's Arrow rows read as a later one's are the later
  version's rows (fields inserted as nulls at any depth, Null columns typed, sliced and
  chunked arrays, list item names), and anything else (a field the target lacks, another
  type) raises.

Usage: python scripts/test_schema_algebra.py
"""

import itertools
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("WIKIDATA_RELEASE", "20261012")

import polars as pl  # noqa: E402
import pyarrow as pa  # noqa: E402

from wikidata import release_schema as rsm  # noqa: E402
from wikidata import schema_evolution as se  # noqa: E402

FAILED = []


def check(ok: bool, what: str) -> None:
    print(f"{'ok  ' if ok else 'FAIL'} {what}", flush=True)
    if not ok:
        FAILED.append(what)


BASE = pl.Schema({"claims": pl.List(pl.Struct({
    "key": pl.String,
    "value": pl.List(pl.Struct({"m": pl.Struct({"b": pl.String, "a": pl.Int64}), "rank": pl.String, "alt": pl.Null})),
}))})
ST = ("claims", "[]", "value", "[]")


def with_fields(**extra) -> pl.Schema:
    """BASE with fields added to its statement (`m.x` for a field of `m`)."""
    dtype = pl.Struct(BASE)
    for name, t in extra.items():
        *parent, leaf = name.split(".")
        dtype = se._with(dtype, (*ST, *parent), lambda n, leaf=leaf, t=t: pl.Struct(
            {**{f.name: f.dtype for f in n.fields}, leaf: t}))
    return pl.Schema(dtype)


def names(schema: pl.Schema, *path) -> list[str]:
    return [f.name for f in se.at(pl.Struct(schema), (*ST, *path)).fields]


def log_tests(tmp: Path) -> None:
    proposals = [with_fields(zeta=pl.String, **{"m.c": pl.Float64}), with_fields(alpha=pl.Int64),
                 with_fields(zeta=pl.String, mid=pl.Struct({"y": pl.Null, "x": pl.String})),
                 with_fields(mid=pl.Struct({"y": pl.Int64}))]
    finals = set()
    for k, order in enumerate(itertools.permutations(range(len(proposals)))):
        rs = rsm.ReleaseSchema("t", BASE, None, lambda s: None, directory=tmp / f"auto-{k}")
        versions = [rs.propose(proposals[i], f"chunk_{i}") for i in order]
        finals.add(str(rs.schema(rs.latest())))
        ok = versions == sorted(versions) and rs.latest() <= len(proposals)
        if not ok:
            check(False, f"versions grow ({versions})")
    final = rsm.ReleaseSchema("t", BASE, None, lambda s: None, directory=tmp / "auto-0")
    s = final.schema(final.latest())
    check(len(finals) == 1, f"auto: the proposals in all {k + 1} orders give one schema, versions only growing")
    check(names(s) == ["m", "rank", "alt", "alpha", "mid", "zeta"] and names(s, "m") == ["b", "a", "c"]
          and names(s, "mid") == ["x", "y"] and se.at(pl.Struct(s), (*ST, "mid", "y")) == pl.Int64,
          "order: stored fields in their order, then the added sorted by name, at every depth; a null field typed")

    # strict: pending, accept merges in any order
    os.environ["WIKIDATA_SCHEMA_EVOLUTION"] = "strict"
    rsm.MODE = "strict"
    accepted = set()
    for k, order in enumerate(itertools.permutations(range(len(proposals)))):
        rs = rsm.ReleaseSchema("t", BASE, None, lambda s: None, directory=tmp / f"strict-{k}")
        halted = 0
        for i in order:
            try:
                rs.propose(proposals[i], f"chunk_{i}")
            except rsm.SchemaHalt as e:
                halted += "accept-schema" in str(e)
        if rs.latest() != 0 or halted != len(proposals) or not rs.log()["pending"]:
            check(False, f"strict: halted {halted}, versions {rs.latest()}")
        v = rs.accept()
        accepted.add(str(rs.schema(v)))
    check(accepted == finals and v == 1, "strict: each proposal halts and is pending; accept makes them one version, "
          "the auto mode's schema, in every order")
    rsm.MODE = "auto"
    os.environ.pop("WIKIDATA_SCHEMA_EVOLUTION")

    # conflicts
    rs = rsm.ReleaseSchema("t", BASE, None, lambda s: None, directory=tmp / "conflicts")
    rs.propose(with_fields(zeta=pl.String), "chunk_0")
    cases = {
        "another type": (with_fields(zeta=pl.Int64), "claims[].value[].zeta: String in the release schema, Int64"),
        "a scalar become a record": (with_fields(**{"rank": pl.Struct({"rank__string": pl.String})}), "a scalar became a record"),
        "a scalar become a list": (with_fields(**{"m.b": pl.List(pl.String)}), "a scalar became a list"),
        "a value in a stored null field": (with_fields(alt=pl.Float64), "claims[].value[].alt: Null in the release schema"),
        "a map become a record": (pl.Schema({"claims": pl.Struct({"P31": pl.List(pl.Struct({"rank": pl.Int64}))})}),
                                  "claims[].value[].rank: String in the release schema, Int64"),
    }
    for what, (inferred, want) in cases.items():
        try:
            rs.propose(inferred, "chunk_9")
            got = "no halt"
        except rsm.SchemaHalt as e:
            got = str(e)
        check(want in got and "chunk_9" in got and rs.latest() == 1, f"conflict, {what}: halts, naming the path, both "
              f"types and the chunk, nothing added ({got.splitlines()[1].strip() if '\n' in got else got})")

    # frozen; another stored schema
    rs.frozen("main")
    try:
        rs.propose(with_fields(late=pl.String), "chunk_8", "main")
        frozen = False
    except rsm.SchemaHalt as e:
        frozen = "frozen" in str(e)
    check(frozen and rs.latest() == 1 and rs.frozen("main") == 1, "a field found once the set is frozen halts, "
          "nothing added; the frozen version stays")
    other = rsm.ReleaseSchema("t", with_fields(z=pl.String), None, lambda s: None, directory=tmp / "conflicts")
    try:
        other.latest()
        halted = False
    except rsm.SchemaHalt:
        halted = True
    check(halted, "a log begun from another stored schema halts")


def arrow_tests() -> None:
    v0 = pa.schema([("id", pa.large_string()), ("m", pa.struct([("b", pa.large_string()), ("a", pa.int64())])),
                    ("refs", pa.large_list(pa.struct([("s", pa.large_string()), ("n", pa.null())])))])
    v1 = pa.schema([("id", pa.large_string()),
                    ("m", pa.struct([("b", pa.large_string()), ("a", pa.int64()), ("c", pa.float64())])),
                    ("refs", pa.large_list(pa.struct([("s", pa.large_string()), ("n", pa.int64()),
                                                      ("t", pa.large_list(pa.large_string()))]))),
                    ("zeta", pa.struct([("x", pa.large_string())]))])
    rows = [{"id": f"Q{i}", "m": None if i % 5 == 0 else {"b": f"b{i}", "a": i},
             "refs": None if i % 7 == 0 else [{"s": f"s{i}{j}", "n": None} for j in range(i % 3)]} for i in range(40)]
    t = pa.Table.from_pylist(rows, schema=v0)
    t = pa.concat_tables([t.slice(3, 20), t.slice(25, 10)])  # (sliced, two chunks)
    want = pa.Table.from_pylist([{**r, "m": None if r["m"] is None else {**r["m"], "c": None},
                                  "refs": None if r["refs"] is None else [{**x, "t": None} for x in r["refs"]],
                                  "zeta": None} for r in t.to_pylist()], schema=v1)
    got = se.conform_table(t, v1)
    check(got.equals(want) and got.schema.equals(v1), "conform: fields inserted as nulls at any depth, a Null column "
          "typed, on sliced and chunked arrays: the later version's rows")
    named = pa.schema([pa.field(f.name, f.type) for f in v1])
    elem = pa.schema([*v1][:2] + [pa.field("refs", pa.large_list(pa.field("element", v1.field("refs").type.value_type))),
                                  v1.field("zeta")])
    got2 = se.conform_table(t, elem)
    check(str(got2.schema.field("refs").type) == str(elem.field("refs").type) and got2.equals(want),
          "conform: a list's item field takes the target's name")
    del named
    for what, target in (("a field the target lacks", pa.schema([("id", pa.large_string())])),
                         ("another type", pa.schema([("id", pa.int64()), *list(v1)[1:]]))):
        try:
            se.conform_table(t, target)
            ok = False
        except ValueError:
            ok = True
        check(ok, f"conform raises for {what}")


def check_tests(tmp: Path) -> None:
    from wikidata.release_schema import ReleaseSchema, tagged

    write = lambda s: pa.schema([("id", pa.large_string()), *[(n, pa.large_string()) for n in s.names()[1:]]])  # noqa: E731
    base = pl.Schema({"claims": pl.String, "x": pl.String})
    rs = ReleaseSchema("c", base, None, write, directory=tmp / "check")
    rs.propose(pl.Schema({"claims": pl.String, "x": pl.String, "y": pl.String}), "chunk_1")
    v0 = rs.file_schema(0)
    v1 = rs.file_schema(1)

    def fails(schema, upto=1) -> bool:
        try:
            rs.check(schema, "f", upto)
            return False
        except RuntimeError:
            return True

    t1 = tagged(v1.empty_table(), 1).schema
    t0 = v0.empty_table().schema
    check(rs.check(t0, "f", 1) == 0 and rs.check(t1, "f", 1) == 1, "check: a file of each version, tagged as it is")
    check(fails(v1.empty_table().schema), "check: a file of version 1's schema untagged (version 0) fails")
    check(fails(tagged(v0.empty_table(), 1).schema), "check: a file tagged 1 of version 0's schema fails")
    check(fails(t1, 0), "check: a file of a version after the limit (the frozen one) fails")
    check(fails(pa.schema([("id", pa.large_string()), ("x", pa.int64())])), "check: a file of another schema fails")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        log_tests(tmp)
        arrow_tests()
        check_tests(tmp)
        promoted = [{"release": "1", "changes": [{"op": "add", "path": [*ST, "zeta"], "dtype": "String"}]},
                    {"release": "2", "changes": [{"op": "add", "path": [*ST, "alpha"], "dtype": "String"}]}]
        from wikidata.claims_schema import promoted_schema

        s = promoted_schema(BASE, promoted)
        check(names(s) == ["m", "rank", "alt", "zeta", "alpha"], "promotion: each release's fields stored in turn, a "
              "later release's after an earlier's (" + json.dumps(names(s)) + ")")
    if FAILED:
        sys.exit(1)
    print("all passed")


if __name__ == "__main__":
    main()
