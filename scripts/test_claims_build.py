"""The claims the split builds (claims_build, reshape/src/claims_arrow.rs) against the
pipeline's claims call (process.normalise_claims_direct, polars-genson against
claims.avsc):

- on each chunk file given (the bench's, say): the same Arrow arrays, in the same
  batches, and the claims partition file byte for byte;
- on claims made to probe the edges: each either built as genson decodes it, or a misfit
  (left as JSON for genson to decode as before), never built otherwise;
- on the misfits genson's strict decode rejects or the reshape's stricter rules add: a
  misfit each.

    WIKIDATA_RELEASE=20261012 .venv/bin/python scripts/test_claims_build.py [chunk.parquet ...]
"""

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("WIKIDATA_RELEASE", "20261012")

import orjson  # noqa: E402
import polars as pl  # noqa: E402
import pyarrow as pa  # noqa: E402

from wikidata import claims_build, process  # noqa: E402


def genson_claims(path: Path) -> pa.Table:
    """The claims call's arrays, as normalise_claims_direct makes the call."""
    with tempfile.TemporaryDirectory() as d:
        stream = process.normalise_in_schema(
            "claims", path, column="claims", output_path=None, output_format="arrow",
            output_column="claims", wrap_root="claims", **process.CLAIMS_INFERENCE_OPTIONS,
            max_builders=100, typed=True, keep_columns=["id"],
            extract_invariants=process.LABEL_INVARIANTS, lookup_output_path=Path(d) / "l",
            prune=process.QUARANTINE_FIELDS, prune_output_path=Path(d) / "p",
            avro_schema=process.CLAIMS_AVRO, map_encoding="kv",
            schema=process.claims_schema, unnest=True,
        )
        return pa.RecordBatchReader.from_stream(stream).read_all()


def check_chunk(path: Path) -> None:
    part = pl.read_parquet(path, columns=["id", "claims"])
    built, why = claims_build.claims_arrays(part["id"], claims_build.text_of(part))
    assert built is not None, f"{path.name}: not built ({why})"
    want = genson_claims(path)
    assert built.schema.equals(want.schema), f"{path.name}: schema {built.schema} != {want.schema}"
    assert built.equals(want), f"{path.name}: arrays differ"
    assert [b.num_rows for b in built.to_batches()] == [b.num_rows for b in want.to_batches()], path.name
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        mine = d / "built.parquet"
        assert claims_build.write_claims(part["id"], claims_build.text_of(part), None, mine) is None
        # (and the part's rows taken from a larger column, as the split takes them)
        wider = pl.concat([part.head(3), part, part.tail(2)])
        rows = pa.array(range(3, 3 + part.height), pa.int64())
        mine2 = d / "built2.parquet"
        assert claims_build.write_claims(part["id"], claims_build.text_of(wider), rows, mine2) is None
        assert mine2.read_bytes() == mine.read_bytes(), f"{path.name}: taken rows differ"
        tmp = process.normalise_claims_direct(
            path, d / "claims" / path.name, d / "lookup.parquet", d / "q.parquet",
            process.n_ids(part),
        )
        same = mine.read_bytes() == tmp.read_bytes()
        assert same, f"{path.name}: the claims partition files differ"
    print(f"{path.name}: {part.height:,} entities, {built.num_rows:,} rows in "
          f"{len(built.to_batches())} batches, arrays and partition file identical")


SNAK = {"snaktype": "value", "property": "P31", "datavalue": {"entity-type": "item", "numeric-id": 5, "id": "Q5"},
        "datavalue_type": "wikibase-entityid", "datatype": "wikibase-item"}


def statement(**over) -> dict:
    return {"mainsnak": SNAK, "type": "statement", "id": "Q1$x", "rank": "normal", **over}


def claims(*statements, prop="P31") -> bytes:
    return orjson.dumps({prop: list(statements)})


def with_snak(**over) -> bytes:
    return claims(statement(mainsnak={**SNAK, **over}))


def q_snak(**over) -> dict:
    return {"snaktype": "value", "property": "P580", "hash": "abc", "datavalue": {"time": "+2001-00-00T00:00:00Z",
            "timezone": 0, "before": 0, "after": 0, "precision": 9, "calendarmodel": "Q1985727"},
            "datavalue_type": "time", "datatype": "time", **over}


# Claims genson decodes against claims.avsc, each built as genson decodes it (or a misfit)
EDGES = {
    "no claims": b"{}",
    "a statement": claims(statement()),
    "a string datavalue": with_snak(datavalue="abc", datavalue_type="string", datatype="string"),
    "escapes in strings": with_snak(datavalue='a"b\\c\ndé\U0001F600\x01', datavalue_type="string"),
    "a somevalue snak": claims(statement(mainsnak={"snaktype": "somevalue", "property": "P31", "datatype": "wikibase-item"})),
    "a null datavalue": with_snak(datavalue=None, datavalue_type=None),
    "a quantity": with_snak(datavalue={"amount": "+5", "unit": "1", "upperBound": "+6", "lowerBound": "+4"}, datavalue_type="quantity"),
    "a coordinate, floats": with_snak(datavalue={"latitude": 52.123456789012345, "longitude": -3.8209166666666667, "altitude": None,
                                                 "precision": 2.7777777777777776e-05, "globe": "Q2"}, datavalue_type="globecoordinate"),
    "a coordinate, integers": with_snak(datavalue={"latitude": 52, "longitude": -3, "altitude": None, "precision": 1, "globe": "Q2"},
                                        datavalue_type="globecoordinate"),
    "a coordinate, null precision": with_snak(datavalue={"latitude": 1.5, "longitude": 2.5, "altitude": None, "precision": None, "globe": "Q2"},
                                              datavalue_type="globecoordinate"),
    "a coordinate, big exponents": with_snak(datavalue={"latitude": 1e-300, "longitude": 1.7976931348623157e308, "altitude": None,
                                                        "precision": 5e-324, "globe": "Q2"}, datavalue_type="globecoordinate"),
    "a monolingual text": with_snak(datavalue={"text": "x", "language": "en"}, datavalue_type="monolingualtext"),
    "a time": with_snak(datavalue=q_snak()["datavalue"], datavalue_type="time"),
    "a large numeric id": with_snak(datavalue={"entity-type": "item", "numeric-id": 999999999999999999, "id": "Q1"}),
    "a numeric id past 18 digits": with_snak(datavalue={"entity-type": "item", "numeric-id": 9223372036854775807, "id": "Q1"}),
    "a negative numeric id": with_snak(datavalue={"entity-type": "item", "numeric-id": -7, "id": "Q1"}),
    "members out of order": orjson.dumps({"P31": [{"rank": "normal", "id": "Q1$x", "mainsnak": {
        "datatype": "wikibase-item", "datavalue_type": "wikibase-entityid", "datavalue": {"id": "Q5", "numeric-id": 5,
        "entity-type": "item"}, "property": "P31", "snaktype": "value"}, "type": "statement"}]}),
    "qualifiers": claims(statement(qualifiers={"P580": [q_snak()], "P582": [q_snak(property="P582"), q_snak(property="P582", snaktype="novalue")]},
                                   **{"qualifiers-order": ["P580", "P582"]})),
    "empty qualifiers": claims(statement(qualifiers={}, **{"qualifiers-order": []})),
    "an empty qualifier list": claims(statement(qualifiers={"P580": []})),
    "references": claims(statement(references=[{"hash": "h", "snaks": {"P248": [q_snak(property="P248")]}, "snaks-order": ["P248"]},
                                               {"hash": "h2", "snaks": {}, "snaks-order": []}])),
    "empty references": claims(statement(references=[])),
    "an empty statement list": orjson.dumps({"P31": [], "P279": [statement()]}),
    "several properties": orjson.dumps({f"P{i}": [statement()] for i in range(1, 40)}),
    "a statement of only a mainsnak": claims({"mainsnak": SNAK}),
}

# Claims the build must leave as JSON (genson's strict misfits, or the reshape's stricter ones)
MISFITS = {
    "an unknown statement field": claims(statement(extra=1)),
    "an unknown snak field": with_snak(extra="x"),
    "a mainsnak hash (not in claims.avsc)": with_snak(hash="abc"),
    "a label field (extract_invariants)": with_snak(datavalue={"entity-type": "item", "numeric-id": 5, "id": "Q5", "labels": {"en": "x"}}),
    "a pruned value field": with_snak(datavalue={"value": "x", "error": "y"}),
    "a pruned bare mainsnak": claims(statement(mainsnak="P31")),
    "a bare qualifier snak": claims(statement(qualifiers={"P580": ["P580"]})),
    "a null snaktype": with_snak(snaktype=None),
    "an absent property": claims(statement(mainsnak={k: v for k, v in SNAK.items() if k != "property"})),
    "a null mainsnak": claims(statement(mainsnak=None)),
    "a number at a string": with_snak(datatype=5),
    "a string at an integer": with_snak(datavalue={"entity-type": "item", "numeric-id": "5", "id": "Q5"}),
    "a fraction at an integer": with_snak(datavalue={"entity-type": "item", "numeric-id": 5.5, "id": "Q5"}),
    "an integer past 64 bits": with_snak(datavalue={"entity-type": "item", "numeric-id": 18446744073709551615, "id": "Q5"}),
    "a boolean": with_snak(datavalue=True),
    "an altitude": with_snak(datavalue={"latitude": 1.5, "longitude": 2.5, "altitude": 3.0, "precision": 1, "globe": "Q2"}),
    "a number datavalue": with_snak(datavalue=5),
    "an array at a record": with_snak(datavalue=[1]),
    "a scalar at an array": claims(statement(references="x")),
    "an object at a string": with_snak(datatype={"a": 1}),
    "an empty object at a record": with_snak(datavalue={}),
    "a repeated key": b'{"P31":[],"P31":[]}',
    "an escaped key": b'{"P\\u0033":[]}',
    "whitespace": b'{ "P31": [] }',
}


def misfit_reason(text: bytes) -> str | None:
    part = pl.DataFrame({"id": ["Q1"], "claims": [text.decode()]})
    built, why = claims_build.claims_arrays(part["id"], claims_build.text_of(part))
    return why


def check_edges() -> None:
    with tempfile.TemporaryDirectory() as d:
        for name, text in EDGES.items():
            path = Path(d) / "edge.parquet"
            part = pl.DataFrame({"id": ["Q1", "Q2"], "claims": [text.decode(), claims(statement()).decode()]})
            part.write_parquet(path)
            built, why = claims_build.claims_arrays(part["id"], claims_build.text_of(part))
            want = genson_claims(path)
            if built is None:
                print(f"edge {name!r}: a misfit ({why}), left to genson")
                continue
            assert built.equals(want), f"edge {name!r}: built, but not as genson decodes it"
            print(f"edge {name!r}: built as genson decodes it")
    for name, text in MISFITS.items():
        why = misfit_reason(text)
        assert why is not None, f"misfit {name!r} was built"
        print(f"misfit {name!r}: left as JSON ({why})")


if __name__ == "__main__":
    check_edges()
    for f in sys.argv[1:]:
        check_chunk(Path(f))
