"""claims.avsc against the target schema (claims_schema) the published claims are in:

- every field of claims_schema is in claims.avsc, so a claim holding it is built by the
  split (claims_build) rather than sending its part to genson's inference; but for the
  promoted integer fields beside a float one (`latitude__integer`, `longitude__integer`),
  which are listed: genson's strict decode gives an integer to `__number` where its record
  has no integer field, as the published claims hold it (see avro_schemas.complete);
- each field claims.avsc has is of its target type (claims_build._plan raises otherwise);
- claims.avsc is its own completion: `scripts/avro_schemas.py --complete` leaves it as it
  is;
- each shape the completion added (MISFITS: a reference snak's globe-coordinate, as on a
  real dump; its quantity bounds; a snak hash) was a misfit of the schema before it, and
  is now built: the same Arrow arrays (schema and values) as genson's inference gives the
  part, which is what a misfit part was published as (the fallback), as genson's strict
  decode against claims.avsc gives it, and as the split builds it. Each part has a
  release's context beside the shape: a reference's time (an integer precision) and a
  mainsnak's float coordinate, as every real chunk has;
- where inference and claims.avsc can part (DIVERGES): inference's types are the part's
  own, so a part showing a place with fewer variants than a real chunk does decodes
  otherwise. An integer latitude where the part has no float one goes to `__integer`
  (claims.avsc: `__number`); a null precision where every snak there has a precision is a
  record of nulls (claims.avsc: null). A mainsnak's has done so on master since
  claims.avsc was introduced (a chunk that fits it); a reference's now does the same.

    WIKIDATA_RELEASE=20261012 .venv/bin/python scripts/test_claims_schema.py
"""

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("WIKIDATA_RELEASE", "20261012")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json  # noqa: E402

import polars as pl  # noqa: E402
import pyarrow as pa  # noqa: E402

from avro_schemas import complete  # noqa: E402
from test_claims_build import claims, genson_claims, q_snak, statement, with_snak  # noqa: E402
from wikidata import claims_build, process  # noqa: E402
from wikidata.claims_schema import CLAIMS_AVRO, CLAIMS_AVRO_PATH  # noqa: E402

# The target fields claims.avsc leaves out, each a promoted integer beside its float field
SNAKS = (".claims{}[].mainsnak", ".claims{}[].qualifiers{}[]", ".claims{}[].references[].snaks{}[]")
LEFT_OUT = {f"{snak}.datavalue.{c}.{c}__integer" for snak in SNAKS for c in ("latitude", "longitude")}


# The fields the completion added to claims.avsc (the bench's 11 chunks never showed them;
# claims_schema has them): without them, the schema before it
REF = ".claims{}[].references[].snaks{}[]"
ADDED = [".claims{}[].mainsnak.hash", f"{REF}.hash", *(f"{REF}.datavalue.{f}" for f in (
    "precision.precision__number", "upperBound", "lowerBound", "latitude", "longitude", "altitude", "globe"))]


def coord(lat, lon, precision=2.777777777777778e-08, **over) -> dict:
    """A globe-coordinate snak, as a reference holds one on a real dump (Q618385's P625)."""
    return {"snaktype": "value", "property": "P625", "datavalue": {"latitude": lat, "longitude": lon, "altitude": None,
            "precision": precision, "globe": "http://www.wikidata.org/entity/Q2"}, "datavalue_type": "globecoordinate",
            "datatype": "globe-coordinate", **over}


def reference(*snaks) -> dict:
    return {"hash": "9628149cac907e79e8d46f7fb40b0d6187e7360e", "snaks": {s["property"]: [s] for s in snaks},
            "snaks-order": [s["property"] for s in snaks]}


QUANTITY = {"snaktype": "value", "property": "P1082", "datavalue": {"amount": "+4217", "unit": "1", "upperBound": "+4218",
            "lowerBound": "+4216"}, "datavalue_type": "quantity", "datatype": "quantity"}
# A release's context: a reference of a stated-in item (no precision) and its time
# retrieved (an integer precision), and a mainsnak's float coordinate, as every real chunk has
STATED_IN = {"snaktype": "value", "property": "P248", "datavalue": {"entity-type": "item", "numeric-id": 36578,
             "id": "Q36578"}, "datavalue_type": "wikibase-entityid", "datatype": "wikibase-item"}
# (a reference's snaks have no hash on a real dump; q_snak, a qualifier's, has one)
RETRIEVED = {k: v for k, v in q_snak(property="P813").items() if k != "hash"}
CONTEXT = [claims(statement(references=[reference(STATED_IN, RETRIEVED)])),
           claims(statement(mainsnak=coord(52.5, 13.25)), prop="P625")]

# Each a misfit of the schema before the completion, built now as the fallback publishes it
MISFITS = {
    "a reference's globe-coordinate (a real dump's misfit)": claims(statement(references=[reference(coord(42.503041, 74.419168))])),
    "a reference's globe-coordinate beside its time": claims(statement(references=[reference(RETRIEVED, coord(47.049845, 8.30308))])),
    "a reference's globe-coordinate, a null precision": claims(statement(references=[reference(coord(1.5, 2.5, None))])),
    "a reference's globe-coordinate, an integer precision": claims(statement(references=[reference(coord(1.5, 2.5, 1))])),
    "a reference's globe-coordinate, an integer latitude beside a float one":
        claims(statement(references=[reference(coord(47, 8.5))]), statement(references=[reference(coord(1.5, 2.5))])),
    "a reference's quantity with bounds": claims(statement(references=[reference(QUANTITY)])),
    "a reference snak's hash": claims(statement(references=[reference(q_snak(property="P813"))])),
    "a mainsnak's hash": with_snak(hash="abc"),
}
# Where the fallback and claims.avsc part: a part showing a place with fewer variants than
# a real chunk does, inference's types being the part's own (each case alone in its part,
# with no context). Each holds for a mainsnak on master as for a reference now
DIVERGES = {
    "a reference's integer latitude, none a float": claims(statement(references=[reference(coord(47, 8.5))])),
    "a mainsnak's integer latitude, none a float (as on master)": claims(statement(mainsnak=coord(47, 8.5)), prop="P625"),
    "a reference's null precision, every reference snak with one":
        claims(statement(references=[reference(coord(1.5, 2.5, None))]), statement(references=[reference(coord(1.5, 2.5))])),
    "a mainsnak's null precision, every mainsnak with one (as on master)":
        claims(statement(mainsnak=coord(1.5, 2.5, None)), statement(mainsnak=coord(1.5, 2.5)), prop="P625"),
}


def without(avro, path: str, at: str = ""):
    """An Avro type with the field at `path` (a phantoms-style path) removed."""
    if isinstance(avro, list):
        return [b if b == "null" else without(b, path, at) for b in avro]
    if isinstance(avro, str):
        return avro
    if avro["type"] == "record":
        fields = [f for f in avro["fields"] if f"{at}.{f['name']}" != path]
        return {**avro, "fields": [{**f, "type": without(f["type"], path, f"{at}.{f['name']}")} for f in fields]}
    if avro["type"] == "map":
        return {**avro, "values": without(avro["values"], path, f"{at}{{}}")}
    if avro["type"] == "array":
        return {**avro, "items": without(avro["items"], path, f"{at}[]")}
    return avro


def built_with(avro: str | None, part: pl.DataFrame):
    """The split's build of a part with claims.avsc as `avro`."""
    saved = claims_build.CLAIMS_AVRO, claims_build.PLAN, claims_build.SCHEMA
    try:
        claims_build.CLAIMS_AVRO = avro
        claims_build.PLAN, claims_build.SCHEMA = claims_build._compile()
        return claims_build.claims_arrays(part["id"], claims_build.text_of(part))
    finally:
        claims_build.CLAIMS_AVRO, claims_build.PLAN, claims_build.SCHEMA = saved


def inferred(path: Path) -> pa.Table:
    """The claims call's arrays without claims.avsc: genson's inference, the fallback."""
    saved, process.CLAIMS_AVRO = process.CLAIMS_AVRO, None
    try:
        return genson_claims(path)
    finally:
        process.CLAIMS_AVRO = saved


def check_fallback() -> None:
    root = json.loads(CLAIMS_AVRO)
    (field,) = [f for f in root["fields"] if f["name"] == "claims"]
    before = field["type"]
    for path in ADDED:
        before = without(before, path, ".claims")
    assert before != field["type"]
    old = json.dumps({**root, "fields": [{**field, "type": before}]})
    with tempfile.TemporaryDirectory() as d:
        for cases, same in ((MISFITS, True), (DIVERGES, False)):
            for name, text in cases.items():
                path = Path(d) / "part.parquet"
                part = pl.DataFrame({"id": [f"Q{i}" for i in range(1, 2 + len(CONTEXT))],
                                     "claims": [text.decode(), *(c.decode() for c in CONTEXT)]})
                if not same:
                    part = part.head(1)
                part.write_parquet(path)
                if same or "reference" in name:  # (the shape's row alone)
                    assert built_with(old, part.head(1))[1] == "a field claims.avsc lacks", f"{name!r}: not a misfit before"
                    assert built_with(old, part.tail(-1))[0] is not None, "the context is a misfit before"
                built, why = built_with(CLAIMS_AVRO, part)
                assert built is not None, f"{name!r}: not built ({why})"
                fallback, strict = inferred(path), genson_claims(path)
                assert strict.equals(built), f"{name!r}: built, but not as genson decodes it against claims.avsc"
                if same:
                    assert built.schema.equals(fallback.schema), f"{name!r}: not the fallback's schema"
                    assert built.equals(fallback), f"{name!r}: not the fallback's values"
                    print(f"misfit {name!r}: built, its arrays the fallback's (schema and values)")
                else:
                    assert built.schema.equals(fallback.schema) and not built.equals(fallback), f"{name!r}: no divergence"
                    if "as on master" in name:  # (fits the schema before, decoded the same)
                        assert built_with(old, part)[0].equals(built), f"{name!r}: not as before"
                    print(f"divergence {name!r}: inference and claims.avsc part, as known")


def phantoms(plan: tuple, path: str = "") -> set[str]:
    """The paths of the target fields a decode plan has no Avro type for."""
    kind = plan[0]
    if kind == "phantom":
        return {path}
    if kind == "record":
        return {p for name, child in plan[2] for p in phantoms(child, f"{path}.{name}")}
    if kind == "map":
        return phantoms(plan[2], f"{path}{{}}")
    if kind == "array":
        return phantoms(plan[2], f"{path}[]")
    return set()


def main() -> None:
    assert CLAIMS_AVRO is not None and claims_build.PLAN is not None, "no claims.avsc, or the build is off"
    missing = phantoms(claims_build.PLAN, ".claims")
    assert missing == LEFT_OUT, (
        f"target fields claims.avsc lacks: {sorted(missing - LEFT_OUT)}; "
        f"listed but present: {sorted(LEFT_OUT - missing)} (rerun scripts/avro_schemas.py --complete)"
    )
    print(f"claims.avsc: every field of claims_schema, but the {len(LEFT_OUT)} promoted integers listed")
    assert complete(CLAIMS_AVRO) + "\n" == CLAIMS_AVRO, f"{CLAIMS_AVRO_PATH} is not complete: run avro_schemas.py --complete"
    print("claims.avsc: its own completion")
    check_fallback()


if __name__ == "__main__":
    main()
