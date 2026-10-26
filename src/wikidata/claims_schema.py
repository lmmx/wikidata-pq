"""The claims' schemas: the target schema every chunk's claims are written in
(`claims_schema`), the Avro schema they are decoded against (`claims.avsc`), and the
options genson normalises them with. Kept apart from process.py so the split
(claims_build.py) reads the same definitions without the pipeline's imports.
"""

import json
from pathlib import Path

import polars as pl

from .config import RELEASE


def n_ids(fr: pl.DataFrame | pl.LazyFrame) -> int:
    """Count the unique IDs (we expect them *all* to be preserved)."""
    return fr.lazy().select(pl.col("id").n_unique()).collect().item()


def check_ids(total: int, fr: pl.DataFrame | pl.LazyFrame, *, table: str) -> None:
    """Halt if a table lost (or gained) entity IDs relative to its source file."""
    if (n := n_ids(fr)) != total:
        raise RuntimeError(f"ID loss in {table}: {total} source ids --> {n}")


KV_SCHEMA = pl.List(pl.Struct({"key": pl.String, "value": pl.String}))
DV_SCHEMA = pl.Struct(
    {
        "id": pl.String,
        "datavalue__string": pl.String,
        "precision": pl.Struct(
            {
                "precision__integer": pl.Int64,
                "precision__number": pl.Float64,
            }
        ),
        "text": pl.String,
        "language": pl.String,
        "amount": pl.String,
        "unit": pl.String,
        "upperBound": pl.String,
        "lowerBound": pl.String,
        "time": pl.String,
        "timezone": pl.Int64,
        "before": pl.Int64,
        "after": pl.Int64,
        "calendarmodel": pl.String,
        "latitude": pl.Struct(
            {
                "latitude__number": pl.Float64,
                "latitude__integer": pl.Int64,
            }
        ),
        "longitude": pl.Struct(
            {
                "longitude__number": pl.Float64,
                "longitude__integer": pl.Int64,
            }
        ),
        "altitude": pl.Null,
        "globe": pl.String,
    }
)


def _claims_schema(snak: pl.Struct) -> pl.Schema:
    groups = pl.List(pl.Struct({"key": pl.String, "value": pl.List(snak)}))
    if RELEASE:
        # An official dump's statements, with their id, type and qualifiers' order, and
        # references with their hash and snaks' order
        reference = pl.Struct(
            {"hash": pl.String, "snaks": groups, "snaks-order": pl.List(pl.String)}
        )
        statement = pl.Struct(
            {
                "mainsnak": snak,
                "type": pl.String,
                "id": pl.String,
                "rank": pl.String,
                "references": pl.List(reference),
                "qualifiers": groups,
                "qualifiers-order": pl.List(pl.String),
            }
        )
        claims = pl.List(pl.Struct({"key": pl.String, "value": pl.List(statement)}))
        return pl.Schema({"claims": claims})
    statement = pl.Struct(
        {
            "mainsnak": snak,
            "rank": pl.String,
            "references": pl.List(groups),
            "qualifiers": groups,
        }
    )
    claims = pl.List(pl.Struct({"key": pl.String, "value": pl.List(statement)}))
    return pl.Schema({"claims": claims})


SNAK_FIELDS = {"property": pl.String, "datavalue": DV_SCHEMA, "datatype": pl.String}
# A release from an official dump keeps what the philippesaade copy had dropped (see
# dump.py): each snak's type ("value", "somevalue" for an unknown value, "novalue"), its
# hash, its datavalue's type, and an entity value's entity-type and numeric-id
if RELEASE:
    DV_SCHEMA = pl.Struct(
        {**{f.name: f.dtype for f in DV_SCHEMA.fields}, "entity-type": pl.String, "numeric-id": pl.Int64}
    )
    SNAK_FIELDS = {
        "snaktype": pl.String,
        "property": pl.String,
        "hash": pl.String,
        "datavalue": DV_SCHEMA,
        "datavalue_type": pl.String,
        "datatype": pl.String,
    }
SCHEMAS_DIR = Path(__file__).parent / "schemas"
# The fields earlier releases added to the claims (release_schema.py), promoted after each
# (`just promote-schema <release>`): each release's changes applied in turn, so the
# fields one release added are stored fields, in their order, for the next
PROMOTED_PATH = SCHEMAS_DIR / "claims.promoted.json"


def stored_claims_schema() -> pl.Schema:
    """The claims schema as defined here, before any promoted release's fields."""
    return _claims_schema(pl.Struct(SNAK_FIELDS))


def promoted_schema(stored: pl.Schema, promoted: list[dict]) -> pl.Schema:
    """`stored` with each promoted release's changes applied in turn."""
    from .schema_evolution import apply

    dtype = pl.Struct(stored)
    for release in promoted:
        dtype = apply(dtype, release["changes"])
    return pl.Schema(dtype)


# The target schema every chunk's claims are written in, a release's version 0 (a release
# adds to it as it runs: release_schema.py)
claims_schema = stored_claims_schema()
if RELEASE and PROMOTED_PATH.exists():
    claims_schema = promoted_schema(claims_schema, json.loads(PROMOTED_PATH.read_text()))

# Snaks on deleted properties (P450, P4003) are pruned by normalise_from_parquet: a
# record holding one of these fields is removed, and a mainsnak takes its statement
# with it. A datatype lookup that failed leaves the datavalue as {value, error}; a snak
# collapsed to its bare property id is promoted to mainsnak__string (a mainsnak:
# force_scalar_promotion) or value__string (a qualifier or reference snak: wrap_scalars).
# The fields are left out of the inferred schema, so it is checked against claims_schema.
QUARANTINE_FIELDS = {"mainsnak__string", "value__string", "value", "error"}

# The Avro schema a release's claims are decoded against without inference, where a chunk's
# claims all fit it (polars-genson's normalise_from_parquet(avro_schema=...)); a chunk
# with any claim it does not hold (a field, a type, a null where it never had one) is
# inferred and checked as before. Inferred from the bench's chunks of a release by
# scripts/avro_schemas.py, which rewrites it
CLAIMS_AVRO_PATH = SCHEMAS_DIR / "claims.avsc"
CLAIMS_AVRO = CLAIMS_AVRO_PATH.read_text() if RELEASE and CLAIMS_AVRO_PATH.exists() else None

CLAIMS_INFERENCE_OPTIONS = {
    "ndjson": True,
    "map_threshold": 0,
    "unify_maps": True,
    "force_field_types": {"mainsnak": "record"},
    "force_scalar_promotion": {
        "mainsnak",
        "datavalue",
        "precision",
        "latitude",
        "longitude",
    },
    "no_unify": {"qualifiers"},
}


# Label maps repeated in every claim that mentions an entity, property or unit, keyed by
# their sibling field. They are moved out to a per-chunk lookup table (Table.CLAIMS_LABELS).
LABEL_INVARIANTS = {
    "labels": "id",
    "property-labels": "property",
    "unit-labels": "unit",
}

