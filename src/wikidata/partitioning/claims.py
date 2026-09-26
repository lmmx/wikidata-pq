"""Claims-specific transforms for language partitioning.

Claims are complex: the language for partitioning depends on the datatype.
Each datatype has different nested structures containing language information:

- wikibase-item/property: match property-labels lang to datavalue.labels lang
- quantity: match property-labels lang to unit-labels lang (when unit has labels)
- scalar types (string, external-id, time, etc.): use property-labels lang directly
- monolingualtext: use datavalue.language, match to property-labels

The label maps are not in the claims rows: processing moves them to a per-chunk lookup
table (field, ref, language, label), and the transforms join against it.
"""

import polars as pl

WIKIBASE_TYPES = ["wikibase-item", "wikibase-property"]

SCALAR_TYPES = [
    "external-id",
    "string",
    "time",
    "globe-coordinate",
    "commonsMedia",
    "math",
    "musical-notation",
    "geo-shape",
    "tabular-data",
    "url",
    "wikibase-lexeme",
    "wikibase-form",
    "wikibase-sense",
    "entity-schema",
]


def claims_base(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Common base transform: explode claims to individual rows, unnest mainsnak."""
    return (
        lf.explode("claims", empty_as_null=True)
        .select("id", pl.col("claims").struct.unnest())
        .drop("key")
        .explode("value", empty_as_null=True)
        .unnest("value")
        .unnest("mainsnak")
    )


def lookup_labels(lookup: pl.LazyFrame, field: str, ref: str, label: str) -> pl.LazyFrame:
    """One field's labels from the lookup table, as (`ref`, language, `label`)."""
    return lookup.filter(pl.col("field") == field).select(
        pl.col("ref").alias(ref), "language", pl.col("label").alias(label)
    )


def transform_wikibase(
    base: pl.LazyFrame, prop_labels: pl.LazyFrame, lookup: pl.LazyFrame
) -> pl.LazyFrame:
    """wikibase-item/property: match property-label lang to datavalue label lang.

    Inner joins keep only the languages in which both labels exist.
    """
    dv_labels = lookup_labels(lookup, "labels", "_dv_id", "datavalue_label")
    return (
        base.filter(pl.col("datatype").is_in(WIKIBASE_TYPES))
        .with_columns(pl.col("datavalue").struct.field("id").alias("_dv_id"))
        .join(prop_labels, on="property", how="inner")
        .join(dv_labels, on=["_dv_id", "language"], how="inner")
        .drop("_dv_id")
    )


def transform_quantity(
    base: pl.LazyFrame, prop_labels: pl.LazyFrame, lookup: pl.LazyFrame
) -> pl.LazyFrame:
    """quantity: match property-label lang to unit-labels lang when unit has labels.

    When unit="1" (dimensionless), there are no unit-labels, so we just use
    property-label language directly.
    """
    unit_labels = lookup_labels(lookup, "unit-labels", "_unit", "unit_label")
    labelled_units = unit_labels.select("_unit").unique()
    qty_base = base.filter(pl.col("datatype") == "quantity").with_columns(
        pl.col("datavalue").struct.field("unit").alias("_unit")
    )

    with_units = (
        qty_base.join(labelled_units, on="_unit", how="semi")
        .join(prop_labels, on="property", how="inner")
        .join(unit_labels, on=["_unit", "language"], how="inner")
    )

    # Without unit-labels: property-label language is sufficient
    without_units = qty_base.join(labelled_units, on="_unit", how="anti").join(
        prop_labels, on="property", how="left"
    )

    return pl.concat([with_units, without_units], how="diagonal").drop("_unit")


def transform_scalar(base: pl.LazyFrame, prop_labels: pl.LazyFrame) -> pl.LazyFrame:
    """Scalar types: no language in datavalue, property-label lang is partition key."""
    return base.filter(pl.col("datatype").is_in(SCALAR_TYPES)).join(
        prop_labels, on="property", how="left"
    )


def transform_monolingualtext(
    base: pl.LazyFrame, prop_labels: pl.LazyFrame
) -> pl.LazyFrame:
    """monolingualtext: datavalue.language IS the language to partition on.

    We still want the property_label in the matching language where available.
    """
    return (
        base.filter(pl.col("datatype") == "monolingualtext")
        .with_columns(pl.col("datavalue").struct.field("language").alias("language"))
        .join(prop_labels, on=["property", "language"], how="inner")
    )


def prepare_claims(lf: pl.LazyFrame, lookup: pl.LazyFrame) -> pl.LazyFrame:
    """Transform claims with proper language matching per datatype.

    Each datatype is handled according to where its language information lives.
    Results are concatenated with diagonal alignment to handle differing schemas.
    `lookup` is the chunk's label lookup table (see `CLAIMS_LABELS`).
    """
    base = claims_base(lf)
    prop_labels = lookup_labels(lookup, "property-labels", "property", "property_label")

    transforms = [
        transform_wikibase(base, prop_labels, lookup),
        transform_quantity(base, prop_labels, lookup),
        transform_scalar(base, prop_labels),
        transform_monolingualtext(base, prop_labels),
    ]

    return pl.concat(transforms, how="diagonal")
