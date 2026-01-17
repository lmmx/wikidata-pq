"""Claims-specific transforms for language partitioning.

Claims are complex: the language for partitioning depends on the datatype.
Each datatype has different nested structures containing language information:

- wikibase-item/property: match property-labels lang to datavalue.labels lang
- quantity: match property-labels lang to unit-labels lang (when unit has labels)
- scalar types (string, external-id, time, etc.): use property-labels lang directly
- monolingualtext: use datavalue.language, match to property-labels

The transforms use join-based lookups to extract matching labels efficiently,
avoiding both cartesian products and slow map_elements calls.
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
        lf.select(pl.col("claims").explode().struct.unnest())
        .drop("key")
        .explode("value")
        .unnest("value")
        .unnest("mainsnak")
    )


def transform_wikibase(base: pl.LazyFrame) -> pl.LazyFrame:
    """wikibase-item/property: match property-label lang to datavalue.labels lang.

    Uses join-based lookup: create a lookup table from exploded labels,
    then inner join on (row_id, language) to get matching values.
    """
    indexed = (
        base
        .filter(pl.col("datatype").is_in(WIKIBASE_TYPES))
        .with_row_index("_row_id")
    )

    # Lookup table: explode datavalue.labels → (row_id, lang, label_value)
    labels_lookup = (
        indexed
        .select(
            "_row_id",
            pl.col("datavalue").struct.field("labels").alias("_dv_labels"),
        )
        .explode("_dv_labels")
        .with_columns(
            pl.col("_dv_labels").struct.field("key").alias("_lang"),
            pl.col("_dv_labels").struct.field("value").alias("datavalue_label"),
        )
        .select("_row_id", "_lang", "datavalue_label")
    )

    # Main table: explode property-labels
    main = (
        indexed
        .explode("property-labels")
        .with_columns(
            pl.col("property-labels").struct.rename_fields(["language", "property_label"])
        )
        .unnest("property-labels")
    )

    # Inner join naturally filters to matching languages only
    return (
        main
        .join(labels_lookup, left_on=["_row_id", "language"], right_on=["_row_id", "_lang"], how="inner")
        .drop("_row_id")
    )


def transform_quantity(base: pl.LazyFrame) -> pl.LazyFrame:
    """quantity: match property-label lang to unit-labels lang when unit has labels.

    When unit="1" (dimensionless), there are no unit-labels, so we just use
    property-label language directly.
    """
    qty_base = base.filter(pl.col("datatype") == "quantity")
    has_unit_labels = pl.col("datavalue").struct.field("unit-labels").list.len() > 0

    # With unit-labels: use join-based lookup
    with_units_base = qty_base.filter(has_unit_labels).with_row_index("_row_id")

    unit_lookup = (
        with_units_base
        .select(
            "_row_id",
            pl.col("datavalue").struct.field("unit-labels").alias("_unit_labels"),
        )
        .explode("_unit_labels")
        .with_columns(
            pl.col("_unit_labels").struct.field("key").alias("_lang"),
            pl.col("_unit_labels").struct.field("value").alias("unit_label"),
        )
        .select("_row_id", "_lang", "unit_label")
    )

    with_units_main = (
        with_units_base
        .explode("property-labels")
        .with_columns(
            pl.col("property-labels").struct.rename_fields(["language", "property_label"])
        )
        .unnest("property-labels")
    )

    with_units = (
        with_units_main
        .join(unit_lookup, left_on=["_row_id", "language"], right_on=["_row_id", "_lang"], how="inner")
        .drop("_row_id")
    )

    # Without unit-labels: property-label language is sufficient
    without_units = (
        qty_base.filter(~has_unit_labels)
        .explode("property-labels")
        .with_columns(
            pl.col("property-labels").struct.rename_fields(["language", "property_label"])
        )
        .unnest("property-labels")
    )

    return pl.concat([with_units, without_units], how="diagonal")


def transform_scalar(base: pl.LazyFrame) -> pl.LazyFrame:
    """Scalar types: no language in datavalue, property-label lang is partition key."""
    return (
        base.filter(pl.col("datatype").is_in(SCALAR_TYPES))
        .explode("property-labels")
        .with_columns(
            pl.col("property-labels").struct.rename_fields(["language", "property_label"])
        )
        .unnest("property-labels")
    )


def transform_monolingualtext(base: pl.LazyFrame) -> pl.LazyFrame:
    """monolingualtext: datavalue.language IS the language to partition on.

    We still want the property_label in the matching language where available.
    Uses join-based lookup on property-labels.
    """
    indexed = (
        base
        .filter(pl.col("datatype") == "monolingualtext")
        .with_columns(pl.col("datavalue").struct.field("language").alias("language"))
        .with_row_index("_row_id")
    )

    # Lookup table: explode property-labels → (row_id, lang, label_value)
    prop_lookup = (
        indexed
        .select("_row_id", "property-labels")
        .explode("property-labels")
        .with_columns(
            pl.col("property-labels").struct.field("key").alias("_lang"),
            pl.col("property-labels").struct.field("value").alias("property_label"),
        )
        .select("_row_id", "_lang", "property_label")
    )

    # Main table already has language from datavalue
    main = indexed.drop("property-labels")

    # Inner join to get matching property_label
    return (
        main
        .join(prop_lookup, left_on=["_row_id", "language"], right_on=["_row_id", "_lang"], how="inner")
        .drop("_row_id")
    )


def prepare_claims(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Transform claims with proper language matching per datatype.

    Each datatype is handled according to where its language information lives.
    Results are concatenated with diagonal alignment to handle differing schemas.
    """
    base = claims_base(lf)

    transforms = [
        transform_wikibase(base),
        transform_quantity(base),
        transform_scalar(base),
        transform_monolingualtext(base),
    ]

    return pl.concat(transforms, how="diagonal")