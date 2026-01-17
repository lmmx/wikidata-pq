"""Claims-specific transforms for language partitioning.

Claims are complex: the language for partitioning depends on the datatype.
Each datatype has different nested structures containing language information:

- wikibase-item/property: match property-labels lang to datavalue.labels lang
- quantity: match property-labels lang to unit-labels lang (when unit has labels)
- scalar types (string, external-id, time, etc.): use property-labels lang directly
- monolingualtext: use datavalue.language, match to property-labels

The transforms use an efficient approach that avoids cartesian product explosion
by filtering on language set membership before extracting matching labels.
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

    Uses efficient approach: filter on language set membership, then map_elements
    to extract the single matching label (avoids cartesian product explosion).
    """
    return (
        base.filter(pl.col("datatype").is_in(WIKIBASE_TYPES))
        # Get available languages in datavalue.labels
        .with_columns(
            pl.col("datavalue")
            .struct.field("labels")
            .list.eval(pl.element().struct.field("key"))
            .alias("dv_langs")
        )
        # Explode property-labels only
        .explode("property-labels")
        .with_columns(
            pl.col("property-labels").struct.rename_fields(
                ["language", "property_label"]
            )
        )
        .unnest("property-labels")
        # Filter to languages that exist in datavalue.labels
        .filter(pl.col("language").is_in(pl.col("dv_langs")))
        # Extract matching label via map_elements (avoids cartesian product)
        .with_columns(
            pl.struct(["datavalue", "language"])
            .map_elements(
                lambda row: next(
                    (
                        item["value"]
                        for item in (row["datavalue"]["labels"] or [])
                        if item["key"] == row["language"]
                    ),
                    None,
                ),
                return_dtype=pl.String,
            )
            .alias("datavalue_label")
        )
        .drop("dv_langs")
    )


def transform_quantity(base: pl.LazyFrame) -> pl.LazyFrame:
    """quantity: match property-label lang to unit-labels lang when unit has labels.

    When unit="1" (dimensionless), there are no unit-labels, so we just use
    property-label language directly.
    """
    qty_base = base.filter(pl.col("datatype") == "quantity")
    has_unit_labels = pl.col("datavalue").struct.field("unit-labels").list.len() > 0

    # With unit-labels: need language matching
    with_units = (
        qty_base.filter(has_unit_labels)
        .with_columns(
            pl.col("datavalue")
            .struct.field("unit-labels")
            .list.eval(pl.element().struct.field("key"))
            .alias("unit_langs")
        )
        .explode("property-labels")
        .with_columns(
            pl.col("property-labels").struct.rename_fields(
                ["language", "property_label"]
            )
        )
        .unnest("property-labels")
        .filter(pl.col("language").is_in(pl.col("unit_langs")))
        .with_columns(
            pl.struct(["datavalue", "language"])
            .map_elements(
                lambda row: next(
                    (
                        item["value"]
                        for item in (row["datavalue"]["unit-labels"] or [])
                        if item["key"] == row["language"]
                    ),
                    None,
                ),
                return_dtype=pl.String,
            )
            .alias("unit_label")
        )
        .drop("unit_langs")
    )

    # Without unit-labels: property-label language is sufficient
    without_units = (
        qty_base.filter(~has_unit_labels)
        .explode("property-labels")
        .with_columns(
            pl.col("property-labels").struct.rename_fields(
                ["language", "property_label"]
            )
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
            pl.col("property-labels").struct.rename_fields(
                ["language", "property_label"]
            )
        )
        .unnest("property-labels")
    )


def transform_monolingualtext(base: pl.LazyFrame) -> pl.LazyFrame:
    """monolingualtext: datavalue.language IS the language to partition on.

    We still want the property_label in the matching language where available.
    """
    return (
        base.filter(pl.col("datatype") == "monolingualtext")
        # The language comes from datavalue
        .with_columns(pl.col("datavalue").struct.field("language").alias("language"))
        # Get available languages in property-labels for filtering
        .with_columns(
            pl.col("property-labels")
            .list.eval(pl.element().struct.field("key"))
            .alias("prop_langs")
        )
        # Only keep rows where property-label exists in this language
        .filter(pl.col("language").is_in(pl.col("prop_langs")))
        # Extract matching property_label
        .with_columns(
            pl.struct(["property-labels", "language"])
            .map_elements(
                lambda row: next(
                    (
                        item["value"]
                        for item in (row["property-labels"] or [])
                        if item["key"] == row["language"]
                    ),
                    None,
                ),
                return_dtype=pl.String,
            )
            .alias("property_label")
        )
        .drop("prop_langs", "property-labels")
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
