"""Claims-specific transforms for language partitioning.

A claim goes into language L if its property has a label in L or its entity (the
subject, `id`) has a label in L (rule E in the 2026-09-26 journal), and a monolingual
text claim also into the text's own language. Each row carries the property, value and
unit labels in L where they exist, and null where they do not (the ids remain, and every
language's labels are in the claims_labels table).

The label maps are not in the claims rows: processing moves them to a per-chunk lookup
table (field, ref, language, label), and the transforms join against it. The entity's
own labels come from the chunk's labels table.
"""

import polars as pl


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


def entity_languages(labels: pl.LazyFrame) -> pl.LazyFrame:
    """The languages each entity has a label in, as (id, language)."""
    return (
        labels.explode("labels", empty_as_null=True)
        .select("id", pl.col("labels").struct.field("key").alias("language"))
        .drop_nulls()
    )


def prepare_claims(
    lf: pl.LazyFrame, lookup: pl.LazyFrame, labels: pl.LazyFrame
) -> pl.LazyFrame:
    """One row per claim per language it goes into, with the labels in that language.

    `lookup` is the chunk's label lookup table (see `Table.CLAIMS_LABELS`), `labels` the
    chunk's labels table (id, labels).
    """
    base = claims_base(lf).with_row_index("_claim")
    prop_labels = lookup_labels(lookup, "property-labels", "property", "property_label")
    value_labels = lookup_labels(lookup, "labels", "_dv_id", "datavalue_label")
    unit_labels = lookup_labels(lookup, "unit-labels", "_unit", "unit_label")

    by_property = base.select("_claim", "property").join(
        prop_labels.select("property", "language"), on="property"
    )
    by_entity = base.select("_claim", "id").join(entity_languages(labels), on="id")
    by_text = base.filter(pl.col("datatype") == "monolingualtext").select(
        "_claim", pl.col("datavalue").struct.field("language")
    )
    languages = pl.concat(
        [f.select("_claim", "language") for f in (by_property, by_entity, by_text)]
    ).unique()

    return (
        base.join(languages, on="_claim")
        .with_columns(
            pl.col("datavalue").struct.field("id").alias("_dv_id"),
            pl.col("datavalue").struct.field("unit").alias("_unit"),
        )
        .join(prop_labels, on=["property", "language"], how="left")
        .join(value_labels, on=["_dv_id", "language"], how="left")
        .join(unit_labels, on=["_unit", "language"], how="left")
        .drop("_claim", "_dv_id", "_unit")
    )
