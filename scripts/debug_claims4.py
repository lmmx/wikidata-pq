#!/usr/bin/env python3
"""Claims transformation with proper language matching."""

from pathlib import Path

import polars as pl

claims_file = Path("results/claims/chunk_0-00400-of-00546.parquet")


# Common base: explode to individual claims, unnest mainsnak
def get_base(lf: pl.LazyFrame) -> pl.LazyFrame:
    return (
        lf.select(pl.col("claims").explode().struct.unnest())
        .drop("key")
        .explode("value")
        .unnest("value")
        .unnest("mainsnak")
    )


# Datatypes that need language matching (have language-keyed labels in datavalue)
WIKIBASE_TYPES = ["wikibase-item", "wikibase-property"]
# Scalar datatypes - no language in datavalue, property-label lang is sufficient
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


def transform_wikibase(base: pl.LazyFrame) -> pl.LazyFrame:
    """wikibase-item/property: match property-label lang to datavalue.labels lang."""
    return (
        base.filter(pl.col("datatype").is_in(WIKIBASE_TYPES))
        # Explode property-labels: one row per property-label language
        .explode("property-labels")
        .with_columns(
            pl.col("property-labels").struct.rename_fields(
                ["prop_lang", "property_label"]
            )
        )
        .unnest("property-labels")
        # Explode datavalue.labels: one row per datavalue language
        .with_columns(pl.col("datavalue").struct.field("labels").alias("dv_labels"))
        .explode("dv_labels")
        .with_columns(
            pl.col("dv_labels").struct.rename_fields(["dv_lang", "datavalue_label"])
        )
        .unnest("dv_labels")
        # FILTER: keep only where languages match
        .filter(pl.col("prop_lang") == pl.col("dv_lang"))
        # Rename to final 'language' column, drop redundant
        .rename({"prop_lang": "language"})
        .drop("dv_lang")
    )


def transform_quantity(base: pl.LazyFrame) -> pl.LazyFrame:
    """quantity: match property-label lang to unit-labels lang (when unit has labels)."""
    qty_base = base.filter(pl.col("datatype") == "quantity")

    # Split: quantities with unit-labels vs without
    has_unit_labels = pl.col("datavalue").struct.field("unit-labels").list.len() > 0

    # With unit-labels: need to match languages
    with_units = (
        qty_base.filter(has_unit_labels)
        .explode("property-labels")
        .with_columns(
            pl.col("property-labels").struct.rename_fields(
                ["prop_lang", "property_label"]
            )
        )
        .unnest("property-labels")
        .with_columns(
            pl.col("datavalue").struct.field("unit-labels").alias("unit_labels")
        )
        .explode("unit_labels")
        .with_columns(
            pl.col("unit_labels").struct.rename_fields(["unit_lang", "unit_label"])
        )
        .unnest("unit_labels")
        .filter(pl.col("prop_lang") == pl.col("unit_lang"))
        .rename({"prop_lang": "language"})
        .drop("unit_lang")
    )

    # Without unit-labels (unit="1"): just use property-label language
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
    """Scalar types: no language in datavalue, property-label lang is the partition key."""
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
    """monolingualtext: datavalue.language IS the language to partition on."""
    return (
        base.filter(pl.col("datatype") == "monolingualtext")
        # The language is inside datavalue, not in property-labels
        .with_columns(pl.col("datavalue").struct.field("language").alias("language"))
        # Still need property_label, but we DON'T filter - we use dv's language as partition key
        # This means property_label will be in whatever languages exist, not matched
        # Actually... should we match here too? Let's think...
        # The 'text' is in a specific language. The property-label tells you what the text IS.
        # I think we want: partition by datavalue.language, but keep the property-label in that same language if available
        .explode("property-labels")
        .with_columns(
            pl.col("property-labels").struct.rename_fields(
                ["prop_lang", "property_label"]
            )
        )
        .unnest("property-labels")
        .filter(pl.col("prop_lang") == pl.col("language"))
        .drop("prop_lang")
    )


# Test on sample
lf = pl.scan_parquet(claims_file).drop_nulls().head(100)
base = get_base(lf).head(500).collect()

print("=== Testing wikibase transform ===")
wb = transform_wikibase(base.lazy()).collect()
print(f"Rows: {len(wb)}")
print(f"Columns: {wb.columns}")
print(wb.head(3))
print(
    f"\nLanguage distribution:\n{wb.group_by('language').len().sort('len', descending=True).head(5)}"
)

print("\n=== Testing scalar transform ===")
sc = transform_scalar(base.lazy()).collect()
print(f"Rows: {len(sc)}")
print(sc.head(3))

print("\n=== Testing quantity transform ===")
qty = transform_quantity(base.lazy()).collect()
print(f"Rows: {len(qty)}")
if len(qty) > 0:
    print(qty.head(3))
