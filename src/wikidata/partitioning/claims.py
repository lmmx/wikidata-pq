"""Claims transform for partitioning.

Claims are not split by language (see `UNSPLIT_COL` in config): each claim is one row,
with its property, value and unit ids. Their labels in every language are in the
claims_labels table and the entity's own labels in the labels table, for the user to join
in whichever languages they want (2026-09-26 journal, claims unsplit).
"""

import polars as pl


def claims_base(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Explode claims to one row per claim, with the mainsnak unnested."""
    return (
        lf.explode("claims", empty_as_null=True)
        .select("id", pl.col("claims").struct.unnest())
        .drop("key")
        .explode("value", empty_as_null=True)
        .unnest("value")
        .unnest("mainsnak")
    )
