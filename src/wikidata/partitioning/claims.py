"""Claims transform for partitioning.

Claims are not split by language (see `UNSPLIT_COL` in config): each claim is one row,
with its property, value and unit ids. Their labels in every language are in the
claims_labels table and the entity's own labels in the labels table, for the user to join
in whichever languages they want (2026-09-26 journal, claims unsplit).
"""

import numpy as np
import polars as pl
import pyarrow as pa


# A release's statements have an id and a type of their own (see process.py), named apart
# from the entity's `id` once the statement is unnested beside it
STATEMENT_RENAMES = {"id": "statement_id", "type": "statement_type"}


def explode(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """`df.explode(col, empty_as_null=True)`, done in Arrow.

    Polars gathers the values of a list column it doesn't know to be free of empty
    lists, as one read from Parquet is, an element at a time, cloning each element's
    nested arrays: partitioning a chunk's claims took up to 71 s of CPU and 13.4 GB,
    and chunk 0's ran out of memory (docs/journal/2026-10-21-claims-arrow-explode.md).
    In Arrow it is a flatten of the lists and a take of the other columns.
    """
    table = df.to_arrow()
    idx = table.schema.get_field_index(col)
    field = table.schema.field(idx)
    lists = table.column(idx).combine_chunks()
    lengths = lists.value_lengths().fill_null(0).to_numpy(zero_copy_only=False)
    values = lists.flatten()  # null and empty lists contribute no values
    # A null or empty list gives one null row, in its place
    counts = np.maximum(lengths, 1)
    if (lengths > 0).all():
        exploded = values
    else:
        out_starts = np.cumsum(counts) - counts
        value_starts = np.cumsum(lengths) - lengths
        take = np.arange(counts.sum()) - np.repeat(out_starts - value_starts, counts)
        exploded = values.take(pa.array(take, mask=np.repeat(lengths == 0, counts)))
    others = table.remove_column(idx).take(np.repeat(np.arange(len(lists)), counts))
    return pl.DataFrame(others.add_column(idx, field.with_type(exploded.type), exploded))


def claims_base(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Explode claims to one row per claim, with the mainsnak unnested.

    Both explodes run in Arrow (see `explode`), so the claims are collected here.
    """
    claims = (
        explode(lf.collect(), "claims")
        .select("id", pl.col("claims").struct.unnest())
        .drop("key")
    )
    return (
        explode(claims, "value")
        .lazy()
        .select(
            "id",
            pl.col("value").name.map_fields(lambda f: STATEMENT_RENAMES.get(f, f)),
        )
        .unnest("value")
        .unnest("mainsnak")
    )
