"""Claims transform for partitioning.

Claims are not split by language (see `UNSPLIT_COL` in config): each claim is one row,
with its property, value and unit ids. Their labels in every language are in the
claims_labels table and the entity's own labels in the labels table, for the user to join
in whichever languages they want (2026-09-26 journal, claims unsplit).
"""

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc


# A release's statements have an id and a type of their own (see process.py), named apart
# from the entity's `id` once the statement is unnested beside it
STATEMENT_RENAMES = {"id": "statement_id", "type": "statement_type"}


def explode(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """`df.explode(col, empty_as_null=True)`, done in Arrow.

    Polars gathers the values of a list column it doesn't know to be free of empty
    lists, as one read from Parquet is, an element at a time, cloning each element's
    nested arrays: partitioning a chunk's claims took up to 71 s of CPU and 13.4 GB,
    and chunk 0's ran out of memory (docs/journal/2026-10-21-claims-arrow-explode.md).
    In Arrow it is a flatten of the lists; the other columns are gathered in Polars.

    Without null or empty lists, the list column goes to Arrow in Polars' own types
    (string views) and its values come back the same way, so neither way copies it.
    Those that have them need a take of the values, which pyarrow has no kernel for on
    string views, so they go in Arrow's large types
    (docs/journal/2026-10-22-claims-explode-views.md).
    """
    series = df.get_column(col)
    lengths = series.list.len().fill_null(0).cast(pl.Int64).to_numpy()
    empty = lengths == 0
    level = pl.CompatLevel.oldest() if empty.any() else pl.CompatLevel.newest()
    values = series.to_arrow(compat_level=level).flatten()  # null and empty lists add none
    # A null or empty list gives one null row, in its place
    counts = np.maximum(lengths, 1)
    if empty.any():
        out_starts = np.cumsum(counts) - counts
        value_starts = np.cumsum(lengths) - lengths
        take = np.arange(counts.sum()) - np.repeat(out_starts - value_starts, counts)
        values = values.take(pa.array(take, mask=np.repeat(empty, counts)))
    others = df.drop(col).gather(np.repeat(np.arange(len(series)), counts))
    return others.insert_column(df.get_column_index(col), pl.Series(col, values))


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


def _has_empty(lists: pa.Array) -> bool:
    """Whether any of `lists` is empty: its shortest list has no values (None, so False,
    for an array of null lists or none).

    No Python value goes into an Arrow compute call here (as `pc.equal(lengths, 0)` would
    take one): pyarrow converts a Python value with a check for pandas types, which
    imports pandas. Nothing else in a chunk worker imports it, and the import was most of
    `statements`' CPU (a quarter of a second, against 0.03 s for the flatten itself) and
    30-50 MB of the worker's peak RSS."""
    return pc.min(pc.list_value_length(lists)).as_py() == 0


def _statement_columns(ids: pa.Array, claims: pa.Array) -> dict[str, pa.Array] | None:
    """A batch's claims as `claims_base` lays them out, by flattening their lists in
    Arrow: no values are copied, only the entity ids are gathered. None if the batch has
    a null id, an empty list, or a null property group or statement list, which
    `claims_base` gives a row of nulls that a flatten drops."""
    groups_parent = pc.list_parent_indices(claims)
    groups = claims.flatten()  # a null list adds none, as drop_nulls drops its row
    key, value = groups.flatten()  # the groups' fields, with the groups' nulls
    nulls = ids.null_count or groups.null_count or value.null_count
    if nulls or _has_empty(claims) or _has_empty(value):
        return None
    statements_parent = pc.list_parent_indices(value)
    statements = value.flatten()
    columns = {"id": pc.take(ids, pc.take(groups_parent, statements_parent))}
    for field, column in zip(statements.type, statements.flatten()):
        if field.name == "mainsnak":
            columns |= {f.name: c for f, c in zip(column.type, column.flatten())}
        else:
            columns[STATEMENT_RENAMES.get(field.name, field.name)] = column
    return columns


def statements(table: pa.Table) -> pa.Table:
    """`claims_base` on the claims as genson writes them (`id`, `claims`), rows with a
    null dropped first as `prepare_for_partition` drops them: one row per statement.

    Each batch's lists are flattened in Arrow (`_statement_columns`), so the statements'
    values are not copied. The columns come out in the Arrow types `claims_base`'s frame
    exports to (its schema, read from it run on no rows); a batch whose columns do not
    have them raises, as pyarrow's cast corrupts the claims' nested structs (see
    compact._source_batches). A table with a row a flatten would drop goes through
    `claims_base` itself.
    """
    # No rows of the table, sliced: `Schema.empty_table` imports pandas (see `_has_empty`)
    empty = pl.from_arrow(table.slice(0, 0))
    schema = claims_base(empty.lazy()).collect().to_arrow().schema
    batches = []
    for batch in table.to_batches():
        columns = _statement_columns(batch.column("id"), batch.column("claims"))
        if columns is None:
            df = pl.from_arrow(table).drop_nulls()
            return claims_base(df.lazy()).collect().to_arrow()
        if [(n, c.type) for n, c in columns.items()] != [(f.name, f.type) for f in schema]:
            raise ValueError("The claims' columns are not those claims_base makes")
        batches.append(pa.RecordBatch.from_arrays(list(columns.values()), schema=schema))
    return pa.Table.from_batches(batches, schema=schema)


def by_id(table: pa.Table) -> pa.Table:
    """`table` stably sorted by `id` in string order, each id's rows in their order: the
    table itself if it is already, as a chunk's statements nearly are (the dump is in id
    order, which is string order among ids of one length), else its rows taken in that
    order (one copy of the rows). finalise sorts each key the same way across all its
    files (sort_by_id), and an entity's statements are all in its chunk, so the published
    rows and their order are the same either way; each claims partition file is one
    sorted run of ids, a row group's ids as narrow a range as its rows allow."""
    ids = table.column("id").combine_chunks()
    n = len(ids)
    if n < 2 or pc.all(pc.less_equal(ids.slice(0, n - 1), ids.slice(1))).as_py():
        return table
    return table.take(pc.sort_indices(table, [("id", "ascending")]))  # a stable sort
