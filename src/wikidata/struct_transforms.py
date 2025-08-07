import polars as pl

__all__ = [
    "unpivot_from_struct_col",
    "unpivot_from_list_struct_col",
    "unpivot_struct",
]


def unpivot_from_struct_col(
    df: pl.DataFrame, struct_col: str, struct_field: str, var_name: str
):
    return (
        df.rename({"id": "_id"})
        .select("_id", pl.col(struct_col).str.json_decode(infer_schema_length=10**8))
        .select("_id", pl.col(struct_col).struct.unnest())
        .unpivot(index="_id", variable_name=var_name)
        .rename({"value": struct_field})
        .with_columns(pl.col(struct_field).struct.field(struct_field))
        .drop_nulls()
        .rename({"_id": "id"})
    )


def unpivot_from_list_struct_col(
    df: pl.DataFrame, struct_col: str, struct_field: str, var_name: str
):
    return (
        df.rename({"id": "_id"})
        .select("_id", pl.col(struct_col).str.json_decode(infer_schema_length=10**8))
        .select("_id", pl.col(struct_col).struct.unnest())
        .unpivot(index="_id", variable_name=var_name)
        .drop_nulls()
        .explode("value")
        .with_columns(pl.col("value").struct.field(struct_field))
        .rename({"_id": "id"})
    )


def unpivot_struct(
    frame: pl.DataFrame, struct_field: str, pivot_var: str, pivot_val: str
) -> pl.DataFrame:
    """Unpivot struct fields, avoiding namespace collision by temporarily renaming."""
    orig_field_names = frame[struct_field].struct.fields
    temp_col_prefix = "__TEMP_COL_"
    temp_field_names = [f"{temp_col_prefix}{fn}" for fn in orig_field_names]
    index_cols = [fc for fc in frame.columns if fc != struct_field]
    new_frame = (
        frame.with_columns(pl.col(struct_field).struct.rename_fields(temp_field_names))
        .unnest(struct_field)
        .unpivot(index=index_cols, variable_name=pivot_var, value_name=pivot_val)
        .with_columns(pl.col(pivot_var).str.strip_prefix(temp_col_prefix))
    )
    return new_frame
