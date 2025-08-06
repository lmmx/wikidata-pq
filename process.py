import orjson
import polars as pl

# df = pl.read_parquet("*.parquet")
df = pl.read_parquet("data/*.parquet")


def unpivot_from_struct_col(struct_col: str, struct_field: str, var_name: str):
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


def unpivot_from_list_struct_col(struct_col: str, struct_field: str, var_name: str):
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


labels = unpivot_from_struct_col("labels", "value", "language")
descs = unpivot_from_struct_col("descriptions", "value", "language")

aliases = unpivot_from_list_struct_col("aliases", "value", "language")
links = unpivot_from_struct_col("sitelinks", "title", "site")

print("Starting claims")
claims_decoded = list(map(orjson.loads, df.get_column("claims")))
print("Done claims parse")

# claims = (
#     df.rename({"id": "_id"})
#     .select("_id", pl.col("claims").str.json_decode(infer_schema_length=10**10))
#     .unnest("claims")
#     .unpivot(index="_id", variable_name="property")
#     .drop_nulls()
#     .explode("value")
#     .with_columns(pl.col("value").struct.unnest())
#     .drop("value")
#     .explode("references")
# )


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
        .with_columns(pl.col(struct_field).struct.unnest())
        .drop(struct_field)
        .unpivot(index=index_cols, variable_name=pivot_var, value_name=pivot_val)
        .with_columns(pl.col(pivot_var).str.strip_prefix(temp_col_prefix))
    )
    return new_frame


# Some entity IDs inferred to be str dtype datavalue from parser at:
# https://github.com/maxlath/wikibase-sdk/blob/2447f7d1355d3461adb2676b0c5336534932feda/src/types/snakvalue.ts#L68

str_snak_values = [
    "commonsMedia",  # str datavalue
    "geo-shape",  # str datavalue
    "tabular-data",  # str datavalue
    "url",  # str datavalue
    "external-id",  # str datavalue
    "wikibase-property",  # str datavalue
    "monolingualtext",  # str datavalue
    "string",  # str datavalue
    "musical-notation",  # str datavalue
    "math",  # str datavalue
    "entity-schema",  # str datavalue
    "wikibase-lexeme",  # str datavalue
    "wikibase-form",  # str datavalue
    "wikibase-sense",  # str datavalue
]
struct_snak_values = {
    "wikibase-item": [
        "id",
        "labels",  # categorical
    ],
    "globe-coordinate": [
        "latitude",
        "longitude",
        "altitude",
        "precision",
        "globe",
    ],
    "quantity": [
        "amount",
        "unit",
        "upperBound",
        "lowerBound",
        "unit-labels",
    ],
    "time": [
        "time",
        "timezone",
        "before",
        "after",
        "precision",
        "calendarmodel",
    ],
}
struct_categoricals = {
    "wikibase-item": {"labels": ["wikibase-item-label-lang", "wikibase-item-label"]},
    "quantity": {"unit-labels": []},
}

goal = "wikibase-item"
dts = {}

all_claims = []
for entity_id, claims_dict in zip(df.get_column("id"), claims_decoded):
    for property_key, claims_list in claims_dict.items():
        for claim in claims_list:
            if claim["mainsnak"]["datatype"] != goal:
                continue
            claim_frame = pl.DataFrame(
                {
                    "id": entity_id,
                    "property": property_key,
                    **claim["mainsnak"],
                    "rank": claim["rank"],
                }
            )
            if "datavalue" not in claim_frame:
                continue  # invalid entry
            dv_fr = claim_frame["datavalue"].struct.unnest()
            dt = dv_fr["labels"].dtype
            dt_str = str(dt).split("(")[0]
            dts.setdefault(dt_str, 0)
            dts[dt_str] += 1
            if dts[dt_str] == 1:
                print(dv_fr)
            continue
            scalar_dv = claim_frame["datatype"].item() in str_snak_values
            if scalar_dv:
                dv_unpivotted_claims = claim_frame
            else:
                unpacked_dv_claims = claim_frame.select(
                    [
                        "id",
                        "property",
                        pl.col("datavalue")
                        .struct.rename_fields(["datavalue-id", "datavalue-labels"])
                        .struct.unnest(),
                        "datatype",
                        "property-labels",
                        "rank",
                    ]
                )
                dv_unpivotted_claims = unpivot_struct(
                    frame=unpacked_dv_claims,
                    struct_field="datavalue-labels",
                    pivot_var="datavalue-label-lang",
                    pivot_val="datavalue-label",
                )
            prop_unpivotted_claims = unpivot_struct(
                frame=dv_unpivotted_claims,
                struct_field="property-labels",
                pivot_var="property-label-lang",
                pivot_val="property-label",
            )
            if scalar_dv:
                final_claim_df = prop_unpivotted_claims
            else:
                final_claim_df = prop_unpivotted_claims.filter(
                    pl.col("datavalue-label-lang") == pl.col("property-label-lang")
                )
            all_claims.append(final_claim_df)
            break

claims_df = pl.concat(all_claims)
claims_normalized = claims_df
