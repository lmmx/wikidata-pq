import orjson
import polars as pl
from tqdm import tqdm

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
        .unnest(struct_field)
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
    "monolingualtext": [
        "text",
        "language",
    ],
}

common_schema = {
    "id": pl.String,
    "rank": pl.String,
    "property": pl.String,
    "property-label-lang": pl.String,
    "property-label": pl.String,
    "datatype": pl.String,
    "datavalue": pl.String,  # Now only used for string dtype datavalues
    # The struct schemata cols should go here
}

struct_schemata = {
    "wikibase-item": {
        "wikibase-id": pl.String,
        "wikibase-label": pl.String,
        "wikibase-label-lang": pl.String,
    },
    "wikibase-property": {
        # overload all of these to match wikibase-item's
        # (we can still distinguish a row on the datatype column)
        "wikibase-id": pl.String,
        "wikibase-label": pl.String,
        "wikibase-label-lang": pl.String,
    },
    "globe-coordinate": {
        "latitude": pl.Float64,
        "longitude": pl.Float64,
        "altitude": pl.Float64,
        "precision": pl.Float64,
        "globe": pl.String,
    },
    "quantity": {
        "amount": pl.String,
        "unit": pl.String,
        "upperBound": pl.String,
        "lowerBound": pl.String,
        "unit-label": pl.String,
        "unit-label-lang": pl.String,
    },
    "time": {
        "time": pl.String,
        "timezone": pl.Int64,
        "before": pl.Int64,
        "after": pl.Int64,
        "ts_precision": pl.Int64,
        "calendarmodel": pl.String,
    },
    "monolingualtext": {
        "datavalue": pl.String,  # act like it was a regular simple type
        "mlt-language": pl.String,  # single universal value, we can't filter on it
    },
}

total_schema = {
    **common_schema,
    **{k: v for d in struct_schemata.values() for k, v in d.items()},
}

null_str = pl.lit(None).cast(pl.String)

dts = {}


all_claims = []
for entity_id, claims_dict in tqdm(
    zip(df.get_column("id"), claims_decoded), total=len(claims_decoded)
):
    for property_key, claims_list in claims_dict.items():
        for claim in claims_list:
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
            step0_claims = unpivot_struct(
                frame=claim_frame,
                struct_field="property-labels",
                pivot_var="property-label-lang",
                pivot_val="property-label",
            )
            # Process datatype
            datatype = claim_frame["datatype"].item()
            scalar_dv = datatype in str_snak_values
            if scalar_dv:
                step1b_claims = step0_claims
            else:
                # We have a struct
                # Firstly deal with the wikibase-item/wikibase-property struct
                if datatype in ["wikibase-item", "wikibase-property"]:
                    step1a_claims = step0_claims.select(
                        [
                            "id",
                            "rank",
                            "property",
                            "property-label-lang",
                            "property-label",
                            "datatype",
                            pl.col("datavalue")
                            .struct.rename_fields(["wikibase-id", "wikibase-labels"])
                            .struct.unnest(),
                        ]
                    )
                    if (
                        step0_claims["datavalue"].struct.field("labels").dtype
                        == pl.String
                    ):
                        # For some reason sometimes 'labels' are one universal label
                        label_col_rename = {"wikibase-labels": "wikibase-label"}
                        # Fake a label-lang col, just give it a null
                        dummy_label_lang = null_str.alias("wikibase-label-lang")
                        step1b_claims = (
                            step1a_claims.rename(
                                label_col_rename
                            ).with_columns(  # lose the 's'
                                dummy_label_lang
                            )  # constant null column
                        )
                    else:
                        step1b_claims = unpivot_struct(
                            frame=step1a_claims,
                            struct_field="wikibase-labels",
                            pivot_var="wikibase-label-lang",
                            pivot_val="wikibase-label",
                        )
                elif datatype == "globe-coordinate":
                    nul_f = pl.col(
                        "latitude", "longitude", "altitude", "precision"
                    ).cast(pl.Float64)
                    step1b_claims = step0_claims.unnest("datavalue").with_columns(nul_f)
                elif datatype == "quantity":
                    step1a_claims = step0_claims.unnest("datavalue")
                    if "unit-labels" in step1a_claims.columns:
                        step1b_claims = unpivot_struct(
                            frame=step1a_claims,
                            struct_field="unit-labels",
                            pivot_var="unit-label-lang",
                            pivot_val="unit-label",
                        )
                    else:
                        # Fake a unit-label-lang col, just give it a null
                        dummy_label_lang = null_str.alias("unit-label-lang")
                        step1b_claims = step1a_claims.with_columns(dummy_label_lang)
                elif datatype == "time":
                    step1b_claims = step0_claims.unnest("datavalue").rename(
                        {"precision": "ts_precision"}
                    )
                elif datatype == "monolingualtext":
                    step1b_claims = step0_claims.unnest("datavalue").rename(
                        {"text": "datavalue", "language": "mlt-language"}
                    )
            if scalar_dv or datatype in ["time", "monolingualtext"]:
                final_claim_df = step1b_claims
            else:
                if (
                    datatype == "wikibase-item"
                    and "wikibase-label-lang" in step1b_claims.columns
                ):
                    # N.B. wikibase-item filter must be after the fix so must allow null
                    label_lang_col = pl.col("wikibase-label-lang")
                    prop_lang_col = pl.col("property-label-lang")
                    label_lang_eq_prop_lang = label_lang_col == prop_lang_col
                    final_filter = label_lang_eq_prop_lang.or_(label_lang_col.is_null())
                elif datatype == "quantity" and "unit-labels" in step1a_claims.columns:
                    # N.B. we can check step 1a here before the dummy null got put in
                    final_filter = pl.col("unit-label-lang") == pl.col(
                        "property-label-lang"
                    )
                else:
                    final_filter = True
                final_claim_df = step1b_claims.filter(final_filter)
            all_claims.append(
                final_claim_df.match_to_schema(total_schema, missing_columns="insert")
            )

claims_df = pl.concat(all_claims)
claims_normalized = claims_df
