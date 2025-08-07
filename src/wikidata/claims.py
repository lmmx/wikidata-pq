import time
from math import ceil
from pathlib import Path

import orjson
import polars as pl
from tqdm import tqdm

from .schemas import str_snak_values, total_schema
from .struct_transforms import unpivot_struct

DEBUG_ON_EXC = True


def unpack_claims(df: pl.DataFrame, temp_store_path: Path) -> pl.DataFrame:
    temp_store_path.mkdir(exist_ok=True, parents=True)
    print("Starting claims")
    claims_decoded = map(orjson.loads, df.get_column("claims"))
    print("Done claims parse")

    null_str = pl.lit(None).cast(pl.String)

    # Drop the accumulated list of DataFrames at this size and save the DF
    batch_size = 25
    total_ents = len(df)
    n_batches = ceil(total_ents / batch_size)
    n_digits = len(str(n_batches))

    batched_claims = []
    iterator = tqdm(zip(df.get_column("id"), claims_decoded), total=total_ents)
    for i, (entity_id, claims_dict) in enumerate(iterator, start=1):
        is_save_point = i % batch_size == 0 or i == total_ents
        batch_no = (i - 1) // batch_size
        batch_file = (
            temp_store_path
            / f"batch-{batch_no:0{n_digits}}-of-{n_batches:0{n_digits}}.parquet"
        )
        if batch_file.exists():
            continue
        # for property_key, claims_list in claims_dict.items():
        for claims_list in claims_dict.values():
            for claim in claims_list:
                claim_frame = pl.DataFrame(
                    {
                        "id": entity_id,
                        # "property": property_key, # This is also in mainsnak
                        "mainsnak": claim["mainsnak"],
                        "rank": claim["rank"],
                    }
                ).unnest("mainsnak")
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
                                .struct.rename_fields(
                                    ["wikibase-id", "wikibase-labels"]
                                )
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
                        step1b_claims = step0_claims.unnest("datavalue").with_columns(
                            nul_f
                        )
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
                        final_filter = label_lang_eq_prop_lang.or_(
                            label_lang_col.is_null()
                        )
                    elif (
                        datatype == "quantity"
                        and "unit-labels" in step1a_claims.columns
                    ):
                        # N.B. we can check step 1a here before the dummy null got put in
                        final_filter = pl.col("unit-label-lang") == pl.col(
                            "property-label-lang"
                        )
                    else:
                        final_filter = True
                    final_claim_df = step1b_claims.filter(final_filter)
                try:
                    batched_claims.append(
                        final_claim_df.match_to_schema(
                            total_schema, missing_columns="insert"
                        )
                    )
                    if is_save_point:
                        t0 = time.time()
                        pl.concat(batched_claims).write_parquet(batch_file)
                        took = f"{time.time() - t0:.1f}s"
                        iterator.set_postfix_str(f"Saved batch {batch_no}/{n_batches} in {took}")
                        batched_claims = []
                except Exception:
                    if final_claim_df.is_empty():
                        # Can happen if the datavalue label sub-struct was empty (i.e. 0
                        # languages) and it got unpivotted on (expanding to 0 rows).
                        # We can remedy this by going and fetching the labels ourself
                        print("Empty claim - bad datavalue probably")
                        print(claim_frame)
                        continue  # Drop the row, print so we know if it's not rare
                    else:
                        print("Error validating non-empty claim")
                        import traceback

                        traceback.print_exc()
                        if DEBUG_ON_EXC:
                            breakpoint()

    return result
