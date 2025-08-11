import time
import traceback
from enum import StrEnum
from math import ceil
from pathlib import Path

import orjson
import polars as pl
from tqdm import tqdm

from .schemas import coalesced_cols, final_schema, str_snak_values, total_schema
from .struct_transforms import unpivot_struct

# Keep your existing imports & functions (unpack_claim_struct, etc.)
DEBUG_ON_EXC = True


class StructDatatype(StrEnum):
    WIKIBASE_ITEM = "wikibase-item"
    WIKIBASE_PROPERTY = "wikibase-property"
    GLOBE_COORDINATE = "globe-coordinate"
    QUANTITY = "quantity"
    TIME = "time"
    MONOLINGUALTEXT = "monolingualtext"


null_str = pl.lit(None).cast(pl.String)


def unpack_claim_struct(
    step0_claims: pl.DataFrame,
    datatype: StructDatatype,
) -> pl.DataFrame:
    assert datatype in StructDatatype, f"The datatype {datatype!r} was not expected"
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
        dv_labels_dtype = step1a_claims["wikibase-labels"].dtype
        if dv_labels_dtype == pl.Struct([]):
            # Awkward, the wikibase-item labels are missing. Just set it to null
            step1a_claims = step1a_claims.with_columns(
                null_str.alias("wikibase-labels")
            )
            dv_labels_dtype = pl.String
        if dv_labels_dtype == pl.String:
            # For some reason sometimes 'labels' are one universal label
            label_col_rename = {"wikibase-labels": "wikibase-label"}
            # Fake a label-lang col, just give it a null
            dummy_label_lang = null_str.alias("wikibase-label-lang")
            step1b_claims = (
                step1a_claims.rename(label_col_rename).with_columns(  # lose the 's'
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
        nul_f = pl.col("latitude", "longitude", "altitude", "precision").cast(
            pl.Float64
        )
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
            {"text": "mlt-text", "language": "mlt-language"}
        )
    else:
        raise ValueError(f"Unexpected claim datatype: {datatype}")
    return step1b_claims


def process_single_entity(entity_id: str, claims_dict: dict) -> list[pl.DataFrame]:
    results = []
    for claims_list in claims_dict.values():
        for claim in claims_list:
            mainsnak = claim["mainsnak"]
            datatype = mainsnak["datatype"]
            try:
                datavalue = mainsnak["datavalue"]
            except KeyError:
                continue  # invalid entry
            scalar_dv = datatype in str_snak_values

            claim_frame = pl.DataFrame(
                {
                    "id": entity_id,
                    # "property": property_key, # This is also in mainsnak
                    "mainsnak": mainsnak,
                    "rank": claim["rank"],
                }
            ).unnest("mainsnak")

            # -- BEGIN PROCESSING ONE CLAIM --

            step0_claims = unpivot_struct(
                frame=claim_frame,
                struct_field="property-labels",
                pivot_var="property-label-lang",
                pivot_val="property-label",
            )
            # Process datatype
            if scalar_dv:
                final_claim_df = step0_claims
            else:
                step1b_claims = unpack_claim_struct(step0_claims, datatype=datatype)
                if datatype in ["time", "monolingualtext"]:
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
                    elif datatype == "quantity" and "unit-labels" in datavalue:
                        # N.B. we can check step 1a here before the dummy null got put in
                        final_filter = pl.col("unit-label-lang") == pl.col(
                            "property-label-lang"
                        )
                    else:
                        final_filter = True
                    final_claim_df = step1b_claims.filter(final_filter)
            try:
                # Expand to total schema, coalesce cols, then validate final schema
                validated_claims = (
                    final_claim_df.match_to_schema(
                        total_schema, missing_columns="insert"
                    )
                    .select(
                        # We ensured unit, property, and wikibase-label langs match
                        # so coalescing is like a union: first non-null, else null
                        pl.coalesce(coalesced_cols["language"]).alias("language"),
                        pl.exclude(coalesced_cols["language"]),
                    )
                    .select(
                        pl.coalesce(coalesced_cols["datavalue"]).alias("datavalue"),
                        pl.exclude(coalesced_cols["datavalue"]),
                    )
                    .match_to_schema(final_schema)
                )
            except Exception:
                print("Error validating non-empty claim")

                traceback.print_exc()
                if DEBUG_ON_EXC:
                    breakpoint()
            else:
                if validated_claims.is_empty():
                    # Can happen if the datavalue label sub-struct was empty (i.e. 0
                    # languages) and it got unpivotted on (expanding to 0 rows).
                    # We can remedy this by going and fetching the labels ourself
                    print("Empty claim - bad datavalue probably")
                    print(claim_frame)
                    continue  # Drop the row, print so we know if it's not rare

                validated_ids = validated_claims.get_column("id").unique()
                id_ensured = validated_ids.len() == claim_frame.height
                if not id_ensured:
                    breakpoint()

                results.append(validated_claims)

    return results


def unpack_claims(
    df: pl.DataFrame, temp_store_path: Path, batch_size: int = 100
) -> pl.DataFrame:
    total_ents = len(df)

    # Drop the accumulated list of DataFrames at this size and save the DF
    n_batches = ceil(total_ents / batch_size)
    n_digits = len(str(n_batches))

    print("Parsing claims")
    claims_decoded = map(orjson.loads, df.get_column("claims"))
    print(
        f"Parsed {total_ents} claims, splitting into {n_batches} batches (size {batch_size})"
    )

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
            iterator.set_postfix_str(f"Reloaded batch {batch_no+1}/{n_batches}")
            continue
        entity_claims = process_single_entity(entity_id, claims_dict)
        batched_claims.extend(entity_claims)

        expected_batch_ids = batch_size if i < total_ents else total_ents % batch_size
        if is_save_point:
            save_batch(
                batched_claims,
                batch_file,
                expected_batch_ids,
                batch_no,
                n_batches,
                iterator,
            )
            batched_claims = []

    return pl.read_parquet(temp_store_path / "*.parquet")


def save_batch(
    batched_claims: list[pl.DataFrame],
    batch_file: Path,
    expected_batch_ids: int,
    batch_no: int,
    n_batches: int,
    iterator: tqdm,
) -> None:
    t0 = time.time()
    batch_df = pl.concat(batched_claims)
    batch_df.lazy().sink_parquet(batch_file, mkdir=True)
    batch_ids = batch_df.get_column("id").n_unique()
    assert (
        batch_ids == expected_batch_ids
    ), f"Batch {batch_no} ID count mismatch: {batch_ids} != {expected_batch_ids}"
    took = f"{time.time() - t0:.1f}s"
    iterator.set_postfix_str(
        f"Saved {batch_ids} IDs as batch {batch_no+1}/{n_batches} in {took}"
    )


unpack_claims_parallel = unpack_claims
