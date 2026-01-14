import re
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl
from deepdiff import DeepDiff
from huggingface_hub import HfFileSystem
from polars_genson import (
    avro_to_polars_schema,
    normalise_from_parquet,
    read_parquet_metadata,
    schema_to_dict,
)

from .claims import unpack_claims
from .config import CAPTURE_GROUP_RE, CHUNK_RE, REMOTE_REPO_PATH, Table
from .pull import _hf_dl_subdir
from .struct_transforms import unpivot_from_list_struct_col, unpivot_from_struct_col

CLEAN_UP_TMP = False
repo_id = "philippesaade/wikidata"
hf_fs = HfFileSystem()


RECORD_SCHEMA = pl.Struct({"language": pl.String, "value": pl.String})
# Either Map<Record> or Map<List<Record>>
MAP_REC_SCHEMA = pl.Struct({"key": pl.String, "value": RECORD_SCHEMA})
MAP_LOR_SCHEMA = pl.Struct({"key": pl.String, "value": pl.List(RECORD_SCHEMA)})

SITELINK_SCHEMA = pl.Struct({"site": pl.String, "title": pl.String})
MAP_SITELINK_SCHEMA = pl.Struct({"key": pl.String, "value": SITELINK_SCHEMA})


def normalise_map(df: pl.DataFrame, *, key: str, lor: bool = False) -> pl.DataFrame:
    """Normalise JSON Map of language codes (e.g. 'en') to {language,value} Records."""
    maps = pl.Struct({key: pl.List(MAP_LOR_SCHEMA if lor else MAP_REC_SCHEMA)})
    return df.genson.normalise_json(
        key, ndjson=True, wrap_root=key, decode=maps, max_builders=100
    )


def load_normalised_map(path: Path, *, key: str, lor: bool = False) -> pl.DataFrame:
    """Normalise JSON Map of language codes (e.g. 'en') to {language,value} Records."""
    maps = pl.Struct({key: pl.List(MAP_LOR_SCHEMA if lor else MAP_REC_SCHEMA)})
    decoded_json = pl.col(key).str.json_decode(dtype=maps)
    return pl.read_parquet(path).select(decoded_json).unnest(key)


def normalise_map_direct(
    input_path: Path,
    output_path: Path,
    *,
    key: str,
    lor: bool = False,
) -> pl.DataFrame:
    """Normalise and cache JSON map, reading from cached Parquet if it exists."""
    with TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / output_path.name
        normalise_from_parquet(
            input_path=input_path,
            column=key,
            output_path=tmp_path,
            output_column=key,
            ndjson=True,
            wrap_root=key,
        )
        result = load_normalised_map(path=tmp_path, key=key, lor=lor)
    return result


def normalise_sitelinks(df: pl.DataFrame) -> pl.DataFrame:
    """Normalise JSON Map of site codes (e.g. 'enwiki') to {site,title} Records."""
    maps = pl.Struct({"sitelinks": pl.List(MAP_SITELINK_SCHEMA)})
    return df.genson.normalise_json(
        "sitelinks", ndjson=True, wrap_root="sitelinks", decode=maps, max_builders=100
    )


def n_ids(fr: pl.DataFrame) -> int:
    """Count the unique IDs (we expect them *all* to be preserved)."""
    return fr.get_column("id").n_unique()


KV_SCHEMA = pl.List(pl.Struct({"key": pl.String, "value": pl.String}))
DV_SCHEMA = pl.Struct(
    {
        "id": pl.String,
        "labels": KV_SCHEMA,
        "datavalue__string": pl.String,
        "precision": pl.Struct(
            {
                "precision__integer": pl.Int64,
                "precision__number": pl.Float64,
            }
        ),
        "text": pl.String,
        "language": pl.String,
        "amount": pl.String,
        "unit": pl.String,
        "unit-labels": KV_SCHEMA,
        "upperBound": pl.String,
        "lowerBound": pl.String,
        "time": pl.String,
        "timezone": pl.Int64,
        "before": pl.Int64,
        "after": pl.Int64,
        "calendarmodel": pl.String,
        "latitude": pl.Float64,
        "longitude": pl.Float64,
        "altitude": pl.Null,
        "globe": pl.String,
    }
)
MAINSNAK_SCHEMA = pl.Struct(
    {
        "property": pl.String,
        "datavalue": DV_SCHEMA,
        "datatype": pl.String,
        "property-labels": KV_SCHEMA,
    }
)
QUALS_SCHEMA = pl.Struct({"key": pl.String, "value": pl.List(MAINSNAK_SCHEMA)})
REFS_SCHEMA = pl.List(QUALS_SCHEMA)
claims_schema = pl.Schema(
    pl.Struct(
        {
            "claims": pl.List(
                pl.Struct(
                    {
                        "key": pl.String,
                        "value": pl.List(
                            pl.Struct(
                                {
                                    "mainsnak": MAINSNAK_SCHEMA,
                                    "rank": pl.String,
                                    "references": pl.List(REFS_SCHEMA),
                                    "qualifiers": pl.List(QUALS_SCHEMA),
                                }
                            )
                        ),
                    }
                )
            )
        }
    )
)


def normalise_claims_direct(
    input_path: Path,
    output_path: Path,
    *,
    key: str = "claims",
    schema: pl.DataType | None = None,
) -> pl.DataFrame:
    """Normalise complex nested JSON claims, decode, and cache to Parquet."""
    inference_options = dict(
        ndjson=True,
        map_threshold=0,
        unify_maps=True,
        force_field_types={"mainsnak": "record"},
        force_scalar_promotion={"datavalue", "precision"},
        no_unify={"qualifiers"},
    )
    with TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / output_path.name
        normalise_from_parquet(
            input_path=input_path,
            column=key,
            output_path=tmp_path,
            output_column=key,
            wrap_root=key,
            **inference_options,
            profile=True,
            max_builders=1000,
        )
        result = pl.read_parquet(tmp_path)
        if schema is None:
            metadata = read_parquet_metadata(tmp_path)
            avro_schema_json = metadata["genson_avro_schema"]
            full_schema = avro_to_polars_schema(avro_schema_json)
            schema = pl.Struct(full_schema)
        result = result.select(pl.col(key).str.json_decode(dtype=schema)).unnest(key)
    return result


def process(
    data_dir: Path, output_dir: Path, repo_id: str, chunk_idx: int | None = None
):
    """Flatten the source files from `data_dir` and store in `output_dir`.

    Args:
        data_dir: Root directory for HuggingFace dataset cache and other source files.
                  A temporary directory will be created directly beneath this path.
        output_dir: Destination directory where the 5 separate tables will be written,
                    each stored in a subdirectory named after the `Table` enum value.
        repo_id: HuggingFace dataset repository ID in the format 'user/dataset'.
        chunk_idx: If set, only process files in the specific chunk.
    """
    tmp_dir = data_dir / "tmp"
    ds_dir = _hf_dl_subdir(data_dir, repo_id=repo_id)
    assert ds_dir.exists(), f"Dataset source directory doesn't exist: {ds_dir!s}"

    # If `chunk_idx` set match f"chunk_{chunk_idx}.parquet" else any "chunk_*-*.parquet"
    chunk_pattern = re.sub(
        CAPTURE_GROUP_RE, "*" if chunk_idx is None else str(chunk_idx), CHUNK_RE
    )
    hf_local_mirror_subpath = f"{REMOTE_REPO_PATH}/{chunk_pattern}*.parquet"

    for pq_path in sorted(ds_dir.glob(hf_local_mirror_subpath)):
        print(f"Processing {pq_path.name}")
        df = pl.read_parquet(pq_path)
        # total = n_ids(df)

        def tbl_pq(tbl: Table) -> Path:
            return output_dir / tbl / pq_path.name

        label_pq, desc_pq, alias_pq, link_pq, claim_pq = map(tbl_pq, Table)

        # Process labels
        if label_pq.exists():
            labels = pl.read_parquet(label_pq)
        else:
            labels = normalise_map_direct(pq_path, label_pq, key="labels")
            labels.lazy().sink_parquet(label_pq, mkdir=True)
        # assert total == n_ids(labels), f"ID loss: {total} --> {n_ids(labels)=}"

        # Process descriptions
        if desc_pq.exists():
            descs = pl.read_parquet(desc_pq)
        else:
            descs = normalise_map_direct(pq_path, desc_pq, key="descriptions")
            descs.lazy().sink_parquet(desc_pq, mkdir=True)
        # assert total == n_ids(descs), f"ID loss: {total} --> {n_ids(descs)=}"

        # Process aliases
        if alias_pq.exists():
            aliases = pl.read_parquet(alias_pq)
        else:
            aliases = normalise_map_direct(pq_path, alias_pq, key="aliases", lor=True)
            aliases.lazy().sink_parquet(alias_pq, mkdir=True)
        # Aliases have known nulls ~10% so drop them deliberately, no point keeping:
        # assert total == n_ids(aliases), f"ID loss: {total} --> {n_ids(aliases)=}"

        # Process links
        if link_pq.exists():
            links = pl.read_parquet(link_pq)
        else:
            links = normalise_sitelinks(df)
            links.lazy().sink_parquet(link_pq, mkdir=True)
        # assert total == n_ids(links), f"ID loss: {total} --> {n_ids(links)=}"

        # Claims are complex nested JSON. Dump them to disk as we go to resume easily
        tmp_batch_store = tmp_dir / pq_path.stem
        if claim_pq.exists():
            claims = pl.scan_parquet(claim_pq)
        else:
            # Claims get very large so cache intermediate parquets to
            # "data/tmp/chunk_000-of-n/" dir, as files named "batch-1-of-5.parquet" etc
            cn = claim_pq.name
            cn_idx = int(cn.split("-")[1])
            # if cn_idx < 27:
            #     continue
            if cn_idx > 544:
                raise SystemExit("Gone too far, halting")
            claims = normalise_claims_direct(pq_path, claim_pq)
            inferred_claims_schema = claims.collect_schema()
            # Check if schema is equivalent [under permutation] to one we have stored
            d1 = schema_to_dict(claims_schema)
            d2 = schema_to_dict(inferred_claims_schema)
            if d1 != d2:
                diff = DeepDiff(d1, d2, ignore_order=True)
                # It's fine if either the diff is empty, or the schema is a subset of
                # the one we have stored
                if diff and list(diff) != ["dictionary_item_removed"]:
                    breakpoint()
            claims.lazy().sink_parquet(claim_pq, mkdir=True)
            # claims = df[["claims"]].genson.normalise_json(
            #     "claims",
            #     ndjson=True,
            #     wrap_root="claims",
            #     map_threshold=0,
            #     unify_maps=True,
            #     force_field_types={"mainsnak": "record"},
            #     no_unify={"qualifiers"},
            #     decode=True,
            #     profile=True,
            #     max_builders=1,
            # )
            # if pl.Struct(claims.collect_schema()) != claims_schema:
            #     print("-> Saving schema")
            #     with open("schemas.txt", "a") as f:
            #         f.write(cn + "\n")
            #         f.write(str(claims.dtypes) + "\n")
            # claims = unpack_claims(df, tmp_batch_store)
            # claims.sink_parquet(claim_pq, mkdir=True)
        if CLEAN_UP_TMP and tmp_batch_store.exists():
            shutil.rmtree(tmp_batch_store)
            print(f"Cleaned up {tmp_batch_store}")
        # assert total == n_ids(
        #     claims.collect()
        # ), f"ID loss: {total} --> {n_ids(claims.collect())=}"

    print("Processing complete!")
