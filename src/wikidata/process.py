import shutil
from pathlib import Path

import orjson
import polars as pl
from huggingface_hub import HfFileSystem
from tqdm import tqdm

from .claims import unpack_claims
from .schemas import str_snak_values, total_schema
from .struct_transforms import unpivot_from_list_struct_col, unpivot_from_struct_col

CLEAN_UP_TMP = False
repo_id = "philippesaade/wikidata"
hf_fs = HfFileSystem()


def n_ids(fr: pl.DataFrame) -> int:
    """Count the unique IDs (we expect them *all* to be preserved)."""
    return fr.get_column("id").n_unique()


def main(local_data_dir: Path = Path("data"), output_dir: Path = Path("results")):
    """Save files in output_dir"""
    assert local_data_dir.exists(), f"Input directory doesn't exist: {local_data_dir!s}"
    output_dir.mkdir(exist_ok=True)

    labels_dir = output_dir / "labels"
    descs_dir = output_dir / "descs"
    aliases_dir = output_dir / "aliases"
    links_dir = output_dir / "links"
    claims_dir = output_dir / "claims"

    tmp_dir = local_data_dir / "tmp"

    for _dir in [labels_dir, descs_dir, aliases_dir, links_dir, claims_dir, tmp_dir]:
        _dir.mkdir(exist_ok=True)

    # repo_subpath = "data/*.parquet"
    # data_files = map("hf://{}".format, hf_fs.glob(f"datasets/{repo_id}/{repo_subpath}"))

    # It is better if we pull the data before we need it, but also we don't want to wait...

    for pq_path in sorted(local_data_dir.glob("*.parquet")):
        print(f"Processing {pq_path.name}")
        df = pl.read_parquet(pq_path)

        # Process labels
        labels_pq = labels_dir / pq_path.name
        if labels_pq.exists():
            labels = pl.read_parquet(labels_pq)
        else:
            labels = unpivot_from_struct_col(df, "labels", "value", "language")
            labels.write_parquet(labels_pq)
        assert n_ids(df) == n_ids(labels), f"ID loss: {n_ids(df)}-->{n_ids(labels)=}"

        # Process descriptions
        descs_pq = descs_dir / pq_path.name
        if descs_pq.exists():
            descs = pl.read_parquet(descs_pq)
        else:
            descs = unpivot_from_struct_col(df, "descriptions", "value", "language")
            descs.write_parquet(descs_pq)
        assert n_ids(df) == n_ids(descs), f"ID loss: {n_ids(df)}-->{n_ids(descs)=}"

        # Process aliases
        aliases_pq = aliases_dir / pq_path.name
        if aliases_pq.exists():
            aliases = pl.read_parquet(aliases_pq)
        else:
            aliases = unpivot_from_list_struct_col(df, "aliases", "value", "language")
            aliases.write_parquet(aliases_pq)
        # Aliases have known nulls ~10% so drop them deliberately, no point keeping:
        # assert n_ids(df) == n_ids(aliases), f"ID loss: {n_ids(df)}-->{n_ids(aliases)=}"

        # Process links
        links_pq = links_dir / pq_path.name
        if links_pq.exists():
            links = pl.read_parquet(links_pq)
        else:
            links = unpivot_from_struct_col(df, "sitelinks", "title", "site")
            links.write_parquet(links_pq)
        assert n_ids(df) == n_ids(links), f"ID loss: {n_ids(df)}-->{n_ids(links)=}"

        # Claims are complex nested JSON. Dump them to disk as we go to resume easily
        claims_pq = claims_dir / pq_path.name
        tmp_batch_store = tmp_dir / pq_path.stem
        if claims_pq.exists():
            claims = pl.read_parquet(claims_pq)
        else:
            # Claims get very large so cache intermediate parquets to
            # "data/tmp/chunk_000-of-n/" dir, as files named "batch-1-of-5.parquet" etc
            claims = unpack_claims(df, tmp_batch_store)
            claims.write_parquet(claims_pq)
        if CLEAN_UP_TMP and tmp_batch_store.exists():
            shutil.rmtree(tmp_batch_store)
            print(f"Cleaned up {tmp_batch_store}")
        assert n_ids(df) == n_ids(claims), f"ID loss: {n_ids(df)}-->{n_ids(claims)=}"

        # Remove the break statement to process all files
        break

    print("Processing complete!")
