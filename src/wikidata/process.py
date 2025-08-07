from pathlib import Path

import orjson
import polars as pl
from huggingface_hub import HfFileSystem
from tqdm import tqdm

from .claims import unpack_claims
from .schemas import str_snak_values, total_schema
from .struct_transforms import unpivot_from_list_struct_col, unpivot_from_struct_col

DEBUG_ON_EXC = True


def main():
    hf_fs = HfFileSystem()
    repo_id = "philippesaade/wikidata"
    repo_subpath = "data/*.parquet"
    # data_files = map("hf://{}".format, hf_fs.glob(f"datasets/{repo_id}/{repo_subpath}"))

    # It is better if we pull the data before we need it, but also we don't want to wait...
    data_dir = Path("data")
    tmp_dir = data_dir / "tmp"

    for pq_path in data_dir.glob("*.parquet"):
        print(f"Processing {pq_path.name}")
        df = pl.read_parquet(repo_subpath)

        labels = unpivot_from_struct_col(df, "labels", "value", "language")
        descs = unpivot_from_struct_col(df, "descriptions", "value", "language")
        aliases = unpivot_from_list_struct_col(df, "aliases", "value", "language")
        links = unpivot_from_struct_col(df, "sitelinks", "title", "site")
        # Claims get very large so cache intermediate parquets to
        # "data/tmp/chunk_000-of-n/" dir, as files named "batch-1-of-5.parquet" etc
        claims_df = unpack_claims(df, temp_store_path=tmp_dir / pq_path.stem)
