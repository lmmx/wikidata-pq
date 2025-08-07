import shutil
from pathlib import Path

import orjson
import polars as pl
from huggingface_hub import HfFileSystem
from tqdm import tqdm

from .claims import unpack_claims
from .schemas import str_snak_values, total_schema
from .struct_transforms import unpivot_from_list_struct_col, unpivot_from_struct_col

DEBUG_ON_EXC = True
CLEAN_UP_TMP = True


def main(local_data_dir: Path = Path("data"), output_dir: Path = Path("results")):
    """Save files in output_dir"""
    assert local_data_dir.exists(), f"Input directory doesn't exist: {local_data_dir!s}"
    output_dir.mkdir(exist_ok=True)
    tmp_dir = local_data_dir / "tmp"

    hf_fs = HfFileSystem()
    repo_id = "philippesaade/wikidata"
    # repo_subpath = "data/*.parquet"
    # data_files = map("hf://{}".format, hf_fs.glob(f"datasets/{repo_id}/{repo_subpath}"))

    # It is better if we pull the data before we need it, but also we don't want to wait...

    for pq_path in sorted(local_data_dir.glob("*.parquet")):
        print(f"Processing {pq_path.name}")
        df = pl.read_parquet(pq_path)

        # These are all trivial to unpack
        labels = unpivot_from_struct_col(df, "labels", "value", "language")
        descs = unpivot_from_struct_col(df, "descriptions", "value", "language")
        aliases = unpivot_from_list_struct_col(df, "aliases", "value", "language")
        links = unpivot_from_struct_col(df, "sitelinks", "title", "site")

        # Claims are complex nested JSON. Dump them to disk as we go to resume easily
        claims_pq = output_dir / pq_path.name
        tmp_batch_store = tmp_dir / pq_path.stem
        if claims_pq.exists():
            claims_df = pl.read_parquet(claims_pq)
        else:
            # Claims get very large so cache intermediate parquets to
            # "data/tmp/chunk_000-of-n/" dir, as files named "batch-1-of-5.parquet" etc
            claims_df = unpack_claims(df, tmp_batch_store)
            claims_df.write_parquet(claims_pq)
        if CLEAN_UP_TMP and tmp_batch_store.exists():
            shutil.rmtree(tmp_batch_store)
            print(f"Cleaned up {tmp_batch_store}")
        break

    breakpoint()
