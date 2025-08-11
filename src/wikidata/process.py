import re
import shutil
from pathlib import Path

import polars as pl
from huggingface_hub import HfFileSystem

from .claims import unpack_claims
from .config import CAPTURE_GROUP_RE, CHUNK_RE, REMOTE_REPO_PATH, Table
from .pull import _hf_dl_subdir
from .struct_transforms import unpivot_from_list_struct_col, unpivot_from_struct_col

CLEAN_UP_TMP = False
repo_id = "philippesaade/wikidata"
hf_fs = HfFileSystem()


def n_ids(fr: pl.DataFrame) -> int:
    """Count the unique IDs (we expect them *all* to be preserved)."""
    return fr.get_column("id").n_unique()


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
        total = n_ids(df)

        def tbl_pq(tbl: Table) -> Path:
            return output_dir / tbl / pq_path.name

        label_pq, desc_pq, alias_pq, link_pq, claim_pq = map(tbl_pq, Table)

        # Process labels
        if label_pq.exists():
            labels = pl.read_parquet(label_pq)
        else:
            labels = unpivot_from_struct_col(df, "labels", "value", "language")
            labels.lazy().sink_parquet(label_pq, mkdir=True)
        assert total == n_ids(labels), f"ID loss: {total} --> {n_ids(labels)=}"

        # Process descriptions
        if desc_pq.exists():
            descs = pl.read_parquet(desc_pq)
        else:
            descs = unpivot_from_struct_col(df, "descriptions", "value", "language")
            descs.lazy().sink_parquet(desc_pq, mkdir=True)
        assert total == n_ids(descs), f"ID loss: {total} --> {n_ids(descs)=}"

        # Process aliases
        if alias_pq.exists():
            aliases = pl.read_parquet(alias_pq)
        else:
            aliases = unpivot_from_list_struct_col(df, "aliases", "value", "language")
            aliases.lazy().sink_parquet(alias_pq, mkdir=True)
        # Aliases have known nulls ~10% so drop them deliberately, no point keeping:
        # assert total == n_ids(aliases), f"ID loss: {total} --> {n_ids(aliases)=}"

        # Process links
        if link_pq.exists():
            links = pl.read_parquet(link_pq)
        else:
            links = unpivot_from_struct_col(df, "sitelinks", "title", "site")
            links.lazy().sink_parquet(link_pq, mkdir=True)
        assert total == n_ids(links), f"ID loss: {total} --> {n_ids(links)=}"

        # Claims are complex nested JSON. Dump them to disk as we go to resume easily
        tmp_batch_store = tmp_dir / pq_path.stem
        if claim_pq.exists():
            claims = pl.read_parquet(claim_pq)
        else:
            # Claims get very large so cache intermediate parquets to
            # "data/tmp/chunk_000-of-n/" dir, as files named "batch-1-of-5.parquet" etc
            claims = unpack_claims(df, tmp_batch_store)
            claims.lazy().sink_parquet(claim_pq, mkdir=True)
        if CLEAN_UP_TMP and tmp_batch_store.exists():
            shutil.rmtree(tmp_batch_store)
            print(f"Cleaned up {tmp_batch_store}")
        assert total == n_ids(claims), f"ID loss: {total} --> {n_ids(claims)=}"

        # Remove the break statement to process all files
        break

    print("Processing complete!")
