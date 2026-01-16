# repro_leak_fast.py
import os
import tempfile
from pathlib import Path

import polars as pl
import psutil


def get_rss_gb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024**3


SOURCE_DIR = Path("data/huggingface_hub/philippesaade/wikidata/data")


def run_repro(n_files=50, rows_per_file=10):
    """Extract small slices from many real files to trigger schema heterogeneity."""
    from polars_genson import normalise_from_parquet

    files = sorted(SOURCE_DIR.glob("chunk_0-*.parquet"))[:n_files]

    print(f"Starting RSS: {get_rss_gb():.2f} GB")
    print(f"Processing {len(files)} files, {rows_per_file} rows each\n")

    for i, src_path in enumerate(files):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Extract small slice with real schema complexity
            input_path = Path(tmpdir) / "input.parquet"
            output_path = Path(tmpdir) / "output.parquet"

            # Read just first N rows, keep only claims column
            pl.scan_parquet(src_path).head(rows_per_file).select(
                "claims"
            ).collect().write_parquet(input_path)

            normalise_from_parquet(
                input_path=input_path,
                column="claims",
                output_path=output_path,
                output_column="claims",
                ndjson=True,
                wrap_root="claims",
                map_threshold=0,
                unify_maps=True,
                force_field_types={"mainsnak": "record"},
                force_scalar_promotion={"datavalue", "precision"},
                no_unify={"qualifiers"},
                max_builders=1000,
            )

        rss = get_rss_gb()
        print(f"File {i+1}: RSS = {rss:.2f} GB")


if __name__ == "__main__":
    run_repro()
