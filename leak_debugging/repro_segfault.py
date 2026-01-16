# repro_leak_sweep.py
import os
import tempfile
from pathlib import Path

import polars as pl
import psutil
from polars_genson import normalise_from_parquet


def get_rss_gb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024**3


SOURCE_DIR = Path("data/huggingface_hub/philippesaade/wikidata/data")
OUTPUT_DIR = Path("repro_results")
OUTPUT_DIR.mkdir(exist_ok=True)


def run_sweep(n_files=50, rows_per_file=10):
    """Run one sweep and return RSS measurements."""
    files = sorted(SOURCE_DIR.glob("chunk_0-*.parquet"))[:n_files]

    measurements = []
    print(f"\n=== rows_per_file={rows_per_file} ===")
    print(f"Starting RSS: {get_rss_gb():.2f} GB")

    for i, src_path in enumerate(files):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.parquet"
            output_path = Path(tmpdir) / "output.parquet"

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
        measurements.append({"file_idx": i + 1, "rss_gb": rss})
        print(f"File {i+1}: RSS = {rss:.2f} GB")

    return pl.DataFrame(measurements).with_columns(
        pl.lit(rows_per_file).alias("rows_per_file")
    )


def main():
    # Sweep schedule: 10,15,20,...,100
    # row_counts = list(range(10, 101, 5))
    row_counts = [20]

    all_results = []

    for rows in row_counts:
        # result = run_sweep(n_files=50, rows_per_file=rows)
        result = run_sweep(n_files=5, rows_per_file=rows)
        result.write_parquet(OUTPUT_DIR / f"sweep_rows_{rows:03d}.parquet")
        all_results.append(result)

        # Also save combined after each sweep (in case you Ctrl+C)
        combined = pl.concat(all_results)
        combined.write_parquet(OUTPUT_DIR / "combined_sweep.parquet")
        print(f"Saved sweep for rows={rows}, combined has {len(combined)} rows")


if __name__ == "__main__":
    main()
