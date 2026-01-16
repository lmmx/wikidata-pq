# repro_full_visibility.py
"""Full visibility repro - tracks RSS, VMS, and system memory."""

import os
import sys
import tempfile
from pathlib import Path

import polars as pl
import psutil
from polars_genson import normalise_from_parquet


def get_memory_info():
    proc = psutil.Process(os.getpid())
    mem = proc.memory_info()
    sys_mem = psutil.virtual_memory()
    return {
        "rss_gb": mem.rss / 1024**3,
        "vms_gb": mem.vms / 1024**3,
        "sys_used_gb": sys_mem.used / 1024**3,
        "sys_avail_gb": sys_mem.available / 1024**3,
        "sys_percent": sys_mem.percent,
    }


SOURCE_DIR = Path("data/huggingface_hub/philippesaade/wikidata/data")
OUTPUT_DIR = Path("repro_results")
OUTPUT_DIR.mkdir(exist_ok=True)


def run_sweep(n_files=50, rows_per_file=10):
    files = sorted(SOURCE_DIR.glob("chunk_0-*.parquet"))[:n_files]

    measurements = []
    print(f"\n=== rows_per_file={rows_per_file} ===")
    mem = get_memory_info()
    print(
        f"Start: RSS={mem['rss_gb']:.2f}GB VMS={mem['vms_gb']:.2f}GB SysUsed={mem['sys_used_gb']:.1f}GB Avail={mem['sys_avail_gb']:.1f}GB ({mem['sys_percent']:.0f}%)"
    )

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

        mem = get_memory_info()
        measurements.append({"file_idx": i + 1, "rows_per_file": rows_per_file, **mem})
        print(
            f"File {i+1}: RSS={mem['rss_gb']:.2f}GB VMS={mem['vms_gb']:.2f}GB Avail={mem['sys_avail_gb']:.1f}GB ({mem['sys_percent']:.0f}%)"
        )
        sys.stdout.flush()

    return pl.DataFrame(measurements)


def main():
    row_counts = list(range(70, 101, 10))

    all_results = []

    for rows in row_counts:
        result = run_sweep(n_files=50, rows_per_file=rows)
        result.write_parquet(OUTPUT_DIR / f"sweep_rows_{rows:03d}.parquet")
        all_results.append(result)

        combined = pl.concat(all_results)
        combined.write_parquet(OUTPUT_DIR / "combined_sweep.parquet")
        print(f"Saved sweep for rows={rows}, combined has {len(combined)} rows")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
