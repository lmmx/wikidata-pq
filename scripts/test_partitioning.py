#!/usr/bin/env python3
"""Test the partitioning module on a single file."""

from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl

from wikidata.config import PARTITION_COLS, Table
from wikidata.partitioning import partition_parquet, prepare_for_partition

TEST_FILE = "chunk_0-00400-of-00546.parquet"
RESULTS_DIR = Path("results")


def test_table(table: Table, tmp_dir: Path) -> None:
    table_file = RESULTS_DIR / table.value / TEST_FILE
    if not table_file.exists():
        print(f"  ⚠️  {table.value}: file not found at {table_file}")
        return

    print(f"\n{'='*60}")
    print(f"{table.value.upper()}")
    print("=" * 60)

    # Prepare
    print("Preparing...")
    lf = prepare_for_partition(table_file, table)
    schema = lf.collect_schema()
    partition_col = PARTITION_COLS[table]

    print(f"  Schema: {list(schema.keys())}")
    assert partition_col in schema, f"Missing partition column '{partition_col}'!"

    # Sample before partitioning
    sample = lf.head(5).collect()
    print(f"  Sample rows: {len(sample)}")
    print(sample)

    # Partition to temp dir
    dst_dir = tmp_dir / table.value
    log_dir = tmp_dir / "audit" / table.value

    print(f"\nPartitioning by '{partition_col}'...")
    partition_parquet(
        by=partition_col,
        lf=lf,
        source_name=TEST_FILE,
        dst_dir=dst_dir,
        log_dir=log_dir,
    )

    # Check what got written
    partitions = sorted(dst_dir.iterdir())
    print(f"  Created {len(partitions)} partitions")
    print(f"  First 5: {[p.name for p in partitions[:5]]}")

    # Verify a partition file
    sample_partition = partitions[0]
    sample_file = list(sample_partition.glob("*.parquet"))[0]
    partition_df = pl.read_parquet(sample_file)
    print(f"\n  Sample partition '{sample_partition.name}':")
    print(f"    Rows: {len(partition_df)}")
    print(f"    Columns: {partition_df.columns}")
    print(partition_df.head(3))

    # Check sidecar
    sidecar_file = log_dir / TEST_FILE
    if sidecar_file.exists():
        sidecar = pl.read_parquet(sidecar_file)
        print(f"\n  Sidecar ({len(sidecar)} entries):")
        print(sidecar.head(5))
    else:
        print(f"\n  ⚠️  Sidecar not found at {sidecar_file}")

    print(f"\n✓ {table.value} OK")


def main():
    with TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        print(f"Using temp dir: {tmp_dir}")

        for table in Table:
            test_table(table, tmp_dir)

    print("\n" + "=" * 60)
    print("All tables processed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()