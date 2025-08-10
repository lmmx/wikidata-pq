from pathlib import Path

from .process import process


def run():
    """Run the pipeline.

    1. Pull
    2. Process
    3. Partition
    4. Push
    5. Post-check
    """
    process(local_data_dir=Path("data"), output_dir=Path("results"))
