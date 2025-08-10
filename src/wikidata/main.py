from pathlib import Path

from .initial import setup_state
from .process import process


def run():
    """Run the pipeline.

    Before we start the pipeline we initialise state in the `state` subdir, as JSONL.
    Initialised has a value of 0, and then there are 5 steps, and 6 means complete.

    1. Pull
    2. Process
    3. Partition
    4. Push
    5. Post-check
    """
    # 0. Initialise state
    if not (state_dir := Path("state")).exists():
        setup_state(state_dir)

    # 1. Pull files

    # 2. Process files
    # process(local_data_dir=Path("data"), output_dir=Path("results"))

    # 3. Partition subsets

    # 4. Push subsets

    # 5. Post-check uploaded subset file integrity

    # 6. Mark complete
