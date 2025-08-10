from pathlib import Path

from .config import LOCAL_DATA_DIR, OUTPUT_DIR, REPO_ID, STATE_DIR
from .initial import setup_state
from .process import process


def run(
    state_dir: Path = STATE_DIR,
    local_data_dir: Path = LOCAL_DATA_DIR,
    output_dir: Path = OUTPUT_DIR,
    repo_id: str = REPO_ID,
):
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
    if not state_dir.exists():
        setup_state(state_dir)

    # 1. Pull files
    return

    # 2. Process files
    process(local_data_dir=local_data_dir, output_dir=output_dir)

    # 3. Partition subsets

    # 4. Push subsets

    # 5. Post-check uploaded subset file integrity

    # 6. Mark complete
