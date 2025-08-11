from pathlib import Path

from .config import (
    HF_USER,
    LOCAL_DATA_DIR,
    OUTPUT_DIR,
    REPO_ID,
    REPO_TARGET,
    STATE_DIR,
    Table,
)
from .initial import setup_state
from .process import process
from .pull import pull_chunk
from .state import get_next_chunk


def run(
    state_dir: Path = STATE_DIR,
    local_data_dir: Path = LOCAL_DATA_DIR,
    output_dir: Path = OUTPUT_DIR,
    repo_id: str = REPO_ID,
    hf_user: str = HF_USER,
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
    target_repos = {tbl: REPO_TARGET.format(hf_user=hf_user, tbl=tbl) for tbl in Table}

    # 0. Initialise state
    if not state_dir.exists():
        setup_state(state_dir)

    # 1. Pull files
    while (chunk_idx := get_next_chunk(state_dir)) is not None:
        pull_chunk(
            chunk_idx=chunk_idx,
            state_dir=state_dir,
            local_data_dir=local_data_dir,
            repo_id=repo_id,
            target_repos=target_repos,
        )
        break

    print("We made it!")
    return

    # 2. Process files
    process(local_data_dir=local_data_dir, output_dir=output_dir)

    # 3. Partition subsets

    # 4. Push subsets

    # 5. Post-check uploaded subset file integrity

    # 6. Mark complete
