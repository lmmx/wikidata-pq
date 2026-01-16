from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from sys import stderr

from .config import (
    AUDIT_DIR,
    HF_USER,
    OUTPUT_DIR,
    PARTITION_COLS,
    PREFETCH_BUDGET_GB,
    PREFETCH_ENABLED,
    PREFETCH_MAX_AHEAD,
    PREFETCH_MIN_FREE_GB,
    REPO_ID,
    REPO_TARGET,
    ROOT_DATA_DIR,
    STATE_DIR,
    Table,
)
from .initial import setup_state
from .partitioning import partition_parquet
from .process import process
from .pull import prefetch_worker, pull_chunk
from .state import get_next_chunk

# Create thread pool executor for prefetching (single worker to avoid resource contention)
prefetch_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="prefetch")


def run(
    state_dir: Path = STATE_DIR,
    data_dir: Path = ROOT_DATA_DIR,
    output_dir: Path = OUTPUT_DIR,
    repo_id: str = REPO_ID,
    hf_user: str = HF_USER,
    prefetch_enabled: bool = PREFETCH_ENABLED,
    prefetch_budget_gb: float = PREFETCH_BUDGET_GB,
    prefetch_max_ahead: int = PREFETCH_MAX_AHEAD,
    prefetch_min_free_gb: float = PREFETCH_MIN_FREE_GB,
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
            root_data_dir=data_dir,
            repo_id=repo_id,
            target_repos=target_repos,
        )

        if prefetch_enabled:
            future = prefetch_executor.submit(
                prefetch_worker,
                chunk_idx,
                state_dir,
                data_dir,
                repo_id,
                target_repos,
                budget_gb=prefetch_budget_gb,
                max_ahead=prefetch_max_ahead,
                min_free_gb=prefetch_min_free_gb,
            )
            future.add_done_callback(
                lambda f: print(f"Prefetch error: {f.exception()}", file=stderr)
                if f.exception()
                else None
            )

        # 2. Process files
        process(data_dir=data_dir, output_dir=output_dir, repo_id=repo_id)

        # 3. Partition subsets
        for tbl in Table:
            table_output_dir = output_dir / tbl
            audit_log_dir = AUDIT_DIR / tbl

            for processed_file in table_output_dir.glob(f"chunk_{chunk_idx}-*.parquet"):
                partition_parquet(
                    by=PARTITION_COLS[tbl],
                    source_pq=processed_file,
                    dst_dir=table_output_dir,
                    log_dir=audit_log_dir,
                )

        # 4. Push subsets

        # 5. Post-check uploaded subset file integrity

        # 6. Mark complete
