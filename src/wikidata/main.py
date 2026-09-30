import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from sys import stderr

import polars as pl
from huggingface_hub import HfApi, snapshot_download

from .cards import push_card, write_cards
from .compact import compact_table
from .config import (
    AUDIT_DIR,
    CLEAN_UP_LOCAL,
    COMPACT_DOWNLOAD_WORKERS,
    HF_USER,
    HUB_COPY_DIR,
    OUTPUT_DIR,
    PARTITION_COLS,
    PREFETCH_BUDGET_GB,
    PREFETCH_CONCURRENCY,
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
from .partitioning import partition_parquet, prepare_for_partition
from .process import process
from .pull import prefetch_worker, pull_chunk
from .pull.prefetch import _expected_chunk_sizes
from .push import (
    Group,
    close_group,
    group_threshold_bytes,
    open_chunks,
    open_group_bytes,
    record_partitioned,
    unfinished_group,
)
from .sort_by_id import sort_table
from .state import (
    Step,
    get_file_step,
    get_next_chunk,
    update_state,
    validate_chunk_outputs,
)

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
    prefetch_concurrency: int = PREFETCH_CONCURRENCY,
):
    """Run the pipeline.

    Before we start the pipeline we initialise state in the `state` subdir, as JSONL.
    Initialised has a value of 0, and then there are 5 steps, and 6 means complete.

    1. Pull
    2. Process
    3. Partition
    4. Push (grouped: see DESIGN.md)
    5. Post-check

    Chunks are processed one at a time and uploaded in groups; the run ends when every
    chunk is complete. Re-running resumes from the chunk states and the group ledger.
    """
    target_repos = {tbl: REPO_TARGET.format(hf_user=hf_user, tbl=tbl) for tbl in Table}

    # 0. Initialise state
    if not state_dir.exists():
        setup_state(state_dir)

    source_sizes = _expected_chunk_sizes(repo_id).collect()
    total_source_bytes = source_sizes["size"].sum()

    # Finish a group interrupted mid-merge/upload/verify before starting new chunks
    if pending := unfinished_group(state_dir):
        group, stage = pending
        print(f"[push] Resuming {group.name} after stage {stage!r}")
        close_group(group, target_repos=target_repos, state_dir=state_dir, stage=stage)

    prefetch_future = None
    try:
        while (chunk_idx := get_next_chunk(state_dir, below=Step.PARTITION)) is not None:
            # 1. Pull
            pull_chunk(
                chunk_idx=chunk_idx,
                state_dir=state_dir,
                root_data_dir=data_dir,
                repo_id=repo_id,
            )
            # Only queue another prefetch pass once the last one has actually finished —
            # submitting one per chunk regardless left a growing backlog of stale, already-
            # redundant scans on the single-worker executor, starving real prefetch work.
            if prefetch_enabled and (prefetch_future is None or prefetch_future.done()):
                future = prefetch_executor.submit(
                    prefetch_worker,
                    chunk_idx,
                    state_dir,
                    data_dir,
                    repo_id,
                    budget_gb=prefetch_budget_gb,
                    max_ahead=prefetch_max_ahead,
                    min_free_gb=prefetch_min_free_gb,
                    concurrency=prefetch_concurrency,
                )
                future.add_done_callback(
                    lambda f: print(f"Prefetch error: {f.exception()}", file=stderr)
                    if f.exception()
                    else None
                )
                prefetch_future = future

            # 2-3. Process and partition, in a child process (see _run_chunk_isolated)
            _run_chunk_isolated(chunk_idx, data_dir, output_dir, repo_id, state_dir)
            chunk_bytes = source_sizes.filter(pl.col("chunk") == chunk_idx)["size"].sum()
            record_partitioned(state_dir, chunk_idx, chunk_bytes)

            # 4-5. Push and post-check, once the open group is big enough
            chunks = open_chunks(state_dir)
            size = open_group_bytes(state_dir, chunks)
            threshold = group_threshold_bytes(state_dir, total_source_bytes)
            print(
                f"[push] Open group: {len(chunks)} chunks, "
                f"{size / 1024**3:.2f} of {threshold / 1024**3:.2f} GB"
            )
            if size >= threshold:
                _close_open_group(chunks, target_repos, state_dir)

        # The remainder, once every chunk is partitioned
        if chunks := open_chunks(state_dir):
            _close_open_group(chunks, target_repos, state_dir)
        print("[run] All chunks complete.")
    finally:
        prefetch_executor.shutdown(wait=False, cancel_futures=True)

    finalise(state_dir=state_dir, hf_user=hf_user)


def finalise(state_dir: Path = STATE_DIR, hf_user: str = HF_USER) -> None:
    """Finalise the uploaded tables, once every chunk is complete: compact each table's
    repo (see compact.py), then sort it by id (see sort_by_id.py), writing its partition
    metadata, then render each table's dataset card and push it where it differs from the
    repo's (see cards.py). Resumes from the compaction and sort ledgers, and does nothing
    for a table already compacted and sorted."""
    if get_next_chunk(state_dir) is not None:
        raise RuntimeError("[finalise] Chunks are not all complete: run process-wikidata")
    if unfinished_group(state_dir):
        raise RuntimeError("[finalise] A group is not yet uploaded: run process-wikidata")
    for tbl in Table:
        compact_table(tbl, REPO_TARGET.format(hf_user=hf_user, tbl=tbl), state_dir)
    print("[finalise] All tables compacted.")
    for tbl in Table:
        sort_table(tbl, REPO_TARGET.format(hf_user=hf_user, tbl=tbl), state_dir)
    print("[finalise] All tables sorted.")
    api = HfApi()
    for tbl, card in write_cards().items():
        push_card(REPO_TARGET.format(hf_user=hf_user, tbl=tbl), card, api)
    print("[finalise] All dataset cards up to date.")


def render_cards() -> None:
    """Render every table's dataset card locally (see cards.py), without pushing."""
    write_cards()


def download(hub_dir: Path = HUB_COPY_DIR, hf_user: str = HF_USER) -> None:
    """Download a local copy of every table's Hub repo to `hub_dir/{table}` (resumable:
    files already there with the Hub's sha256 are kept)."""
    for tbl in Table:
        repo_id = REPO_TARGET.format(hf_user=hf_user, tbl=tbl)
        print(f"[download] {repo_id} to {hub_dir / tbl}", flush=True)
        snapshot_download(
            repo_id,
            repo_type="dataset",
            local_dir=hub_dir / tbl,
            max_workers=COMPACT_DOWNLOAD_WORKERS,
        )
    print("[download] All tables downloaded.")


def process_and_partition(
    chunk_idx: int, data_dir: Path, output_dir: Path, repo_id: str, state_dir: Path
) -> None:
    process(
        data_dir=data_dir,
        output_dir=output_dir,
        repo_id=repo_id,
        state_dir=state_dir,
        chunk_idx=chunk_idx,
    )
    partition_chunk(chunk_idx, state_dir, output_dir)


def _run_chunk_isolated(
    chunk_idx: int, data_dir: Path, output_dir: Path, repo_id: str, state_dir: Path
) -> None:
    """Process and partition one chunk in a fresh interpreter, so all the memory it used
    goes back to the OS when it exits (freed native memory otherwise stays in the
    allocator and RSS climbs chunk after chunk). Spawned, not forked: the parent has the
    prefetch thread running. Progress is in the state files, so the parent reads it from
    there; a failure in the child halts the run.
    """
    ctx = multiprocessing.get_context("spawn")
    child = ctx.Process(
        target=process_and_partition,
        args=(chunk_idx, data_dir, output_dir, repo_id, state_dir),
        name=f"chunk_{chunk_idx}",
    )
    child.start()
    child.join()
    if child.exitcode != 0:
        cause = (
            f"killed by signal {-child.exitcode}"
            if child.exitcode < 0
            else f"exit code {child.exitcode}"
        )
        raise RuntimeError(f"[run] Chunk {chunk_idx} failed in its subprocess ({cause})")


def _close_open_group(
    chunks: list[int], target_repos: dict[Table, str], state_dir: Path
) -> None:
    group = Group(chunks[0], chunks[-1])
    assert list(group.chunks) == chunks, f"Open chunks are not contiguous: {chunks}"
    close_group(group, target_repos=target_repos, state_dir=state_dir)


def partition_chunk(chunk_idx: int, state_dir: Path, output_dir: Path) -> None:
    """Partition each table of a processed chunk by language (links by site)."""
    expected_files, missing = validate_chunk_outputs(
        chunk_idx, state_dir, output_dir, [tbl.value for tbl in Table]
    )
    if missing:
        raise RuntimeError(f"[partition] Halting - missing processed files: {missing}")

    for filename in expected_files:
        if (get_file_step(filename, state_dir) or Step.INIT) >= Step.PARTITION:
            continue

        for tbl in Table:
            table_output_dir = output_dir / tbl
            lf = prepare_for_partition(table_output_dir / filename, tbl)
            partition_parquet(
                by=PARTITION_COLS[tbl],
                lf=lf,
                source_name=filename,
                dst_dir=table_output_dir,
                log_dir=AUDIT_DIR / tbl,
            )

        update_state(Path(filename), Step.PARTITION, state_dir)
        if CLEAN_UP_LOCAL:
            for tbl in Table:
                (output_dir / tbl / filename).unlink()
            print(f"[partition] Deleted processed tables for {filename}")
