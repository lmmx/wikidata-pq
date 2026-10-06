import shutil
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from sys import stderr

import polars as pl
from huggingface_hub import HfApi, snapshot_download

from . import dump
from .card_stats import update_stats
from .claims_labels import build_claims_labels, collect_refs_stage
from .cards import push_card, write_cards
from .compact import compact_table
from .config import (
    HUB_REVISION,
    AUDIT_DIR,
    CHUNK_WORKERS,
    CLEAN_UP_LOCAL,
    COMPACT_DOWNLOAD_WORKERS,
    HF_USER,
    HUB_COPY_DIR,
    OTHER_WORK_DIR,
    OUTPUT_DIR,
    PARTITION_COLS,
    PREFETCH_BUDGET_GB,
    PREFETCH_CONCURRENCY,
    PREFETCH_ENABLED,
    PREFETCH_MAX_AHEAD,
    PREFETCH_MIN_FREE_GB,
    RELEASE,
    REPO_ID,
    REPO_TARGET,
    ROOT_DATA_DIR,
    SCHOLAR,
    STATE_DIR,
    Table,
)
from .initial import setup_state
from .partitioning import partition_parquet, prepare_for_partition
from .pool import process_chunks
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
from .sort_by_id import last_stage as sort_stage
from .sort_by_id import sort_table
from .state import (
    Step,
    get_all_state,
    get_file_step,
    get_next_chunk,
    last_chunk,
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
    workers: int = CHUNK_WORKERS,
):
    """Run the pipeline.

    Before we start the pipeline we initialise state in the `state` subdir, as JSONL.
    Initialised has a value of 0, and then there are 5 steps, and 6 means complete.

    1. Pull
    2. Process
    3. Partition
    4. Push (grouped: see DESIGN.md)
    5. Post-check

    Chunks are processed `workers` at a time and uploaded in groups; the run ends when every
    chunk is complete. Re-running resumes from the chunk states and the group ledger.
    """
    target_repos = {tbl: REPO_TARGET.format(hf_user=hf_user, tbl=tbl) for tbl in Table}

    # 0. Initialise state
    if not state_dir.exists():
        setup_state(state_dir)

    # A release's chunks are split from its dump (see dump.py): local already, so no
    # download or prefetch, and their sizes from the split's manifest
    if RELEASE:
        prefetch_enabled = False
    source_sizes = (dump.chunk_sizes() if RELEASE else _expected_chunk_sizes(repo_id)).collect()
    total_source_bytes = source_sizes["size"].sum()

    # Finish a group interrupted mid-merge/upload/verify before starting new chunks
    if pending := unfinished_group(state_dir):
        group, stage = pending
        print(f"[push] Resuming {group.name} after stage {stage!r}")
        close_group(group, target_repos=target_repos, state_dir=state_dir, stage=stage)

    prefetch_future = None

    # 1. Pull, in the parent before each chunk's process
    def start(chunk_idx: int) -> None:
        nonlocal prefetch_future
        if RELEASE:
            dump.check_chunk(chunk_idx, state_dir)
            return
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

    def done(chunk_idx: int) -> None:
        chunk_bytes = source_sizes.filter(pl.col("chunk") == chunk_idx)["size"].sum()
        record_partitioned(state_dir, chunk_idx, chunk_bytes)

    # 4-5. Push and post-check, once the closable chunks are a big enough group
    def ready(chunks: list[int]) -> bool:
        size = open_group_bytes(state_dir, chunks)
        threshold = group_threshold_bytes(state_dir, total_source_bytes)
        print(
            f"[push] Open group: {len(chunks)} chunks, "
            f"{size / 1024**3:.2f} of {threshold / 1024**3:.2f} GB",
            flush=True,
        )
        return size >= threshold

    state = get_all_state(state_dir)
    todo = state.filter(pl.col("step") < Step.PARTITION)["chunk"].drop_nulls().unique().sort()
    try:
        # 2-3. Process and partition, CHUNK_WORKERS chunks at once (see pool.py)
        process_chunks(
            todo.to_list(),
            open_chunks(state_dir),
            workers=workers,
            start=start,
            work=partial(
                process_and_partition,
                data_dir=data_dir,
                output_dir=output_dir,
                repo_id=repo_id,
                state_dir=state_dir,
            ),
            done=done,
            ready=ready,
            close=partial(_close_open_group, target_repos=target_repos, state_dir=state_dir),
        )
        _remove_empty_dirs(output_dir)
        print("[run] All chunks complete.")
    finally:
        prefetch_executor.shutdown(wait=False, cancel_futures=True)

    # A release's sets are finalised once both are processed (see `finalise`), by the
    # `release` recipe
    if not RELEASE:
        finalise(state_dir=state_dir, hf_user=hf_user)


# Written by `finalise` once a set's tables are all sorted and its cards pushed;
# `promote-release` refuses a set without it
FINALISE_DONE = "finalise.done"


def _set_name(scholar: bool) -> str:
    return "scholarly set" if scholar else "main set"


def finalise(state_dir: Path = STATE_DIR, hf_user: str = HF_USER) -> None:
    """Finalise the uploaded tables, once every chunk is complete: compact each table's
    repo (see compact.py), then sort it by id (see sort_by_id.py), writing its partition
    metadata, then compute the cards' figures from the local copy where they are stale
    (see card_stats.py), render each table's dataset card and push it where it differs
    from the repo's (see cards.py). Resumes from the compaction and sort ledgers, and does
    nothing for a table already compacted and sorted.

    For a release, claims go first, as the largest table (its sort holds the local copy,
    its buckets and the sorted files, about 3x its size) while the other copies are not
    yet local; once sorted, claims_labels' refs are taken from them and their local copy
    deleted (CLEAN_UP_LOCAL; `download-wikidata` fetches it again). claims_labels reads
    the labels of both sets of the release (see claims_labels.py): without the other
    set's labels sorted, finalise stops before it, and is run again once they are."""
    if get_next_chunk(state_dir) is not None:
        raise RuntimeError("[finalise] Chunks are not all complete: run process-wikidata")
    if unfinished_group(state_dir):
        raise RuntimeError("[finalise] A group is not yet uploaded: run process-wikidata")
    repo = {tbl: REPO_TARGET.format(hf_user=hf_user, tbl=tbl) for tbl in Table}
    # A release's claims_labels is built from its sorted claims and labels (see
    # claims_labels.py), so it is compacted and sorted after them
    tables = [t for t in Table if not (RELEASE and t == Table.CLAIMS_LABELS)]
    if RELEASE:
        tables.sort(key=lambda t: t != Table.CLAIMS)
    for tbl in tables:
        compact_table(tbl, repo[tbl], state_dir)
        if RELEASE:  # the sort downloads the local copy it sorts from
            (HUB_COPY_DIR / tbl).mkdir(parents=True, exist_ok=True)
        sort_table(tbl, repo[tbl], state_dir)
        if RELEASE and tbl == Table.CLAIMS:
            collect_refs_stage(state_dir)
            if CLEAN_UP_LOCAL:
                shutil.rmtree(HUB_COPY_DIR / Table.CLAIMS, ignore_errors=True)
    print("[finalise] All tables compacted and sorted.")
    if RELEASE:
        assert OTHER_WORK_DIR is not None
        if sort_stage(OTHER_WORK_DIR / "state", Table.LABEL) != "done":
            print(
                f"[finalise] claims_labels waits for the {_set_name(not SCHOLAR)}'s labels "
                "to be sorted: finalise it, then run this again",
                flush=True,
            )
            return
        build_claims_labels(repo[Table.CLAIMS_LABELS], state_dir, last_chunk(state_dir), HfApi())
        compact_table(Table.CLAIMS_LABELS, repo[Table.CLAIMS_LABELS], state_dir)
        (HUB_COPY_DIR / Table.CLAIMS_LABELS).mkdir(parents=True, exist_ok=True)
        sort_table(Table.CLAIMS_LABELS, repo[Table.CLAIMS_LABELS], state_dir)
        print("[finalise] claims_labels built, compacted and sorted.")
    update_stats()
    api = HfApi()
    for tbl, card in write_cards().items():
        push_card(REPO_TARGET.format(hf_user=hf_user, tbl=tbl), card, api)
    print("[finalise] All dataset cards up to date.")
    (state_dir / FINALISE_DONE).write_text("")


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
            revision=HUB_REVISION,
            local_dir=hub_dir / tbl,
            max_workers=COMPACT_DOWNLOAD_WORKERS,
        )
    print("[download] All tables downloaded.")


def process_and_partition(
    chunk_idx: int, data_dir: Path, output_dir: Path, repo_id: str, state_dir: Path
) -> None:
    """Steps 2-3 for one chunk, in its own process (see pool.py)."""
    process(
        data_dir=data_dir,
        output_dir=output_dir,
        repo_id=repo_id,
        state_dir=state_dir,
        chunk_idx=chunk_idx,
    )
    partition_chunk(chunk_idx, state_dir, output_dir)


def _remove_empty_dirs(output_dir: Path) -> None:
    """Partition directories (a language's, a site's) left empty once merged: kept while
    chunks are partitioned, as a worker may be about to write into one."""
    for tbl in Table:
        for d in (output_dir / tbl).glob("*/"):
            try:
                d.rmdir()
            except OSError:
                pass


def _close_open_group(
    chunks: list[int], target_repos: dict[Table, str], state_dir: Path
) -> None:
    group = Group(chunks[0], chunks[-1], last_chunk(state_dir))
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
            if RELEASE and tbl == Table.ENTITIES:  # one row per entity: none may be dropped
                n_in = pl.scan_parquet(table_output_dir / filename).select(pl.len()).collect().item()
                n_out = pl.read_parquet(AUDIT_DIR / tbl / filename)["num_rows"].sum()
                if n_in != n_out:
                    raise RuntimeError(f"[partition] {tbl} {filename}: {n_in} rows in, {n_out} out")

        update_state(Path(filename), Step.PARTITION, state_dir)
        if CLEAN_UP_LOCAL:
            for tbl in Table:
                (output_dir / tbl / filename).unlink()
            print(f"[partition] Deleted processed tables for {filename}")
