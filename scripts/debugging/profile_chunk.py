"""Time each step of one chunk's lifecycle (pull, process substeps, partition) in
isolation, so we can see where the ~5 min/chunk cost actually goes without running the
full pipeline. Uses its own scratch dirs under testing_area/profile_chunk/ - never
touches the real state/data/results/audit dirs.

Picks a small chunk (1037, ~30MB) deliberately, so iteration is fast. Safe to re-run:
the source file is not deleted, and each step's output is overwritten.

Run from the repo root: uv run python scripts/debugging/profile_chunk.py
"""

import time
from contextlib import contextmanager
from pathlib import Path

import polars as pl

from wikidata.config import PARTITION_COLS, REPO_ID, Table
from wikidata.initial import init_files
from wikidata.partitioning import partition_parquet, prepare_for_partition
from wikidata.process import (
    n_ids,
    normalise_claims_direct,
    normalise_map_direct,
    normalise_sitelinks,
)
from wikidata.pull import pull_chunk
from wikidata.pull.core import _hf_dl_subdir

# CHUNK = 1037
CHUNK = 3

SCRATCH = Path("testing_area/profile_chunk")
STATE_DIR = SCRATCH / "state"
DATA_DIR = SCRATCH / "data"
OUTPUT_DIR = SCRATCH / "results"
AUDIT_DIR = SCRATCH / "audit"

timings: list[tuple[str, float]] = []


@contextmanager
def timed(label: str):
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    timings.append((label, elapsed))
    print(f"[{elapsed:7.2f}s] {label}", flush=True)


def main() -> None:
    for d in (STATE_DIR, DATA_DIR, OUTPUT_DIR, AUDIT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    state_file = STATE_DIR / f"chunk_{CHUNK}.jsonl"
    if not state_file.exists():
        init_files([Path(f"chunk_{CHUNK}.parquet")], STATE_DIR)

    with timed("pull"):
        pull_chunk(
            chunk_idx=CHUNK,
            state_dir=STATE_DIR,
            root_data_dir=DATA_DIR,
            repo_id=REPO_ID,
        )

    ds_dir = _hf_dl_subdir(DATA_DIR, repo_id=REPO_ID)
    pq_path = ds_dir / "data" / f"chunk_{CHUNK}.parquet"
    assert pq_path.exists(), pq_path

    with timed("read source parquet"):
        df = pl.read_parquet(pq_path)
    with timed("n_ids (total)"):
        total = n_ids(df)
    print(f"chunk {CHUNK}: {df.height} rows, {pq_path.stat().st_size / 1e6:.1f} MB")

    def tbl_pq(tbl: Table) -> Path:
        return OUTPUT_DIR / tbl / pq_path.name

    with timed("labels"):
        labels = normalise_map_direct(pq_path, tbl_pq(Table.LABEL), key="labels")
        labels.lazy().sink_parquet(tbl_pq(Table.LABEL), mkdir=True)

    with timed("descriptions"):
        descs = normalise_map_direct(pq_path, tbl_pq(Table.DESC), key="descriptions")
        descs.lazy().sink_parquet(tbl_pq(Table.DESC), mkdir=True)

    with timed("aliases"):
        aliases = normalise_map_direct(
            pq_path, tbl_pq(Table.ALIAS), key="aliases", lists=True
        )
        aliases.lazy().sink_parquet(tbl_pq(Table.ALIAS), mkdir=True)

    with timed("links (sitelinks)"):
        links = normalise_sitelinks(df)
        links.lazy().sink_parquet(tbl_pq(Table.LINKS), mkdir=True)

    with timed("claims (normalise_from_parquet + lookup)"):
        claims, _inferred = normalise_claims_direct(
            pq_path, tbl_pq(Table.CLAIMS), tbl_pq(Table.CLAIMS_LABELS)
        )
        claims.lazy().sink_parquet(tbl_pq(Table.CLAIMS), mkdir=True)

    with timed("check_ids (5x .collect() over each table)"):
        for tbl, fr in [
            (Table.LABEL, labels),
            (Table.DESC, descs),
            (Table.ALIAS, aliases),
            (Table.LINKS, links),
            (Table.CLAIMS, claims),
        ]:
            n = n_ids(fr)
            if n != total:
                print(f"  WARNING: {tbl} has {n} ids, expected {total}")

    for tbl in Table:
        with timed(f"partition: {tbl}"):
            lf = prepare_for_partition(tbl_pq(tbl), tbl)
            partition_parquet(
                by=PARTITION_COLS[tbl],
                lf=lf,
                source_name=pq_path.name,
                dst_dir=OUTPUT_DIR / tbl,
                log_dir=AUDIT_DIR / tbl,
            )

    total_time = sum(t for _, t in timings)
    print("\n--- summary ---")
    for label, t in timings:
        print(f"{t / total_time * 100:5.1f}%  {t:7.2f}s  {label}")
    print(f"{'':5}  {total_time:7.2f}s  TOTAL")


if __name__ == "__main__":
    main()
