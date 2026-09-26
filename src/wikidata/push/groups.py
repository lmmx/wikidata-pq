"""Group ledger and adaptive group size for grouped uploads (see DESIGN.md, 4. Push).

A group is a contiguous range of partitioned chunks that is merged, uploaded and verified
together. Two append-only JSONL files in the state dir record progress:

- `partition_sizes.jsonl`: one line per partitioned chunk, its source and partition bytes
- `groups.jsonl`: one line per group stage reached (closed, merged, pushed, verified, done)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from ..config import (
    AUDIT_DIR,
    GROUP_MAX_GB,
    GROUP_MIN_GB,
    GROUP_TARGET_COUNT,
    Table,
)
from ..state import Step, get_all_state

STAGES = ["closed", "merged", "pushed", "verified", "done"]


@dataclass(frozen=True)
class Group:
    first: int
    last: int

    @property
    def name(self) -> str:
        return f"chunks-{self.first:04d}-{self.last:04d}"

    @property
    def chunks(self) -> range:
        return range(self.first, self.last + 1)


def _append(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def record_stage(state_dir: Path, group: Group, stage: str) -> None:
    assert stage in STAGES, stage
    _append(
        state_dir / "groups.jsonl",
        {"first": group.first, "last": group.last, "stage": stage},
    )


def unfinished_group(state_dir: Path) -> tuple[Group, str] | None:
    """The group closed but not yet done, with the last stage it reached, if any."""
    latest: dict[Group, str] = {}
    for r in _read(state_dir / "groups.jsonl"):
        latest[Group(r["first"], r["last"])] = r["stage"]
    open_ = [(g, s) for g, s in latest.items() if s != "done"]
    assert len(open_) <= 1, f"More than one unfinished group: {open_}"
    return open_[0] if open_ else None


def chunk_partition_bytes(chunk_idx: int, audit_dir: Path = AUDIT_DIR) -> int:
    """Total size of a chunk's partition files, from its audit sidecars."""
    return sum(
        pl.read_parquet(audit_dir / tbl / f"chunk_{chunk_idx}.parquet")["file_size"].sum()
        for tbl in Table
    )


def record_partitioned(state_dir: Path, chunk_idx: int, source_bytes: int) -> None:
    _append(
        state_dir / "partition_sizes.jsonl",
        {
            "chunk": chunk_idx,
            "source_bytes": source_bytes,
            "partition_bytes": chunk_partition_bytes(chunk_idx),
        },
    )


def _sizes(state_dir: Path) -> pl.DataFrame:
    rows = _read(state_dir / "partition_sizes.jsonl")
    schema = {"chunk": pl.Int64, "source_bytes": pl.Int64, "partition_bytes": pl.Int64}
    # A chunk re-partitioned after a crash has several lines: keep the latest
    return pl.DataFrame(rows, schema=schema).unique("chunk", keep="last")


def open_chunks(state_dir: Path) -> list[int]:
    """Partitioned chunks not yet in a group, in order."""
    state = get_all_state(state_dir)
    at_partition = state.filter(pl.col("step") == Step.PARTITION)["chunk"].to_list()
    pending = unfinished_group(state_dir)
    in_group = set(pending[0].chunks) if pending else set()
    return sorted(c for c in at_partition if c not in in_group)


def group_threshold_bytes(state_dir: Path, total_source_bytes: int) -> int:
    """Partition bytes at which to close a group: the projected total partition size over
    GROUP_TARGET_COUNT, projected from the partition/source size ratio so far."""
    sizes = _sizes(state_dir)
    gb = 1024**3
    if sizes.is_empty() or sizes["source_bytes"].sum() == 0:
        return int(GROUP_MIN_GB * gb)
    ratio = sizes["partition_bytes"].sum() / sizes["source_bytes"].sum()
    target = ratio * total_source_bytes / GROUP_TARGET_COUNT
    return int(min(max(target, GROUP_MIN_GB * gb), GROUP_MAX_GB * gb))


def open_group_bytes(state_dir: Path, chunks: list[int]) -> int:
    sizes = _sizes(state_dir).filter(pl.col("chunk").is_in(chunks))
    return int(sizes["partition_bytes"].sum())
