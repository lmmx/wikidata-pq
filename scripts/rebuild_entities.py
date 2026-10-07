#!/usr/bin/env python3
"""Rebuild a release's entities table in both sets, from the dump, after partitioning
dropped every entity with a null field (docs/journal/2026-10-06-entities-dropped.md).

Every other table is left as it is. The entities rows are made by the pipeline's own code,
so they are what the fixed pipeline makes:

    build    Read the dump once (lbzip2), cut it into the split's chunks (CHUNK_ENTITIES
             entities each), reshape each entity (dump.entity_row), route it to its set
             (scholarly.is_scholarly), take its entities row (process: the `entity` JSON
             decoded with ENTITY_SCHEMA) and partition it (prepare_for_partition,
             partition_parquet). Each chunk's part of each set is checked against the route
             log (rows, first and last id) and its partitioned rows against its rows. Then
             each set's chunks are merged into the same group files as before (the names in
             compaction's manifest) under compact/src/entities/all, and the entities ledgers
             are reset: compaction resumes after its download, with the group files built
             here. Resumable; local only.
    hub      Leave each entities repo's release branch (build-{release}) with no data files:
             created from main if promotion deleted it. The sort then commits the sorted
             files to it.
    promote  After finalise: copy each entities repo's branch to main, replacing its files,
             tag it as the release (moving the tag if promotion made it) and delete the
             branch. Other repos are left to `just promote-release`.

Work files go to releases/{release}/entities_rebuild (deleted once both sets are built).

Usage: python scripts/rebuild_entities.py build|hub|promote [RELEASE]  (default 20260928)
"""

import json
import multiprocessing
import os
import shutil
import sys
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

RELEASE = sys.argv[2] if len(sys.argv) > 2 else "20260928"
os.environ["WIKIDATA_RELEASE"] = RELEASE
os.environ.pop("WIKIDATA_SCHOLAR", None)

import orjson  # noqa: E402
import polars as pl  # noqa: E402
from tqdm import tqdm  # noqa: E402

from wikidata.config import HF_USER, PARTITION_COLS, RELEASES_DIR, UNSPLIT_KEY, Table  # noqa: E402
from wikidata.dump import (  # noqa: E402
    CHUNK_ENTITIES,
    COLUMNS,
    MAX_PENDING_BYTES,
    _dump_lines,
    dump_path,
    entity_row,
)
from wikidata.partitioning import partition_parquet, prepare_for_partition  # noqa: E402
from wikidata.process import ENTITY_SCHEMA  # noqa: E402
from wikidata.scholarly import is_scholarly  # noqa: E402

TABLE = Table.ENTITIES
KEY = UNSPLIT_KEY  # entities are not split: every row is in this key
SETS = {"main": RELEASES_DIR / RELEASE, "scholar": RELEASES_DIR / f"{RELEASE}-scholar"}
REPOS = {"main": f"{HF_USER}/wikidata-{TABLE}", "scholar": f"{HF_USER}/wikidata-scholar-{TABLE}"}
WORK = RELEASES_DIR / RELEASE / "entities_rebuild"
BRANCH = f"build-{RELEASE}"


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _work(s: str) -> Path:
    return WORK / s


# build: one chunk of the dump


def chunk_entities(lines: list[bytes], targets: dict) -> dict[str, int]:
    """A split chunk's entities rows for each set, partitioned as the pipeline does.
    targets: {set: (set chunk number, the route log's entry for the set or None)}."""
    fields = {"entity": set(), "sitelink": set()}
    rows = [entity_row(orjson.loads(line), [0], fields) for line in lines]
    frame = pl.DataFrame(rows, schema={c: pl.String for c in COLUMNS}, orient="row")
    scholarly = frame.select(is_scholarly(pl.col("claims"))).to_series()
    parts = {"scholar": frame.filter(scholarly), "main": frame.filter(~scholarly)}
    out = {}
    for s, part in parts.items():
        number, want = targets[s]
        got = None if part.is_empty() else [part.height, part["id"][0], part["id"][-1]]
        expected = None if want is None else [want["rows"], want["first"], want["last"]]
        if got != expected:
            raise RuntimeError(f"{s} chunk {number}: rows, first, last {got}; the route log says {expected}")
        if part.is_empty():
            continue
        name = f"chunk_{number}.parquet"
        processed = _work(s) / "processed" / name
        processed.parent.mkdir(parents=True, exist_ok=True)
        entities = part.select("id", pl.col("entity").str.json_decode(ENTITY_SCHEMA)).unnest("entity")
        entities.write_parquet(processed)
        dst, audit = _work(s) / "partitioned", _work(s) / "audit"
        partition_parquet(PARTITION_COLS[TABLE], prepare_for_partition(processed, TABLE), name, dst, audit)
        n = pl.read_parquet(audit / name)["num_rows"].sum()
        if n != entities.height:
            raise RuntimeError(f"{s} chunk {number}: {entities.height} rows, {n} partitioned")
        processed.unlink()
        out[s] = n
    return out


def build_rows() -> None:
    """Every chunk's partitioned entities, for both sets."""
    route = {r["chunk"]: r for r in _jsonl(SETS["main"] / "data" / "route.jsonl")}
    number = {s: {e["split_chunk"]: e["chunk"] for e in _jsonl(d / "data" / "manifest.jsonl")} for s, d in SETS.items()}
    split_chunks = len(route)

    def targets(i: int) -> dict:
        return {s: (number[s].get(i), route[i][s]) for s in SETS}

    def done(i: int) -> bool:
        return all(n is None or (_work(s) / "audit" / f"chunk_{n}.parquet").exists() for s, (n, _) in targets(i).items())

    todo = [i for i in range(split_chunks) if not done(i)]
    print(f"[rebuild] {len(todo):,} of {split_chunks:,} chunks to build", flush=True)
    if not todo:
        return
    path = dump_path(RELEASE)
    if not path.exists():
        raise SystemExit(f"{path} is missing: run `just download-dump {RELEASE}`")
    workers = max(1, (os.cpu_count() or 2) - 2)
    pending: dict = {}
    pending_bytes = 0
    entities = {"main": 0, "scholar": 0}
    bar = tqdm(total=path.stat().st_size, unit="B", unit_scale=True, desc="dump (compressed)")
    ctx = multiprocessing.get_context("spawn")
    with path.open("rb") as source, ProcessPoolExecutor(workers, mp_context=ctx) as pool:

        def collect(block: bool) -> None:
            nonlocal pending_bytes
            if not pending:
                return
            finished, _ = wait(pending, return_when=FIRST_COMPLETED, timeout=None if block else 0)
            for future in finished:
                pending_bytes -= pending.pop(future)
                for s, n in future.result().items():
                    entities[s] += n

        def submit(i: int, batch: list[bytes], size: int) -> None:
            nonlocal pending_bytes
            if done(i):
                return
            while pending and pending_bytes + size > MAX_PENDING_BYTES:
                collect(block=True)
            pending[pool.submit(chunk_entities, batch, targets(i))] = size
            pending_bytes += size
            collect(block=False)

        index, batch, size = 0, [], 0
        for line in _dump_lines(source):
            batch.append(line)
            size += len(line)
            if len(batch) < CHUNK_ENTITIES:
                continue
            submit(index, batch, size)
            index, batch, size = index + 1, [], 0
            bar.set_postfix(chunks=index, **{s: f"{n:,}" for s, n in entities.items()})
            bar.update(os.lseek(source.fileno(), 0, os.SEEK_CUR) - bar.n)
        if batch:
            submit(index, batch, size)
            index += 1
        while pending:
            collect(block=True)
    bar.close()
    if index != split_chunks:
        raise RuntimeError(f"the dump gave {index} chunks; the split made {split_chunks}")


def build_groups(s: str) -> None:
    """The set's chunks merged into its group files, as push does, in the compaction
    source directory."""
    set_dir = SETS[s]
    saved = _work(s) / "compact_manifest.jsonl"
    if not saved.exists():  # compaction's manifest, before the reset removes it
        shutil.copy(set_dir / "compact" / "out" / TABLE / "manifest.jsonl", saved)
    (entry,) = _jsonl(saved)
    # The set's groups from its group ledger: compaction's manifest names only the group
    # files that had entities rows (in the scholarly set, 1 of 30). Named as they were then
    ranges = {(r["first"], r["last"]) for r in _jsonl(set_dir / "state" / "groups.jsonl")}
    groups = sorted((a, b, f"chunks-{a:04d}-{b:04d}.parquet") for a, b in ranges)
    if not set(entry["sources"]) <= {name for _, _, name in groups}:
        raise RuntimeError(f"{s}: group files {entry['sources']} are not all in the group ledger")
    manifest = _jsonl(set_dir / "data" / "manifest.jsonl")
    if [c for a, b, _ in groups for c in range(a, b + 1)] != list(range(len(manifest))):
        raise RuntimeError(f"{s}: the groups do not cover chunks 0 to {len(manifest) - 1} once each")
    expected = {e["chunk"]: e["rows"] for e in manifest}
    dst_dir = set_dir / "compact" / "src" / TABLE / KEY
    dst_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for a, b, name in tqdm(groups, desc=f"[rebuild] {s}: groups", unit="group"):
        want = sum(expected[c] for c in range(a, b + 1))
        dst = dst_dir / name
        if not (dst.exists() and pl.scan_parquet(dst).select(pl.len()).collect().item() == want):
            paths = [_work(s) / "partitioned" / KEY / f"chunk_{c}.parquet" for c in range(a, b + 1)]
            tmp = dst.with_suffix(".tmp")
            pl.scan_parquet(paths).sink_parquet(tmp)
            n = pl.scan_parquet(tmp).select(pl.len()).collect().item()
            if n != want:
                tmp.unlink()
                raise RuntimeError(f"{s} {name}: merged {n} rows, the set's manifest says {want}")
            tmp.replace(dst)
        total += want
    print(f"[rebuild] {s}: {len(groups)} group files, {total:,} entities", flush=True)


def reset_ledgers(s: str) -> None:
    """Compaction and the sort of entities to run again from the group files built here;
    the set to be finalised again before promotion. Once only."""
    set_dir = SETS[s]
    marker = _work(s) / "reset.done"
    if marker.exists():
        return
    state = set_dir / "state"
    for ledger in ("compact.jsonl", "sort.jsonl"):
        path = state / ledger
        shutil.copy(path, _work(s) / f"{ledger}.before-reset")
        kept = [line for line in path.read_text().splitlines() if line and json.loads(line).get("table") != TABLE]
        if ledger == "compact.jsonl":
            kept.append(json.dumps({"table": str(TABLE), "stage": "downloaded"}))
        path.write_text("\n".join(kept) + "\n")
    for d in (
        set_dir / "compact" / "out" / TABLE,
        set_dir / "compact" / "sort" / "out" / TABLE,
        set_dir / "compact" / "sort" / "buckets" / TABLE,
        set_dir / "hub" / TABLE,
    ):
        shutil.rmtree(d, ignore_errors=True)
    (state / "finalise.done").unlink(missing_ok=True)
    marker.write_text("")
    print(f"[rebuild] {s}: entities ledgers reset", flush=True)


def build() -> None:
    build_rows()
    for s in SETS:
        build_groups(s)
    for s in SETS:
        reset_ledgers(s)
    shutil.rmtree(WORK)
    print(
        f"[rebuild] done. Next: python scripts/rebuild_entities.py hub {RELEASE}",
        flush=True,
    )


# hub, promote


def hub() -> None:
    from huggingface_hub import CommitOperationDelete, HfApi

    from wikidata.hub import _data_files, ensure_build_branch

    api = HfApi()
    for repo in REPOS.values():
        ensure_build_branch(repo, api)  # created from main without its data files
        if old := _data_files(api, repo, BRANCH):
            api.create_commit(
                repo,
                repo_type="dataset",
                revision=BRANCH,
                operations=[CommitOperationDelete(path_in_repo=f) for f in old],
                commit_message=f"Release {RELEASE}: entities to be rebuilt",
            )
        print(f"[rebuild] {repo}@{BRANCH}: no data files", flush=True)
    print(
        f"[rebuild] Next: just finalise-release {RELEASE} main, "
        f"then just finalise-release {RELEASE} scholar",
        flush=True,
    )


def promote() -> None:
    from huggingface_hub import CommitOperationCopy, CommitOperationDelete, HfApi

    from wikidata.hub import _data_files

    api = HfApi()
    for s, repo in REPOS.items():
        if not (SETS[s] / "state" / "finalise.done").exists():
            raise SystemExit(f"{s} is not finalised: run just finalise-release {RELEASE} {s}")
        refs = api.list_repo_refs(repo, repo_type="dataset")
        if not any(b.name == BRANCH for b in refs.branches):
            print(f"[rebuild] {repo}: no branch {BRANCH}, already promoted", flush=True)
            continue
        new = sorted(set(_data_files(api, repo, BRANCH)) | {"README.md"})
        stale = [f for f in _data_files(api, repo, None) if f not in set(new)]
        ops = [
            CommitOperationCopy(src_path_in_repo=f, path_in_repo=f, src_revision=BRANCH) for f in new
        ] + [CommitOperationDelete(path_in_repo=f) for f in stale]
        api.create_commit(repo, repo_type="dataset", operations=ops, commit_message=f"Release {RELEASE}: entities rebuilt")
        if any(t.name == RELEASE for t in refs.tags):
            api.delete_tag(repo, tag=RELEASE, repo_type="dataset")
        api.create_tag(repo, tag=RELEASE, repo_type="dataset", revision="main")
        api.delete_branch(repo, branch=BRANCH, repo_type="dataset")
        print(f"[rebuild] {repo}: main is release {RELEASE}, entities rebuilt", flush=True)


if __name__ == "__main__":
    steps = {"build": build, "hub": hub, "promote": promote}
    if len(sys.argv) < 2 or sys.argv[1] not in steps:
        raise SystemExit(__doc__)
    steps[sys.argv[1]]()
