"""claims_labels for a release, from its claims and labels.

The philippesaade copy carried, in every snak, the labels of its property, item value and
unit, and process.py moved them out into claims_labels chunk by chunk. An official dump's
claims carry only ids, so a release's claims_labels is built once its claims and labels
are sorted and copied locally (HUB_COPY_DIR): every property, item (or property) value and
unit referenced by a snak (mainsnak, qualifier or reference), with each of its labels:

- `property-labels`: ref the property id
- `labels`: ref the value's id
- `unit-labels`: ref the unit as the dump writes it (http://www.wikidata.org/entity/Q…)

A release's two sets (config.SCHOLAR) refer to each other's entities (an article's
authors and journal, an item's described-by-source article), so each set's labels are
looked up in both sets' local copies of labels: `collect_refs_stage` runs once the set's
claims are sorted (after which the claims' local copy can go), and `build_claims_labels`
once the labels of both sets are sorted.

The rows are written as one group's files per language (`{lang}/chunks-0000-NNNN.parquet`,
see chunk_range_name) into compaction's source directory, from where compaction and the
sort take them as they take every other table's downloaded group files (they are not
uploaded: the sort uploads the table). Each claims file's refs (FINALISE_LARGE_WORKERS at
once) and each language's rows (FINALISE_WORKERS at once) are a job, kept as it
finishes, so a restart redoes only the unfinished ones.
"""

import json
import shutil
from pathlib import Path

import polars as pl
from huggingface_hub import HfApi

from .compact import _src_dir as _compact_src_dir
from .config import (
    COMPACT_DIR,
    FINALISE_LARGE_WORKERS,
    HF_REPO_PRIVATE,
    HUB_COPY_DIR,
    OTHER_WORK_DIR,
    Table,
)
from .hub import ensure_build_branch
from .parallel import in_parallel
from .push.groups import chunk_range_name

ENTITY_URL = "http://www.wikidata.org/entity/"
BUILD_DIR = COMPACT_DIR / "claims_labels_build"
REFS_PATH = BUILD_DIR / "refs.parquet"
REFS_DIR = BUILD_DIR / "refs"  # each claims file's refs
STAGES = ["refs", "written", "staged"]


def _ledger(state_dir: Path) -> Path:
    return state_dir / "claims_labels_build.jsonl"


def last_stage(state_dir: Path) -> str | None:
    path = _ledger(state_dir)
    if not path.exists():
        return None
    lines = [json.loads(line) for line in path.read_text().splitlines() if line]
    return lines[-1]["stage"] if lines else None


def record_stage(state_dir: Path, stage: str) -> None:
    with _ledger(state_dir).open("a") as f:
        f.write(json.dumps({"stage": stage}) + "\n")


def _snak_refs(snaks: pl.LazyFrame) -> pl.LazyFrame:
    """(field, ref, id) for snaks with `property` and `datavalue` columns: `id` is the
    entity whose labels name the ref."""
    value = pl.col("datavalue")
    unit = value.struct.field("unit")
    return pl.concat(
        [
            snaks.select(
                pl.lit("property-labels").alias("field"),
                pl.col("property").alias("ref"),
                pl.col("property").alias("id"),
            ),
            snaks.select(
                pl.lit("labels").alias("field"),
                value.struct.field("id").alias("ref"),
                value.struct.field("id").alias("id"),
            ),
            snaks.select(
                pl.lit("unit-labels").alias("field"),
                unit.alias("ref"),
                unit.str.strip_prefix(ENTITY_URL).alias("id"),
            ).filter(pl.col("ref").str.starts_with(ENTITY_URL)),
        ]
    ).drop_nulls()


def _snaks(groups: pl.Expr) -> pl.Expr:
    """The snaks of a list of {key: property, value: [snak]} groups, one per row."""
    return groups.explode().struct.field("value").explode()


def _refs(lf: pl.LazyFrame) -> pl.LazyFrame:
    main = lf.select("property", "datavalue")
    qualifiers = lf.select(_snaks(pl.col("qualifiers")).alias("s")).unnest("s")
    references = lf.select(
        _snaks(pl.col("references").explode().struct.field("snaks")).alias("s")
    ).unnest("s")
    parts = [_snak_refs(p.select("property", "datavalue")) for p in (main, qualifiers, references)]
    return pl.concat(parts).unique()


def file_refs(path: Path) -> pl.DataFrame:
    """The distinct (field, ref, id) referenced in one claims file, read whole by Polars'
    streaming engine, which spreads one file over the cores (about 50 s a 500 MB file, an
    estimated 20 GB of memory). Read a row group at a time it was slower however run: a
    Polars scan sliced to each row group about 6 min a file, pyarrow row groups about 90 s
    a file with 6 at once (docs/journal/2026-10-05-sort-speed.md)."""
    return _refs(pl.scan_parquet(path)).collect(engine="streaming")


def _write(df: pl.DataFrame, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp")
    df.write_parquet(tmp)
    tmp.replace(dst)


def _file_refs_job(path: Path, dst: Path) -> None:
    """file_refs of one claims file, written to dst (run in a worker process)."""
    _write(file_refs(path), dst)


def collect_refs(claims_dir: Path) -> pl.DataFrame:
    """The distinct refs of every claims file: each file's written to REFS_DIR as it
    finishes, FINALISE_LARGE_WORKERS at once (each holds a claims file read whole), and
    reused on a restart."""
    files = sorted(claims_dir.glob("*.parquet"))
    if not files:
        raise RuntimeError(f"[claims_labels] no claims in {claims_dir}")
    dst = {f: REFS_DIR / f.name for f in files}
    jobs = {f: (f, dst[f]) for f in files if not dst[f].exists()}
    desc = "claims_labels refs"
    workers = FINALISE_LARGE_WORKERS
    for _ in in_parallel(_file_refs_job, jobs, desc, "file", len(files), workers, tasks_per_child=1):
        pass
    return pl.concat([pl.read_parquet(dst[f]) for f in files]).unique().sort("field", "ref")


def _write_language(refs_path: Path, label_dirs: list[Path], dst: Path) -> int:
    """One language's labels of the refs, from its directories of labels, written to dst
    if any; returns rows (run in a worker process)."""
    labels = pl.concat([pl.scan_parquet(d / "*.parquet") for d in label_dirs]).select(
        "id", "language", pl.col("value").alias("label")
    )
    rows = (
        pl.scan_parquet(refs_path)
        .join(labels, on="id", how="inner")
        .select("field", "ref", "language", "label")
        .sort("ref", "field")
        .collect(engine="streaming")
    )
    if not rows.is_empty():
        _write(rows, dst)
    return rows.height


def write_groups(refs_path: Path, labels_dirs: list[Path], out_dir: Path, group: str) -> int:
    """Each language's labels of the refs, from every directory of labels_dirs that has
    the language, as out_dir/{lang}/{group}.parquet, FINALISE_WORKERS languages at once;
    a language whose file is there (from before a restart) is kept. Returns the rows
    written by this call."""
    by_lang: dict[str, list[Path]] = {}
    for labels_dir in labels_dirs:
        for d in labels_dir.iterdir():
            if d.is_dir():
                by_lang.setdefault(d.name, []).append(d)
    dst = {lang: out_dir / lang / f"{group}.parquet" for lang in by_lang}
    jobs = {
        lang: (refs_path, by_lang[lang], dst[lang])
        for lang in sorted(by_lang)
        if not dst[lang].exists()
    }
    desc = "claims_labels languages"
    return sum(n for _, n in in_parallel(_write_language, jobs, desc, "lang", len(by_lang)))



def collect_refs_stage(state_dir: Path) -> None:
    """Collect the refs from the local copy of the set's sorted claims (stage `refs`)."""
    if last_stage(state_dir) is not None:
        return
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    refs = collect_refs(HUB_COPY_DIR / Table.CLAIMS / "all")
    refs.write_parquet(REFS_PATH)
    counts = refs.group_by("field").len().sort("field").rows()
    print(f"[claims_labels] {refs.height:,} refs: {counts}", flush=True)
    record_stage(state_dir, "refs")


def labels_dirs() -> list[Path]:
    """The local copies of labels this set's claims_labels reads: its own, then the other
    set's of the release."""
    dirs = [HUB_COPY_DIR / Table.LABEL]
    if OTHER_WORK_DIR is not None:
        dirs.append(OTHER_WORK_DIR / "hub" / Table.LABEL)
    return dirs


def build_claims_labels(repo_id: str, state_dir: Path, last_chunk: int, api: HfApi) -> None:
    """Build the release's claims_labels from the refs and both sets' labels, as group
    files in compaction's source directory (resumable by stage: refs, written, staged).
    The repo and the release's branch are created, for the sort to upload to."""
    collect_refs_stage(state_dir)
    stage = last_stage(state_dir)
    done = STAGES.index(stage) if stage else -1
    if done >= STAGES.index("staged"):
        print("[claims_labels] already built", flush=True)
        return
    out_dir = BUILD_DIR / "groups"
    group = chunk_range_name(0, last_chunk, last_chunk)
    if done < STAGES.index("written"):
        n = write_groups(REFS_PATH, labels_dirs(), out_dir, group)
        print(f"[claims_labels] {n:,} rows written to {out_dir}", flush=True)
        record_stage(state_dir, "written")
    api.create_repo(repo_id, repo_type="dataset", private=HF_REPO_PRIVATE, exist_ok=True)
    ensure_build_branch(repo_id, api)
    src = _compact_src_dir(Table.CLAIMS_LABELS)
    for f in out_dir.glob("*/*.parquet"):
        (src / f.parent.name).mkdir(parents=True, exist_ok=True)
        f.replace(src / f.parent.name / f.name)
    record_stage(state_dir, "staged")
    shutil.rmtree(BUILD_DIR, ignore_errors=True)
    print(f"[claims_labels] group files moved to {src} for compaction", flush=True)
