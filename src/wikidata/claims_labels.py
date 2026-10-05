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
see chunk_range_name) and uploaded to the release's branch, from where compaction and the
sort take them as they take every other table's.
"""

import json
import shutil
from pathlib import Path

import polars as pl
from huggingface_hub import HfApi
from tqdm import tqdm

from .config import (
    COMPACT_DIR,
    HF_REPO_PRIVATE,
    HUB_COPY_DIR,
    HUB_REVISION,
    OTHER_WORK_DIR,
    Table,
)
from .hub import ensure_build_branch
from .push.groups import chunk_range_name

ENTITY_URL = "http://www.wikidata.org/entity/"
BUILD_DIR = COMPACT_DIR / "claims_labels_build"
STAGES = ["refs", "written", "uploaded"]


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


def file_refs(path: Path) -> pl.DataFrame:
    """The distinct (field, ref, id) referenced in one claims file."""
    lf = pl.scan_parquet(path)
    main = lf.select("property", "datavalue")
    qualifiers = lf.select(_snaks(pl.col("qualifiers")).alias("s")).unnest("s")
    references = lf.select(
        _snaks(pl.col("references").explode().struct.field("snaks")).alias("s")
    ).unnest("s")
    parts = [_snak_refs(p.select("property", "datavalue")) for p in (main, qualifiers, references)]
    return pl.concat(parts).unique().collect(engine="streaming")


def collect_refs(claims_dir: Path) -> pl.DataFrame:
    files = sorted(claims_dir.glob("*.parquet"))
    if not files:
        raise RuntimeError(f"[claims_labels] no claims in {claims_dir}")
    parts = [file_refs(f) for f in tqdm(files, desc="claims_labels refs", unit="file")]
    return pl.concat(parts).unique().sort("field", "ref")


def write_groups(refs: pl.DataFrame, labels_dirs: list[Path], out_dir: Path, group: str) -> int:
    """Each language's labels of the refs, from every directory of labels_dirs that has
    the language, as out_dir/{lang}/{group}.parquet; returns rows."""
    total = 0
    by_lang: dict[str, list[Path]] = {}
    for labels_dir in labels_dirs:
        for d in labels_dir.iterdir():
            if d.is_dir():
                by_lang.setdefault(d.name, []).append(d)
    for lang in tqdm(sorted(by_lang), desc="claims_labels languages", unit="lang"):
        labels = pl.concat(
            [pl.scan_parquet(d / "*.parquet") for d in by_lang[lang]]
        ).select("id", "language", pl.col("value").alias("label"))
        rows = (
            refs.lazy()
            .join(labels, on="id", how="inner")
            .select("field", "ref", "language", "label")
            .sort("ref", "field")
            .collect(engine="streaming")
        )
        if rows.is_empty():
            continue
        dst = out_dir / lang / f"{group}.parquet"
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".tmp")
        rows.write_parquet(tmp)
        tmp.replace(dst)
        total += rows.height
    return total


REFS_PATH = BUILD_DIR / "refs.parquet"


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
    """Build the release's claims_labels from the refs and both sets' labels, and upload
    it to the release's branch (resumable by stage: refs, written, uploaded)."""
    collect_refs_stage(state_dir)
    stage = last_stage(state_dir)
    done = STAGES.index(stage) if stage else -1
    if done >= STAGES.index("uploaded"):
        print("[claims_labels] already built and uploaded", flush=True)
        return
    out_dir = BUILD_DIR / "groups"
    group = chunk_range_name(0, last_chunk, last_chunk)
    if done < STAGES.index("written"):
        shutil.rmtree(out_dir, ignore_errors=True)
        n = write_groups(pl.read_parquet(REFS_PATH), labels_dirs(), out_dir, group)
        print(f"[claims_labels] {n:,} rows written to {out_dir}", flush=True)
        record_stage(state_dir, "written")
    api.create_repo(repo_id, repo_type="dataset", private=HF_REPO_PRIVATE, exist_ok=True)
    ensure_build_branch(repo_id, api)
    api.upload_large_folder(
        repo_id=repo_id, folder_path=out_dir, repo_type="dataset", revision=HUB_REVISION
    )
    record_stage(state_dir, "uploaded")
    shutil.rmtree(BUILD_DIR, ignore_errors=True)
    print(f"[claims_labels] uploaded to {repo_id}@{HUB_REVISION}", flush=True)
