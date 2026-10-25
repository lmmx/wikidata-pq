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
once the labels of both sets are sorted. The refs are the union of the sidecars written
with the claims partition files (claims_refs.py), or, without a sidecar for every claims
row, read again from the sorted claims (`collect_refs`).

Each language's rows are built sorted by (`ref`, `field`), and since 2026-10-25 written
straight as their published part files (`build_language`), which the sort's own writer
writes and checks (sort_by_id.write_parts), listed in the sort's manifest for the sort's
commit, verification and local copy. The rows are already what compaction and the sort
would have made of them: compaction's deduplicating rewrite keeps each distinct row's
first occurrence, and the rows are distinct (the (`ref`, `field`) of each row ascends
strictly: a ref is one id's, and an id has a label a language), else deduplicated as it
would; and the sort's stable sort by `ref` keeps rows that `ref` already orders where
they are (checked as the sort checks its sorts, `_check_sort`, on that permutation). The
part and row-group boundaries are the sort's, planned from the rows alone
(sort_by_id.plan_rows, row_group_rows), and the files the same bytes as the sort writes
from the compacted files (scripts/test_finalise.py, docs/journal/2026-10-25-claims-labels-direct.md).
So the language's group file, compaction's rewrite of it, its sha256 against the
compaction manifest and the sort's read and sort go.

A language the sort would not take in memory, or whose rows or schema are not as above
(nulls in `ref`, `ref` not ascending, a nested column, rows Arrow cannot take in the
group file's schema), is written as before: as one group's file
(`{lang}/chunks-0000-NNNN.parquet`, see chunk_range_name) into compaction's source
directory, from where compaction and the sort take it as they take every other table's
downloaded group files (they are not uploaded: the sort uploads the table). Each claims
file's refs (FINALISE_LARGE_WORKERS at once) and each language (FINALISE_WORKERS at
once, a language whose rows may be large FINALISE_LARGE_WORKERS at once, as the sort
takes a larger key) are a job, kept as it finishes, so a restart redoes only the
unfinished ones.
"""

import io
import json
import re
import shutil
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from huggingface_hub import HfApi

from . import claims_refs
from . import sort_by_id
# (_snak_refs, _snak_table: kept importable from here, for scripts/refs_bench.py)
from .claims_refs import ENTITY_URL, _snak_refs, _snak_table, table_refs  # noqa: F401
from .compact import _src_dir as _compact_src_dir
from .config import (
    COMPACT_DIR,
    FINALISE_LARGE_WORKERS,
    HF_REPO_PRIVATE,
    SORT_STRING_BYTES,
    HUB_COPY_DIR,
    OTHER_WORK_DIR,
    WORK_DIR,
    Table,
)
from .hub import ensure_build_branch
from .parallel import in_parallel
from .push.core import _sha256
from .push.groups import chunk_range_name

BUILD_DIR = COMPACT_DIR / "claims_labels_build"
REFS_PATH = BUILD_DIR / "refs.parquet"
REFS_DIR = BUILD_DIR / "refs"  # each claims file's refs
# The refs sidecars of the set's claims partition files (claims_refs.py)
SIDECARS = WORK_DIR / claims_refs.SIDECAR_DIR
STAGES = ["refs", "written", "staged"]
# Each language written as its published files where it can be (build_language); False:
# every language as its group file, as before 2026-10-25 (scripts/test_finalise.py
# compares the two)
DIRECT = True


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


def _snaks(groups: pl.Expr) -> pl.Expr:
    """The snaks of a list of {key: property, value: [snak]} groups, one per row."""
    return groups.explode().struct.field("value").explode()


def _refs(lf: pl.LazyFrame) -> pl.LazyFrame:
    """The distinct refs of claims rows, every snak exploded by Polars (file_refs before
    2026-10-05; kept as the reference test_finalise.py and refs_bench.py compare with)."""
    main = lf.select("property", "datavalue")
    qualifiers = lf.select(_snaks(pl.col("qualifiers")).alias("s")).unnest("s")
    references = lf.select(
        _snaks(pl.col("references").explode().struct.field("snaks")).alias("s")
    ).unnest("s")
    parts = [_snak_refs(p.select("property", "datavalue")) for p in (main, qualifiers, references)]
    return pl.concat(parts).unique()


# A snak's property, and its datavalue's id and unit, at any depth: the leaves refs read
_REFS_TOP = ("property", "datavalue", "qualifiers", "references")
_REFS_LEAF = re.compile(r"(^|\.)(property|datavalue\.(id|unit))$")


def _refs_leaves(f: pq.ParquetFile) -> list[str]:
    paths = [f.schema.column(i).path for i in range(len(f.schema))]
    return [p for p in paths if p.split(".")[0] in _REFS_TOP and _REFS_LEAF.search(p)]


def file_refs(path: Path) -> pl.DataFrame:
    """The distinct (field, ref, id) referenced in one claims file: only the leaves refs
    need (each snak's property, datavalue id and unit) read by pyarrow, and the qualifier
    and reference snaks taken out of their lists by pyarrow (`list_flatten`,
    `struct_field`, no copy), not exploded by Polars as `_refs` does. For a 529 MB file:
    2 s and 6.8 GiB, against 29 s and 17.4 GiB exploding the same leaves with Polars and
    46 s and 22.3 GiB reading every column, the same refs (scripts/refs_bench.py,
    docs/journal/2026-10-05-sort-speed.md)."""
    f = pq.ParquetFile(path)
    return table_refs(f.read(columns=_refs_leaves(f)))


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


def _language_rows(refs_path: Path, label_dirs: list[Path]) -> pl.DataFrame:
    """One language's labels of the refs, from its directories of labels, sorted by
    (`ref`, `field`)."""
    labels = pl.concat([pl.scan_parquet(d / "*.parquet") for d in label_dirs]).select(
        "id", "language", pl.col("value").alias("label")
    )
    return (
        pl.scan_parquet(refs_path)
        .join(labels, on="id", how="inner")
        .select("field", "ref", "language", "label")
        .sort("ref", "field")
        .collect(engine="streaming")
    )


def _write_language(refs_path: Path, label_dirs: list[Path], dst: Path) -> int:
    """One language's labels of the refs, from its directories of labels, written to dst
    if any; returns rows (run in a worker process)."""
    rows = _language_rows(refs_path, label_dirs)
    if not rows.is_empty():
        _write(rows, dst)
    return rows.height


# claims_labels' columns, each a string (SORT_STRING_BYTES a value in a plan's estimate)
_COLUMNS = ("field", "ref", "language", "label")


def _group_schema(rows: pl.DataFrame) -> pa.Schema:
    """The Arrow schema of the rows' group file, as Polars writes it: the schema compaction
    rewrites it in (compact._key_schema) and the sort reads and writes the compacted files
    in, so the published files' schema."""
    buf = io.BytesIO()
    rows.clear().write_parquet(buf)
    return pq.read_schema(pa.BufferReader(buf.getvalue())).remove_metadata()


def _footer_decoded(schema: pa.Schema, rows: int) -> int | None:
    """The decoded bytes sort_by_id.footer_size estimates from the footers of any files of
    `rows` rows of this schema, if it is flat (None if not): a flat column's chunks hold a
    value a row (null or not), so `rows` values of each leaf at its nominal width, however
    the rows are split into files and row groups."""
    if any(pa.types.is_nested(f.type) for f in schema):
        return None
    buf = io.BytesIO()
    pq.write_table(pa.Table.from_batches([], schema), buf)
    meta = pq.read_metadata(pa.BufferReader(buf.getvalue()))
    return rows * sum(sort_by_id._nominal_width(meta.schema.column(c)) for c in range(meta.num_columns))


def _strictly_ascending(t: pa.Table, cols: list[str]) -> bool:
    """Each row's `cols` are after the row before's, lexicographically, none null: so no
    two rows are the same."""
    n = t.num_rows
    if any(t.column(c).null_count for c in cols):
        return False
    if n < 2:
        return True
    after = pa.array(np.zeros(n - 1, dtype=bool))
    tied = pa.array(np.ones(n - 1, dtype=bool))
    for c in cols:
        col = t.column(c).combine_chunks()
        prev, cur = col.slice(0, n - 1), col.slice(1)
        after = pc.or_(after, pc.and_(tied, pc.less(prev, cur)))
        tied = pc.and_(tied, pc.equal(prev, cur))
    return pc.all(after).as_py()


def _distinct_rows(t: pa.Table) -> int:
    """The table's distinct rows, counted by Arrow's hash group-by (another implementation
    than the Polars `unique` compaction deduplicates with)."""
    return t.group_by(t.column_names, use_threads=False).aggregate([]).num_rows


def _published_rows(rows: pl.DataFrame, what: str) -> tuple[pa.Table | None, pa.Schema, str]:
    """The rows compaction's deduplicating rewrite and the sort would publish of the
    language's group file, as the sort holds them before it writes its part files, if they
    can be taken here; else None and why. Returns (rows, schema, why).

    - **Compaction** keeps each distinct row's first occurrence, in order (Polars'
      `unique(keep="first", maintain_order=True)` over the key's one group file). Rows
      whose (`ref`, `field`) ascends strictly are distinct already: kept as they are.
      Otherwise (the benchmark's run on three times the rows has each label three times)
      they are deduplicated by the same `unique`, and the result checked against Arrow's
      own count: as many rows as the rows have distinct rows, none twice.
    - **The sort** sorts them stably by `ref` (`pc.sort_indices`), nulls last. Rows with
      no null `ref`, `ref` ascending, are that sort's result as they stand (a stable sort
      moves no row of a sequence already in order), which the sort's own check of a sort
      shows of the identity permutation (_check_sort).
    - **Arrow's types** are the group file's (_group_schema), which compaction's rewrite
      and the sort's read keep, in one chunk a column, as the sort's take leaves them."""
    schema = _group_schema(rows)
    ref = rows["ref"]
    if ref.null_count():
        return None, schema, "a null ref (the sort puts nulls last)"
    t = rows.to_arrow()
    if not t.schema.remove_metadata().equals(schema):
        try:
            t = t.cast(schema)
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as e:
            return None, schema, f"rows not in the group file's schema ({e})"
    t = t.combine_chunks()
    col = sort_by_id.sort_column(Table.CLAIMS_LABELS)
    refs = t.column(col)
    if t.num_rows > 1 and not pc.all(pc.less_equal(refs.slice(0, t.num_rows - 1), refs.slice(1))).as_py():
        return None, schema, f"{col} not ascending"
    if not _strictly_ascending(t, [col, "field"]):
        # (as compaction: _deduplicated)
        unique = rows.unique(subset=rows.columns, keep="first", maintain_order=True)
        distinct = _distinct_rows(t)
        t = unique.to_arrow().cast(schema).combine_chunks()
        if t.num_rows != distinct or _distinct_rows(t) != t.num_rows:
            raise RuntimeError(f"[claims_labels] {what}: deduplicated to {t.num_rows} rows, of {distinct} distinct")
    # The sort's check of its sort, of the identity: `ref` ascends, nulls last, so a stable
    # sort leaves every row where it is
    n = t.num_rows
    sort_by_id._check_sort(what, t.select([col]), pa.array(np.arange(n, dtype=np.uint64)), t, col)
    return t, schema, ""


def build_language(
    refs_path: Path, label_dirs: list[Path], lang: str, group_dst: Path, in_memory_bytes: int, file_bytes: int
) -> dict:
    """One language's labels of the refs, written as its published part files where the
    sort would sort them in memory and they are as the sort would leave them
    (_published_rows), with its plan (sort_by_id.plan_rows: `in_memory_bytes` and
    `file_bytes` its parent's, the sort's), by its writer and checked by its checks
    (sort_by_id.write_parts); else as its group file, `group_dst`, for compaction and the
    sort as before. Returns its rows, and its part files' manifest entries (None for a
    group file) (run in a worker process)."""
    table = Table.CLAIMS_LABELS
    what = f"{table}/{lang}"
    rows = _language_rows(refs_path, label_dirs)
    if rows.is_empty():
        return {"rows": 0, "files": None}
    t, schema, why = _published_rows(rows, what)
    if t is not None:
        decoded = _footer_decoded(schema, t.num_rows)
        if decoded is None:
            why = "a nested column"
        else:
            plan = sort_by_id.plan_rows(table, t.num_rows, decoded, in_memory_bytes, file_bytes)
            if plan.in_memory:
                del rows
                shutil.rmtree(sort_by_id._out_dir(table) / lang, ignore_errors=True)  # (a run interrupted mid-key)
                files = sort_by_id.write_parts(table, lang, t, schema, plan)
                # The part files' footers give the plan given (the sort plans from its
                # input's footers: the same counts for the same rows, footer_size)
                parts = [sort_by_id._out_dir(table) / lang / f["name"] for f in files]
                got = sort_by_id.plan_rows(table, *sort_by_id.footer_size(parts), in_memory_bytes, file_bytes)
                if got != plan:
                    raise RuntimeError(f"[claims_labels] {what}: the part files' footers plan {got}, not {plan}")
                return {"rows": t.num_rows, "files": files}
            why = "sorted through buckets"
    print(f"[claims_labels] {lang}: {why}: written as a group file for compaction and the sort", flush=True)
    _write(rows, group_dst)
    return {"rows": rows.height, "files": None}


def _built(lang: str, entry: dict | None) -> bool:
    """The language's part files, listed in the sort's manifest, are there as listed (by
    size and sha256): written before a restart."""
    if entry is None or entry["sources"]:
        return False
    d = sort_by_id._out_dir(Table.CLAIMS_LABELS) / lang
    return all(
        (d / f["name"]).is_file() and (d / f["name"]).stat().st_size == f["bytes"] and _sha256(d / f["name"]) == f["sha256"]
        for f in entry["files"]
    )


def _label_rows(dirs: list[Path]) -> int:
    return sum(pq.read_metadata(f).num_rows for d in dirs for f in d.glob("*.parquet"))


def write_groups(
    refs_path: Path, labels_dirs: list[Path], out_dir: Path, group: str, direct: bool = False
) -> int:
    """Each language's labels of the refs, from every directory of labels_dirs that has
    the language, as out_dir/{lang}/{group}.parquet, FINALISE_WORKERS languages at once;
    a language whose file is there (from before a restart) is kept. Returns the rows
    written by this call.

    `direct`: each language written as its published part files where it can be
    (build_language), listed in the sort's manifest as each finishes, and kept on a
    restart once listed and there as listed (_built); the others as their group files.
    A language whose rows may be a larger key's (KeyPlan.larger, its rows taken as its
    labels' rows or the refs, the fewer: a label row is in a row a field that refers to its
    id, most ids one field's) is a job in a pool of FINALISE_LARGE_WORKERS, a worker each,
    as the sort runs a larger key; the rest share the run's pool. (Only where the jobs run:
    the files are the same either way.)"""
    by_lang: dict[str, list[Path]] = {}
    for labels_dir in labels_dirs:
        for d in labels_dir.iterdir():
            if d.is_dir():
                by_lang.setdefault(d.name, []).append(d)
    dst = {lang: out_dir / lang / f"{group}.parquet" for lang in by_lang}
    desc = "claims_labels languages"
    if not direct:
        jobs = {
            lang: (refs_path, by_lang[lang], dst[lang])
            for lang in sorted(by_lang)
            if not dst[lang].exists()
        }
        # Light jobs: the workers are kept for every language
        return sum(n for _, n in in_parallel(_write_language, jobs, desc, "lang", len(by_lang)))
    table = Table.CLAIMS_LABELS
    listed = sort_by_id.read_manifest(table)
    todo = [lang for lang in sorted(by_lang) if not dst[lang].exists() and not _built(lang, listed.get(lang))]
    # (the sort's plan settings, read here: the benchmark sets them in this process alone)
    limits = (sort_by_id.SORT_IN_MEMORY_BYTES, sort_by_id.COMPACT_FILE_BYTES)
    refs = pq.read_metadata(refs_path).num_rows
    rows = {lang: min(refs, _label_rows(by_lang[lang])) for lang in todo}
    larger = {lang for lang in todo if rows[lang] * len(_COLUMNS) * SORT_STRING_BYTES > limits[0] * 3 // 16}
    n = 0
    for langs, workers, per_child in (
        ([lang for lang in todo if lang in larger], FINALISE_LARGE_WORKERS, 1),
        ([lang for lang in todo if lang not in larger], None, None),
    ):
        jobs = {lang: (refs_path, by_lang[lang], lang, dst[lang], *limits) for lang in langs}
        runs = in_parallel(build_language, jobs, desc, "lang", workers=workers, tasks_per_child=per_child)
        for lang, built in runs:
            n += built["rows"]
            if built["files"] is not None:
                sort_by_id._append_jsonl(
                    sort_by_id._manifest_path(table), {"key": lang, "sources": [], "files": built["files"]}
                )
    return n


def collect_refs_stage(state_dir: Path) -> None:
    """Collect the refs of the set's claims (stage `refs`): the union of the sidecars the
    split and the pipeline wrote beside its claims files (claims_refs.py), when they cover
    every row of the local copy of its sorted claims, else from that copy (collect_refs).
    Once recorded, the sidecars go."""
    if last_stage(state_dir) is not None:
        return
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    claims_dir = HUB_COPY_DIR / Table.CLAIMS / "all"
    refs = claims_refs.sidecar_refs(SIDECARS, claims_dir)
    if refs is None:
        print("[claims_labels] refs read from the sorted claims", flush=True)
        refs = collect_refs(claims_dir)
    refs.write_parquet(REFS_PATH)
    counts = refs.group_by("field").len().sort("field").rows()
    print(f"[claims_labels] {refs.height:,} refs: {counts}", flush=True)
    record_stage(state_dir, "refs")
    claims_refs.remove(SIDECARS)


def labels_dirs() -> list[Path]:
    """The local copies of labels this set's claims_labels reads: its own, then the other
    set's of the release."""
    dirs = [HUB_COPY_DIR / Table.LABEL]
    if OTHER_WORK_DIR is not None:
        dirs.append(OTHER_WORK_DIR / "hub" / Table.LABEL)
    return dirs


def build_claims_labels(repo_id: str, state_dir: Path, last_chunk: int, api: HfApi) -> None:
    """Build the release's claims_labels from the refs and both sets' labels: each
    language as its published part files, listed in the sort's manifest, or as its group
    file in compaction's source directory (write_groups, `direct`; resumable by stage:
    refs, written, staged). Compaction and the sort then take the group files as before,
    and the sort commits, verifies and keeps every language's part files. The repo and
    the release's branch are created, for the sort to upload to."""
    collect_refs_stage(state_dir)
    stage = last_stage(state_dir)
    done = STAGES.index(stage) if stage else -1
    if done >= STAGES.index("staged"):
        print("[claims_labels] already built", flush=True)
        return
    out_dir = BUILD_DIR / "groups"
    group = chunk_range_name(0, last_chunk, last_chunk)
    if done < STAGES.index("written"):
        n = write_groups(REFS_PATH, labels_dirs(), out_dir, group, direct=DIRECT)
        built = sort_by_id.read_manifest(Table.CLAIMS_LABELS)
        print(f"[claims_labels] {n:,} rows written: {len(built)} languages as their published files"
              f" ({sort_by_id._out_dir(Table.CLAIMS_LABELS)}), the rest to {out_dir}", flush=True)
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
