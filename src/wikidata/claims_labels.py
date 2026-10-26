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

Since 2026-10-25 too, the languages are joined with the refs a batch of them at once
(build_batch), each batch one join of the refs with all its languages' labels, each
language's rows then taken from it in (`ref`, `field`) order: the refs are read and
joined once a batch rather than once a language. A language's rows are its own join's
(_language_rows) where no two rows of the same (`ref`, `field`) differ (_ties_alike: a
release's case, and the benchmark's labels three times over), as every sort by (`ref`,
`field`) then orders them the same; else they are taken from its own join, as before. A
batch holds at most a share of a larger key's rows of labels (_batch_cap); a language
with more is built alone, as a larger key (docs/journal/2026-10-25-claims-labels-one-join.md).

Since 2026-10-26, a language built alone (build_language) is joined with the refs a row
group of its labels at a time (_streamed_rows): each row group with the refs of its ids'
range, its rows copied out as they are taken, sorted by (`ref`, `field`) at the end. The
join of all its labels at once held about every label (the rows it gave pointed into the
labels' buffers), so its peak grew with the language's labels of both sets (`en`: about
105M at a release); this holds about its rows (17.3M), whatever its labels. The rows are
the same, so the files are the same bytes; refs or labels it cannot take so are joined
as before (docs/journal/2026-10-26-claims-labels-memory.md).

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
    FINALISE_WORKERS,
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
# Languages batched (build_batch) where they can be; False: a job a language, as before
# (scripts/test_finalise.py compares the two)
BATCHED = True
# A language built alone (build_language) joined with the refs a row group of its labels
# at a time (_streamed_rows) where it can be; False: its labels joined whole, as before
# (scripts/test_finalise.py compares the two)
STREAMED = True


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


def _refs_by_field(refs_path: Path) -> list[tuple[str, pl.Series, pl.Series | None]] | None:
    """The refs, a field at a time, as _streamed_rows looks them up: each field's `id`s,
    ascending, and its `ref`s (None where each is its row's `id`, as for every field but
    `unit-labels`: the field's ids then stand for them). None if the refs are not so: not
    three string columns, a null, or a field's ids not ascending in the file (as
    collect_refs_stage writes them, sorted by (`field`, `ref`): a ref names its id, or is
    ENTITY_URL and its id, so the ids of a field ascend with its refs)."""
    scan = pl.scan_parquet(refs_path)
    if dict(scan.collect_schema()) != dict.fromkeys(["field", "ref", "id"], pl.String):
        return None
    if any(scan.select(pl.all().null_count()).collect(engine="streaming").row(0)):
        return None
    # A field's columns at a time, by the streaming engine (in the file's order), so the
    # refs are never held whole: about 3 strings a ref, against a field's ids
    runs = []
    fields = scan.select(pl.col("field").unique()).collect(engine="streaming").to_series().sort()
    for field in fields:
        mine = scan.filter(pl.col("field") == field)
        got = mine.select("id", (pl.col("ref") == pl.col("id")).alias("same")).collect(engine="streaming")
        ids = got.get_column("id")
        if not ids.is_sorted():
            return None
        same = got.get_column("same").all()
        del got
        runs.append((field, ids, None if same else mine.select("ref").collect(engine="streaming").to_series()))
    return runs


def _streamed_rows(refs_path: Path, label_dirs: list[Path]) -> pa.Table | None:
    """One language's labels of the refs, as _language_rows gives them, sorted by (`ref`,
    `field`), in the group file's types (large strings, one chunk a column), if they can
    be taken so; else None (_language_rows' own join, as before).

    The language's labels are read a row group at a time, and each row group joined
    (inner, on `id`, as _language_rows) with the refs of its ids' range, [min, max], from
    each field's refs (_refs_by_field): every ref of an id of the row group is in that
    range, so a row group's pairs are exactly the join's pairs of its label rows, and the
    row groups' pairs the join's (a label row is in one row group). Each row group's rows
    are copied out of its buffers (to Arrow) as it is joined, so the rows hold only the
    labels they take; the labels are never held whole, nor hashed whole. The rows are
    sorted by (`ref`, `field`) at the end, a column at a time. The order is _language_rows'
    where no two rows of the same (`ref`, `field`) differ (the ties checked, as
    _ties_alike does, a release's case: an id has a label a language): else None.

    The labels are held a row group at a time, the refs' ids (a field's refs only where
    they are not its ids) and the rows taken: at a release, the main set's `en` (48.7M
    labels, 20.7M refs, 17.3M rows) holds about its rows' bytes, where the join held
    about its labels' (docs/journal/2026-10-26-claims-labels-memory.md)."""
    for d in label_dirs:
        try:
            scan = pl.scan_parquet(d / "*.parquet")
            if dict(scan.select("id", "language", pl.col("value").alias("label")).collect_schema()) != _LABEL_SCHEMA:
                return None
            files = sorted(d.glob("*.parquet"))
            # (the files Polars' glob reads: _language_rows')
            if sum(pq.read_metadata(f).num_rows for f in files) != scan.select(pl.len()).collect().item():
                return None
        except (pl.exceptions.PolarsError, OSError):
            return None
    runs = _refs_by_field(refs_path)
    if runs is None:
        return None
    parts = []
    for d in label_dirs:
        for f in sorted(d.glob("*.parquet")):
            meta, at = pq.read_metadata(f), 0
            for g in range(meta.num_row_groups):
                n = meta.row_group(g).num_rows
                labels = pl.scan_parquet(f).slice(at, n).select("id", "language", pl.col("value").alias("label")).collect()
                at += n
                ids = labels.get_column("id")
                lo, hi = ids.min(), ids.max()
                if lo is None:
                    continue
                refs = []
                for field, run_ids, run_refs in runs:
                    a, z = run_ids.search_sorted(lo, side="left"), run_ids.search_sorted(hi, side="right")
                    if z > a:
                        ref = run_ids.slice(a, z - a)
                        refs.append(pl.DataFrame({
                            "field": pl.repeat(field, z - a, dtype=pl.String, eager=True),
                            "ref": ref if run_refs is None else run_refs.slice(a, z - a), "id": ref}))
                if refs:
                    rows = pl.concat(refs).join(labels, on="id", how="inner").select(*_COLUMNS)
                    parts.append(rows.to_arrow(compat_level=pl.CompatLevel.oldest()))
                del labels, ids, refs
    del runs
    schema = pa.schema([(c, pa.large_string()) for c in _COLUMNS])
    t = pa.concat_tables([p if p.schema == schema else p.cast(schema) for p in parts]) if parts else schema.empty_table()
    del parts
    order = pc.sort_indices(t, [("ref", "ascending"), ("field", "ascending")])
    # A column at a time: its parts made one array (a take of chunks makes one anyway),
    # freed, then taken in order, so at most the rows and one column more are held
    cols = {}
    for c in _COLUMNS:
        col = t.column(c).combine_chunks()
        t = t.drop_columns([c])
        cols[c] = pa.chunked_array([col.take(order)], pa.large_string())
        del col
    del order
    t = pa.table(cols)
    n = t.num_rows
    if n > 1:
        ref, field = t.column("ref").chunk(0), t.column("field").chunk(0)
        tied = pc.and_(pc.equal(ref.slice(0, n - 1), ref.slice(1)), pc.equal(field.slice(0, n - 1), field.slice(1)))
        if pc.any(tied).as_py() and _distinct_rows(t) != _distinct_rows(t.select(["ref", "field"])):
            return None
    return t


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


def _frame(rows: pl.DataFrame | pa.Table) -> pl.DataFrame:
    """The rows as a Polars frame (rows from _streamed_rows are an Arrow table)."""
    return rows if isinstance(rows, pl.DataFrame) else pl.from_arrow(rows)


def _published_rows(rows: pl.DataFrame | pa.Table, what: str) -> tuple[pa.Table | None, pa.Schema, str]:
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
      and the sort's read keep, in one chunk a column, as the sort's take leaves them.

    `rows` from _streamed_rows are an Arrow table already: taken as they are, as Polars'
    `to_arrow` gives a frame of them (the same values in the same types), and made a
    Polars frame only to be deduplicated as compaction does."""
    schema = _group_schema(rows if isinstance(rows, pl.DataFrame) else pl.from_arrow(rows.slice(0, 0)))
    if (rows["ref"].null_count() if isinstance(rows, pl.DataFrame) else rows.column("ref").null_count):
        return None, schema, "a null ref (the sort puts nulls last)"
    t = rows.to_arrow() if isinstance(rows, pl.DataFrame) else rows
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
        frame = _frame(rows)
        unique = frame.unique(subset=frame.columns, keep="first", maintain_order=True)
        del frame
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
    """One language's labels of the refs, a row group of its labels at a time
    (_streamed_rows, STREAMED) where they can be taken so, else from their own join
    (_language_rows), built (_build_rows) (run in a worker process)."""
    rows = _streamed_rows(refs_path, label_dirs) if STREAMED else None
    if rows is None:
        rows = _language_rows(refs_path, label_dirs)
    return _build_rows(rows, lang, group_dst, in_memory_bytes, file_bytes)


def _build_rows(rows: pl.DataFrame | pa.Table, lang: str, group_dst: Path, in_memory_bytes: int, file_bytes: int) -> dict:
    """One language's labels of the refs, `rows`, written as its published part files
    where the sort would sort them in memory and they are as the sort would leave them
    (_published_rows), with its plan (sort_by_id.plan_rows: `in_memory_bytes` and
    `file_bytes` its parent's, the sort's), by its writer and checked by its checks
    (sort_by_id.write_parts); else as its group file, `group_dst`, for compaction and the
    sort as before. Returns its rows, and its part files' manifest entries (None for a
    group file)."""
    table = Table.CLAIMS_LABELS
    what = f"{table}/{lang}"
    if not len(rows):
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
    rows = _frame(rows)
    _write(rows, group_dst)
    return {"rows": rows.height, "files": None}


# A batch's labels, as _language_rows selects them: (id, language, label), each a string
_LABEL_SCHEMA = {"id": pl.String, "language": pl.String, "label": pl.String}


def _labels_scan(label_dirs: list[Path]) -> pl.LazyFrame:
    """One language's labels, from its directories of labels, as _language_rows reads them."""
    return pl.concat([pl.scan_parquet(d / "*.parquet") for d in label_dirs]).select(
        "id", "language", pl.col("value").alias("label")
    )


def _batchable(label_dirs: list[Path]) -> bool:
    """The language's labels are strings, as a batch's one join takes every language's
    (a column of another type would not concatenate with the others')."""
    try:
        return dict(_labels_scan(label_dirs).collect_schema()) == _LABEL_SCHEMA
    except (pl.exceptions.PolarsError, OSError):
        return False


def _ties_alike(rows: pl.DataFrame) -> bool:
    """No two of the rows, sorted by (`ref`, `field`), have the same (`ref`, `field`) and
    differ: their order is then the one every sort by (`ref`, `field`) gives, however it
    breaks ties. A tie is two adjacent rows of the same (`ref`, `field`) (nulls equal);
    only where there is one are the rows counted (distinct rows against distinct (`ref`,
    `field`), nulls a value)."""
    same = pl.col("ref").eq_missing(pl.col("ref").shift()) & pl.col("field").eq_missing(pl.col("field").shift())
    if not rows.select(same.fill_null(False).any()).item():
        return True
    return rows.n_unique() == rows.select("ref", "field").n_unique()


def _batch_rows(refs_path: Path, batch: list[tuple[str, list[Path]]]) -> tuple[pl.DataFrame, list[int]]:
    """The labels of the refs of every language of the batch, from one join of the refs
    with all their labels, each language's rows sorted by (`ref`, `field`), the languages
    in the batch's order: the rows, and where each language's start (and the last's end).
    A label row is tagged with its language's place in the batch (from its directory, as
    _language_rows takes a language's rows from its directories), so a language's rows
    are the join's pairs of a ref and one of its label rows: _language_rows' join's."""
    labels = pl.concat(
        [_labels_scan(dirs).with_columns(_k=pl.lit(k, pl.UInt32)) for k, (_, dirs) in enumerate(batch)]
    )
    rows = (
        pl.scan_parquet(refs_path)
        .join(labels, on="id", how="inner")
        .select("_k", "field", "ref", "language", "label")
        .sort("_k", "ref", "field")
        .collect(engine="streaming")
    )
    k = rows.get_column("_k").to_numpy()
    bounds = np.searchsorted(k, np.arange(len(batch) + 1), side="left").tolist()
    return rows.drop("_k"), bounds


def build_batch(
    refs_path: Path, batch: list[tuple[str, list[Path], Path]], in_memory_bytes: int, file_bytes: int
) -> dict[str, dict]:
    """Several languages' labels of the refs, from one join (_batch_rows), each built
    (_build_rows) in the batch's order: the refs read and joined once for the batch rather
    than once a language. A language's rows are _language_rows' when no two of them tie
    on (`ref`, `field`) and differ (_ties_alike: the release's case, a label an id, and
    the benchmark's labels three times over, each tie the same row): any sort by (`ref`,
    `field`) then orders them the same. Else they come from the language's own join
    (_language_rows) as before. Returns each language's build (run in a worker process)."""
    if len(batch) == 1:
        lang, dirs, group_dst = batch[0]
        return {lang: build_language(refs_path, dirs, lang, group_dst, in_memory_bytes, file_bytes)}
    rows, bounds = _batch_rows(refs_path, [(lang, dirs) for lang, dirs, _ in batch])
    built = {}
    for k, (lang, dirs, group_dst) in enumerate(batch):
        mine = rows.slice(bounds[k], bounds[k + 1] - bounds[k])
        if not _ties_alike(mine):
            print(f"[claims_labels] {lang}: rows of the same (ref, field) differ: from its own join", flush=True)
            mine = _language_rows(refs_path, dirs)
        built[lang] = _build_rows(mine, lang, group_dst, in_memory_bytes, file_bytes)
        del mine
    return built


def _batches(rows: dict[str, int], cap: int, least: int) -> list[list[str]]:
    """The languages, by their rows (their labels'), in batches of at most `cap` rows a
    batch, and at least `least` batches (as many as there are languages, if fewer): each
    language, the most rows first, into the batch of the fewest rows that it fits, else a
    new one. A language over `cap` is a batch of its own. Each batch's languages in name
    order."""
    n = min(least, len(rows))
    load, members = [0] * n, [[] for _ in range(n)]
    for lang in sorted(rows, key=lambda x: (-rows[x], x)):
        fits = [i for i in range(len(load)) if load[i] + rows[lang] <= cap or not members[i]]
        if fits:
            i = min(fits, key=lambda i: (load[i], len(members[i]), i))
        else:
            load.append(0)
            members.append([])
            i = len(load) - 1
        load[i] += rows[lang]
        members[i].append(lang)
    return [sorted(m) for m in members if m]


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
    refs_path: Path, labels_dirs: list[Path], out_dir: Path, group: str, direct: bool = False,
    batched: bool = True,
) -> int:
    """Each language's labels of the refs, from every directory of labels_dirs that has
    the language, as out_dir/{lang}/{group}.parquet, FINALISE_WORKERS languages at once;
    a language whose file is there (from before a restart) is kept. Returns the rows
    written by this call.

    `direct`: each language written as its published part files where it can be
    (_build_rows), listed in the sort's manifest as its job finishes, and kept on a
    restart once listed and there as listed (_built); the others as their group files.
    A language whose rows may be a larger key's (KeyPlan.larger, its rows taken as its
    labels' rows or the refs, the fewer: a label row is in a row a field that refers to its
    id, most ids one field's) is a job in a pool of FINALISE_LARGE_WORKERS, a worker each,
    as the sort runs a larger key; the rest share the run's pool.

    `batched` (with `direct`): the languages joined with the refs a batch of them at once
    (build_batch, _batched_builds), not a language at a time, so the refs are read and
    joined once a batch, not once a language (618 languages a release); a language of
    more labels than a batch's, as a larger key. A job killed mid-batch leaves its
    languages unlisted: a restart rebuilds them. (Only where the jobs run: the files are
    the same either way.)"""
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
    labels = {lang: _label_rows(by_lang[lang]) for lang in todo}
    rows = {lang: min(refs, labels[lang]) for lang in todo}
    larger = {lang for lang in todo if rows[lang] * len(_COLUMNS) * SORT_STRING_BYTES > limits[0] * 3 // 16}
    n = 0
    if batched:
        builds = _batched_builds(refs_path, by_lang, dst, labels, larger, limits, desc)
    else:
        builds = _language_builds(refs_path, by_lang, dst, todo, larger, limits, desc)
    for lang, built in builds:
        n += built["rows"]
        if built["files"] is not None:
            sort_by_id._append_jsonl(
                sort_by_id._manifest_path(table), {"key": lang, "sources": [], "files": built["files"]}
            )
    return n


def _language_builds(refs_path, by_lang, dst, todo, larger, limits, desc):
    """Each language's build, a job each (build_language), (language, build) as each
    finishes: the larger ones in a pool of FINALISE_LARGE_WORKERS, a worker each, the
    rest in the run's pool."""
    for langs, workers, per_child in (
        ([lang for lang in todo if lang in larger], FINALISE_LARGE_WORKERS, 1),
        ([lang for lang in todo if lang not in larger], None, None),
    ):
        jobs = {lang: (refs_path, by_lang[lang], lang, dst[lang], *limits) for lang in langs}
        yield from in_parallel(build_language, jobs, desc, "lang", workers=workers, tasks_per_child=per_child)


# A language's job's estimated peak, for the memory budget of those built alone (in_parallel):
# a worker's floor, and a label row's bytes held by the join (its build side, with the rows
# it gives): 125-212 bytes measured on the benchmark's labels at 16 times (2026-10-25)
_LANGUAGE_FLOOR_BYTES = 400 * 1024**2
_LABEL_PEAK_BYTES = 200
# And of one built a row group of its labels at a time (_streamed_rows): a worker's floor
# with a row group of labels read and joined (128 MiB of decoded rows, about 0.5-0.9 GB
# while it is), and a row's bytes held, its rows taken by `min(refs, labels)` (each ref
# has a label a language at most): about 90 bytes a row measured on en at a quarter of a
# release's (bench/labelsjob.py's inputs, 2026-10-26), 200 taken
_STREAMED_FLOOR_BYTES = 1024**3
_ROW_PEAK_BYTES = 200
# Batches a worker at least (_batched_builds)
_BATCHES_A_WORKER = 4


def _batch_cap(in_memory_bytes: int) -> int:
    """A batch's labels' rows at most: a larger key's estimated rows (KeyPlan.larger)
    shared among the FINALISE_WORKERS batches that run at once."""
    return max(1, in_memory_bytes * 3 // 16 // (len(_COLUMNS) * SORT_STRING_BYTES) // FINALISE_WORKERS)


def _batched_builds(refs_path, by_lang, dst, labels, larger, limits, desc):
    """Each language's build, (language, build) as each finishes, in batches of languages
    a job (build_batch: one join of the refs with all their labels), the batches' labels'
    rows (the build side of the join, which a job holds) at most _batch_cap, so the
    FINALISE_WORKERS batches running at once hold at most a larger key's rows of labels
    between them, and at least four batches a worker (where the languages fit in fewer:
    smaller batches, so a worker holds about what a language's job did on the benchmark's
    sample, 2026-10-25). A
    language of more labels than a batch's (a larger key among them) is a job of its own,
    as before (build_language), FINALISE_LARGE_WORKERS at once, a worker each, as a larger
    key is, and started while its estimated peak fits the memory budget (in_parallel: its
    rows' where it is built a row group of its labels at a time, else its labels'); one
    whose labels are not strings (_batchable) is a job of its own in the run's pool. (Only
    which job builds a language: its files are the same either way.)"""
    cap = _batch_cap(limits[0])
    alone = [lang for lang in labels if lang in larger or labels[lang] > cap]
    jobs = {lang: (refs_path, by_lang[lang], lang, dst[lang], *limits) for lang in alone}
    refs = pq.read_metadata(refs_path).num_rows
    memory = {lang: _STREAMED_FLOOR_BYTES + _ROW_PEAK_BYTES * min(refs, labels[lang])
              if STREAMED and _batchable(by_lang[lang]) else _LANGUAGE_FLOOR_BYTES + _LABEL_PEAK_BYTES * labels[lang]
              for lang in alone}
    yield from in_parallel(build_language, jobs, desc, "lang", workers=FINALISE_LARGE_WORKERS, tasks_per_child=1,
                           memory=memory)
    rest = [lang for lang in labels if lang not in set(alone)]
    one = [lang for lang in rest if not _batchable(by_lang[lang])]
    jobs = {lang: (refs_path, by_lang[lang], lang, dst[lang], *limits) for lang in one}
    yield from in_parallel(build_language, jobs, desc, "lang")
    batches = _batches({lang: labels[lang] for lang in rest if lang not in set(one)}, cap, _BATCHES_A_WORKER * FINALISE_WORKERS)
    jobs = {b[0]: (refs_path, [(lang, by_lang[lang], dst[lang]) for lang in b], *limits) for b in batches}
    for _, built in in_parallel(build_batch, jobs, f"{desc} (batches)", "batch"):
        yield from built.items()


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
        n = write_groups(REFS_PATH, labels_dirs(), out_dir, group, direct=DIRECT, batched=BATCHED)
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
