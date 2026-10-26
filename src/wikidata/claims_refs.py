"""claims_labels' refs of each claims partition file, taken from its statements while they
are in memory and kept as a small sidecar, so that a release's claims_labels reads the
sidecars rather than every sorted claims file again (claims_labels.collect_refs).

A claims partition file is written in two places: by the split (claims_build.write_claims),
a batch of statements at a time through `with_refs`, which gathers each batch's refs, and,
for a part whose claims stay JSON, by the pipeline (process.normalise_claims_direct) with
`write_with_refs`. Either way the file is written to its temporary path, then its sidecar,
and only then is the file renamed into place. So a claims file in place has its sidecar, unless
it was written before sidecars were (or outside a release's work directories: a test's).

A sidecar holds the distinct (field, ref) of its statements: the refs file_refs takes from
the file (the same set; scripts/test_claims_refs.py), gathered without Polars or a Python
loop by a row (`Refs`). Each batch's referenced columns (each snak's property, datavalue id
and unit, flattened out of their lists without a copy) go through one pass of Arrow's hash
kernel (`pc.unique`), and the batches' distinct values once more when the file closes:
about 0.15 CPU-s for the bench's largest part (c0, 620k statements), against 0.45 s for
the Polars pass (table_refs) it replaced. It also holds the statements' row count, in its
metadata. The refs' `id` is left out (half a sidecar's bytes): it is a function of
(field, ref) (see `_snak_refs`), put back as the sidecars are read (`with_id`). A sidecar
is about 2.5-4.5% of its claims file's bytes (the bench's chunks: 487 KiB beside 10.6 MiB). It is named by the statements' row count
and the statements' first and last ids, so a part has one name whichever of the two writes
it, and a part written again replaces its own. The sidecars are kept in their set's work
directory (`claims_refs/`), which neither routing nor the pipeline's clean-up touches, and
where both writers find it from the claims file's path.

claims_labels takes the union of a set's sidecars only when their rows add up to the rows of
the set's sorted claims (`sidecar_refs`): each statement is in exactly one claims file, and
the sorted claims are those files' rows. A missing sidecar (or a stray one) makes the counts
differ, and the refs are read from the sorted claims as before.
"""

import shutil
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .config import OTHER_WORK_DIR, RELEASE, WORK_DIR
from .claims_derived import reduce
from .partitioning.core import write_table

ENTITY_URL = "http://www.wikidata.org/entity/"
# A set's sidecars, in its work directory
SIDECAR_DIR = "claims_refs"
# The statements' row count, in a sidecar's Parquet metadata
ROWS_KEY = b"wikidata.claims_rows"
SIDECAR_SCHEMA = dict.fromkeys(["field", "ref"], pl.String)


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


def with_id(refs: pl.DataFrame) -> pl.DataFrame:
    """Refs (field, ref) with the `id` `_snak_refs` gives each: the ref itself, or a unit's
    entity id without ENTITY_URL."""
    unit = pl.col("field") == "unit-labels"
    ref = pl.col("ref")
    return refs.with_columns(pl.when(unit).then(ref.str.strip_prefix(ENTITY_URL)).otherwise(ref).alias("id"))


def _datavalue(values: pa.ChunkedArray) -> pa.ChunkedArray:
    """A datavalue column with only the fields refs read (its `id` and `unit`), as a file's
    two leaves read back give it: each field null where its datavalue is (`struct_field`
    applies the parent's nulls). So the rest of a statement's value is not handed to
    Polars."""
    chunks = [
        pa.StructArray.from_arrays(
            [pc.struct_field(c, "id"), pc.struct_field(c, "unit")],
            names=["id", "unit"],
            mask=c.is_null() if c.null_count else None,
        )
        for c in values.chunks
    ]
    if not chunks:  # (no rows)
        return pa.chunked_array([], pa.struct([values.type.field("id"), values.type.field("unit")]))
    return pa.chunked_array(chunks)


def _snaks_table(property: pa.ChunkedArray, datavalue: pa.ChunkedArray) -> pa.Table:
    """Snaks' columns as the (property, datavalue) table _snak_refs takes."""
    return pa.table({"property": property, "datavalue": _datavalue(datavalue)})


def _snak_table(snaks: pa.ChunkedArray) -> pa.Table:
    """A flat array of snaks as the (property, datavalue) table _snak_refs takes."""
    return _snaks_table(pc.struct_field(snaks, "property"), pc.struct_field(snaks, "datavalue"))


def table_refs(t: pa.Table) -> pl.DataFrame:
    """The distinct (field, ref, id) referenced by a table of claims statements (the claims
    partition's layout: at least its `property`, `datavalue`, `qualifiers` and
    `references`), unordered. The qualifier and reference snaks are taken out of their
    lists by pyarrow (`list_flatten`, `struct_field`, no copy), not exploded by Polars."""
    # qualifiers: [{key, value: [snak]}]; references: [{snaks: [{key, value: [snak]}]}]
    qualifiers = pc.list_flatten(pc.struct_field(pc.list_flatten(t["qualifiers"]), "value"))
    groups = pc.list_flatten(pc.struct_field(pc.list_flatten(t["references"]), "snaks"))
    references = pc.list_flatten(pc.struct_field(groups, "value"))
    parts = [_snaks_table(t["property"], t["datavalue"]), _snak_table(qualifiers), _snak_table(references)]
    return pl.concat([_snak_refs(pl.from_arrow(p).lazy()) for p in parts]).unique().collect()


def sidecar_dir(claims_file: Path) -> Path | None:
    """The sidecar directory of the release set whose work directory holds `claims_file`
    (the split writes both sets' claims files, each into its set's data directory), or
    None outside them."""
    if not RELEASE:
        return None
    path = claims_file.resolve()
    for work in (WORK_DIR, OTHER_WORK_DIR):
        if work is not None and path.is_relative_to(work.resolve()):
            return work / SIDECAR_DIR
    return None


# The fields of a sidecar's refs, in its order
FIELDS = ("labels", "property-labels", "unit-labels")


def _referenced(t: pa.Table) -> dict[str, list[pa.ChunkedArray]]:
    """Each field's referenced values in a table of statements, the main snaks', the
    qualifiers' and the references' (as table_refs takes them, by pyarrow's `list_flatten`
    and `struct_field`, no copy): each snak's property, its datavalue's id and its unit,
    null where the snak or its datavalue is (struct_field applies the parent's nulls)."""
    qualifiers = pc.list_flatten(pc.struct_field(pc.list_flatten(t["qualifiers"]), "value"))
    groups = pc.list_flatten(pc.struct_field(pc.list_flatten(t["references"]), "snaks"))
    references = pc.list_flatten(pc.struct_field(groups, "value"))
    snaks = [qualifiers, references]
    values = [t["datavalue"], *(pc.struct_field(s, "datavalue") for s in snaks)]
    return {
        "property-labels": [t["property"], *(pc.struct_field(s, "property") for s in snaks)],
        "labels": [pc.struct_field(v, "id") for v in values],
        "unit-labels": [pc.struct_field(v, "unit") for v in values],
    }


def _distinct(arrays: list) -> pa.Array:
    """The distinct non-null values of `arrays` (strings), in one hash pass (pc.unique)."""
    chunks = [c for a in arrays for c in (a.chunks if isinstance(a, pa.ChunkedArray) else [a])]
    return pc.drop_null(pc.unique(pa.chunked_array(chunks, pa.large_string())))


class Refs:
    """A claims file's refs, gathered as its statements are written, a batch at a time (or
    all at once): each field's distinct values in each batch (Arrow's hash kernel over the
    columns already in memory, one pass, no Python per row and no Polars), the statements'
    rows, and their first and last ids. The same refs as table_refs (and file_refs) takes:
    `_snak_refs` keeps a non-null property, datavalue id or entity unit, and its `id` is a
    function of the ref (`with_id`)."""

    def __init__(self):
        self.parts: dict[str, list[pa.Array]] = {f: [] for f in FIELDS}
        self.rows, self.first, self.last = 0, None, None

    def add(self, table: pa.Table) -> None:
        if table.num_rows == 0:
            return
        ids = table.column("id")
        if self.first is None:
            self.first = ids[0].as_py()
        self.last = ids[-1].as_py()
        self.rows += table.num_rows
        for field, arrays in _referenced(table).items():
            self.parts[field].append(_distinct(arrays))

    def table(self) -> pa.Table:
        """The distinct (field, ref), by field, each field's refs in the order they first
        appear (not sorted: the union deduplicates and sorts them, and a sort was a fifth of
        the sidecar's cost)."""
        refs = []
        for field in FIELDS:
            values = _distinct(self.parts[field])
            if field == "unit-labels":
                values = values.filter(pc.starts_with(values, pattern=ENTITY_URL))
            refs.append(values)
        counts = np.array([len(r) for r in refs])
        fields = pa.DictionaryArray.from_arrays(
            pa.array(np.repeat(np.arange(len(FIELDS), dtype=np.int8), counts)),
            pa.array(FIELDS, pa.large_string()),
        ).dictionary_decode()
        return pa.table({"field": fields, "ref": pa.chunked_array(refs, pa.large_string())})


def write_refs(refs: "Refs | pa.Table", claims_file: Path) -> Path | None:
    """Write the sidecar of `claims_file`'s statements (`refs`, or the statements
    themselves): their distinct refs' (field, ref) (Refs.table), their row count in its metadata,
    by way of a temporary file: dictionary-encoded, zstd, without statistics (a sidecar is
    only ever read whole). Returns it, or None if the file is outside a release's work
    directories or has no rows (as no claims file of no rows reaches the sorted claims)."""
    if isinstance(refs, pa.Table):
        table, refs = refs, Refs()
        refs.add(table)
    d = sidecar_dir(claims_file)
    if d is None or refs.rows == 0:
        return None
    dst = d / f"{refs.rows}-{refs.first}-{refs.last}.parquet"
    out = refs.table().replace_schema_metadata({ROWS_KEY: str(refs.rows).encode()})
    d.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp")
    pq.write_table(out, tmp, compression="zstd", write_statistics=False)
    tmp.replace(dst)
    return dst


def write_with_refs(table: pa.Table, tmp: Path, claims_file: Path) -> None:
    """Write a claims partition file's statements to `tmp` (write_table), without the
    derived leaves their values allow (claims_derived.reduce), then their refs sidecar
    (write_refs), before the caller renames `tmp` to `claims_file`."""
    write_table(reduce(table), tmp)
    write_refs(table, claims_file)


class _WithRefs:
    """A claims file writer (claims_build.RowGroups) whose statements' refs are gathered as
    they are added, and written as the file's sidecar when it is closed, before the file is
    renamed into place. Every other attribute is the writer's."""

    def __init__(self, writer):
        self.writer, self.refs = writer, Refs()

    def add(self, table: pa.Table) -> None:
        self.refs.add(table)
        self.writer.add(table)

    def close(self, dst: Path) -> None:
        write_refs(self.refs, dst)
        self.writer.close(dst)

    def __getattr__(self, name):
        return getattr(self.writer, name)


def with_refs(writer):
    """`writer` (with `add(statements)` and `close(dst)`), writing the file's refs sidecar
    too (see _WithRefs)."""
    return _WithRefs(writer)


def _rows(sidecar: Path) -> int:
    return int(pq.read_schema(sidecar).metadata[ROWS_KEY])


def union(paths: list[Path], every: int = 4_000_000) -> pl.DataFrame:
    """The distinct refs (field, ref, id) of the sidecars at `paths`, sorted by (field, ref),
    as collect_refs gives them. Deduplicated as
    they are read, whenever the rows read since the last time pass those kept (and
    `every`), so at most about twice the distinct refs are held, not every sidecar's."""
    kept = pl.DataFrame(schema=SIDECAR_SCHEMA)
    pending: list[pl.DataFrame] = []
    n = 0
    for p in paths:
        pending.append(pl.read_parquet(p))
        n += pending[-1].height
        if n >= max(kept.height, every):
            kept, pending, n = pl.concat([kept, *pending]).unique(), [], 0
    return with_id(pl.concat([kept, *pending]).unique().sort("field", "ref"))


def sidecar_refs(sidecars: Path, claims_dir: Path) -> pl.DataFrame | None:
    """The distinct refs of the claims files in `claims_dir` from the sidecars in
    `sidecars`, as collect_refs gives them, if the sidecars' rows add up to the claims'
    (read from their footers), else None."""
    paths = sorted(sidecars.glob("*.parquet")) if sidecars.is_dir() else []
    claims = sum(pq.read_metadata(f).num_rows for f in claims_dir.glob("*.parquet"))
    rows = sum(map(_rows, paths))
    print(f"[claims_labels] {len(paths):,} refs sidecars of {rows:,} claims rows, "
          f"{claims:,} in the sorted claims", flush=True)
    if not paths or rows != claims:
        return None
    return union(paths)


def remove(sidecars: Path) -> None:
    shutil.rmtree(sidecars, ignore_errors=True)
