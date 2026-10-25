import json
import shutil
from pathlib import Path
from typing import Any
from tempfile import TemporaryDirectory

import polars as pl
import pyarrow as pa
from huggingface_hub import HfFileSystem
from polars_genson import (
    avro_to_polars_schema,
    normalise_from_parquet,
    read_parquet_metadata,
)

from .config import (
    CLEAN_UP_LOCAL,
    RELEASE,
    QUARANTINE_DIR,
    REMOTE_REPO_PATH,
    Table,
    chunk_glob,
)
from .claims_build import CLAIMS_DIR
from .claims_schema import (  # noqa: F401 (the claims' schemas, kept importable from here)
    CLAIMS_AVRO,
    CLAIMS_AVRO_PATH,
    CLAIMS_INFERENCE_OPTIONS,
    DV_SCHEMA,
    KV_SCHEMA,
    LABEL_INVARIANTS,
    QUARANTINE_FIELDS,
    SCHEMAS_DIR,
    SNAK_FIELDS,
    claims_schema,
    check_ids,
    n_ids,
)
from .partitioning import fused
from .partitioning.claims import statements
from .partitioning.core import write_table
from .pull import _hf_dl_subdir
from .state import Step, file_at_or_past, get_all_state, get_chunk_state, update_state

CLEAN_UP_TMP = False
repo_id = "philippesaade/wikidata"
hf_fs = HfFileSystem()


SITELINK_FIELDS = {"site": pl.String, "title": pl.String}
if RELEASE:  # an official dump's sitelinks also list their badges (featured article, ...)
    SITELINK_FIELDS["badges"] = pl.List(pl.String)
SITELINK_SCHEMA = pl.Struct(SITELINK_FIELDS)
_AVRO_TYPES = {
    pl.String: ["null", "string"],
    pl.List(pl.String): {"type": "array", "items": ["null", "string"]},
}
# The Avro schema sitelinks are normalised against (normalise_sitelinks): a map of
# SITELINK_FIELDS records, each nullable, the widest a chunk's sitelinks infer to
SITELINKS_AVRO = {
    "type": "record",
    "name": "document",
    "namespace": "genson",
    "fields": [
        {
            "name": "sitelinks",
            "type": {
                "type": "map",
                "name": "sitelinks",
                "values": {
                    "type": "record",
                    "name": "sitelinks_values",
                    "namespace": "genson.document_types",
                    "fields": [
                        {"name": k, "type": _AVRO_TYPES[v]} for k, v in SITELINK_FIELDS.items()
                    ],
                },
            },
        }
    ],
}
# A release's entities' own fields (dump.py keeps them as JSON in the `entity` column)
ENTITY_SCHEMA = pl.Struct(
    {
        "type": pl.String,
        # A property's value type (wikibase-item, external-id, ...); null for an item
        "datatype": pl.String,
        "ns": pl.Int64,
        "title": pl.String,
        "pageid": pl.Int64,
        "lastrevid": pl.Int64,
        "modified": pl.String,
    }
)


def _map_schema(key: str, lists: bool = False) -> pl.Schema:
    """Expected schema for a map of language code to string (labels, descriptions)
    or to a list of strings (aliases)."""
    value_type = pl.List(pl.String) if lists else pl.String
    return pl.Schema(
        pl.Struct({key: pl.List(pl.Struct({"key": pl.String, "value": value_type}))})
    )


def inferred_schema(path: Path) -> pl.Schema:
    """The schema genson inferred for the column it wrote to `path`, from the file's
    metadata (the file itself holds the expected schema it was written in)."""
    avro_schema_json = read_parquet_metadata(path)["genson_avro_schema"]
    return pl.Schema(avro_to_polars_schema(avro_schema_json, map_encoding="kv"))


def normalise_in_schema(table: str, input_path: Path, **options) -> Any:
    """`normalise_from_parquet` with a stored `schema` in `options`: genson checks the
    chunk's inferred schema against it before writing anything (`schema_check=
    "compatible"`: a field the stored schema lacks, or another dtype, does not fit; a
    field the chunk lacks, or one null in every row, does), and a schema that does not
    fit halts the run with the offending paths. Returns what normalise_from_parquet
    returns: the output's Arrow stream with `output_format="arrow"`."""
    try:
        return normalise_from_parquet(input_path=input_path, **options)
    except ValueError as e:
        if "does not fit the target schema" not in str(e):
            raise
        raise SystemExit(
            f"Schema mismatch in {table} for {input_path.name} - update its stored schema:\n{e}"
        ) from None


def normalise_map_direct(
    input_path: Path,
    output_path: Path,
    *,
    key: str,
    lists: bool = False,
) -> None:
    """Normalise a JSON map column to `output_path`, unnested in its expected schema
    (checked by `normalise_in_schema`; fields the chunk lacks are written as nulls, and a
    map empty in every row has Null values)."""
    expected_schema = _map_schema(key, lists=lists)
    # Decoded against its stored Avro schema (`schemas/{key}.avsc`) where every row fits
    # it, else inferred and checked as before (see CLAIMS_AVRO)
    avro = SCHEMAS_DIR / f"{key}.avsc"
    avro_schema = avro.read_text() if RELEASE and avro.exists() else None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".tmp")
    normalise_in_schema(
        key,
        input_path,
        column=key,
        output_path=tmp_path,
        output_column=key,
        ndjson=True,
        wrap_root=key,
        map_threshold=0,
        map_encoding="kv",
        typed=True,
        keep_columns=["id"],
        schema=expected_schema,
        unnest=True,
        avro_schema=avro_schema,
    )
    print(f"Inferred Schema: {pl.Struct(inferred_schema(tmp_path))}", flush=True)
    tmp_path.replace(output_path)


def normalise_sitelinks(df: pl.DataFrame) -> pl.DataFrame:
    """Normalise JSON Map of site codes (e.g. 'enwiki') to {site,title} Records."""
    # A map whatever the number of distinct sites, of one record, decoded to
    # SITELINK_SCHEMA. Normalised against SITELINKS_AVRO rather than a schema inferred
    # from the chunk, which took most of the step's time: the decode keeps the same fields
    # either way, and the rows are the same (an empty map or `badges` null under either).
    raw = pl.Struct(
        {"sitelinks": pl.List(pl.Struct({"key": pl.String, "value": SITELINK_SCHEMA}))}
    )
    links = df.genson.normalise_json(
        "sitelinks",
        ndjson=True,
        wrap_root="sitelinks",
        avro_schema=SITELINKS_AVRO,
        map_encoding="kv",
        decode=raw,
    )
    # normalise_json gives one row per input row, so the ids line up
    return pl.concat([df.select("id"), links], how="horizontal")


def atomic_sink_parquet(lf: pl.LazyFrame, dst: Path) -> None:
    """Sink to a temp file then rename into place, so a Ctrl+C or crash mid-write can
    never leave a truncated file at `dst` for a later run's resume check to trust."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp")
    lf.sink_parquet(tmp)
    tmp.replace(dst)


def lookup_to_long(lookup: pl.DataFrame) -> pl.DataFrame:
    """Turn the extract_invariants lookup (field, key, value as a JSON map of language to
    label) into one row per label: field, ref (the id/property/unit), language, label.
    An official dump's claims carry no labels, so their lookup is empty."""
    if lookup.is_empty():
        return pl.DataFrame(schema=dict.fromkeys(["field", "ref", "language", "label"], pl.String))
    maps = pl.Struct({"labels": KV_SCHEMA})
    # ndjson: genson splits non-delimited input on unescaped braces, even inside strings
    # (a label "…Iphigenie}" broke it); each value is one line of serialised JSON
    langs = lookup.genson.normalise_json(
        "value",
        ndjson=True,
        wrap_root="labels",
        map_threshold=0,
        map_encoding="kv",
        decode=maps,
    )
    return (
        pl.concat(
            [lookup.select("field", pl.col("key").alias("ref")), langs],
            how="horizontal",
        )
        .explode("labels", empty_as_null=True)
        .unnest("labels")
        .rename({"key": "language", "value": "label"})
        .drop_nulls("language")
    )


def quarantine_rows(pruned: pl.DataFrame) -> pl.DataFrame:
    """The pruned snaks, one row each: the entity, the claim's property, the part of the
    claim the snak is in, the property it is under there, and the snak (as JSON).

    A pruned statement's path is [claims, P, i] (its mainsnak is the snak), a qualifier
    snak's [claims, P, i, qualifiers, Q, j] and a reference snak's
    [claims, P, i, references, r, R, j].
    """
    rows = []
    for id_, path, value in pruned.select("id", "path", "value").iter_rows():
        path = json.loads(path)
        claim = path[1]
        if len(path) == 3:
            part, key, snak = "mainsnak", claim, json.dumps(json.loads(value)["mainsnak"])
        else:
            part, key, snak = path[3], path[-2], value
        rows.append((id_, claim, part, key, snak))
    return pl.DataFrame(
        rows, schema=["id", "claim", "part", "key", "snak"], orient="row"
    )


def normalise_claims_direct(
    input_path: Path,
    output_path: Path,
    lookup_path: Path,
    quarantine_path: Path,
    total: int,
    *,
    key: str = "claims",
) -> Path:
    """Normalise complex nested JSON claims to typed Parquet, writing their label maps
    to `lookup_path` (see `LABEL_INVARIANTS`) and their snaks on deleted properties to
    `quarantine_path` (if any, see `QUARANTINE_FIELDS`).

    The claims without those snaks are normalised in the stored `claims_schema` (so every
    chunk has the same schema, with fields it lacks as null), the inferred schema checked
    against it first (`normalise_in_schema`). genson hands them over as Arrow record
    batches (an Arrow C stream, its buffers moved, not copied), one row per entity, and
    they are written to Parquet beside `output_path` laid out as the claims partition,
    one row per statement (partitioning.claims.statements). The nested Parquet file
    genson would write was read back and rewritten in that layout when partitioned, which
    took most of partitioning's CPU (docs/journal/2026-10-23-claims-stream.md). The
    entities' ids are checked against `total` first, as no file keeps an entity without
    claims. Returns that file, to be renamed into place.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".tmp")
    with TemporaryDirectory() as tmpdir:
        tmp_lookup = Path(tmpdir) / f"lookup_{output_path.name}"
        tmp_pruned = Path(tmpdir) / f"pruned_{output_path.name}"
        stream = normalise_in_schema(
            key,
            input_path,
            column=key,
            output_path=None,
            output_format="arrow",
            output_column=key,
            wrap_root=key,
            **CLAIMS_INFERENCE_OPTIONS,
            profile=True,
            max_builders=100,
            typed=True,
            keep_columns=["id"],
            extract_invariants=LABEL_INVARIANTS,
            lookup_output_path=tmp_lookup,
            prune=QUARANTINE_FIELDS,
            prune_output_path=tmp_pruned,
            avro_schema=CLAIMS_AVRO,
            map_encoding="kv",
            schema=claims_schema,
            unnest=True,
        )
        claims = pa.RecordBatchReader.from_stream(stream).read_all()
        check_ids(total, pl.from_arrow(claims.select(["id"])), table="claims")
        write_table(statements(claims), tmp_path)
        del claims
        atomic_sink_parquet(lookup_to_long(pl.read_parquet(tmp_lookup)).lazy(), lookup_path)
        if (pruned := pl.read_parquet(tmp_pruned)).height:
            print(f"Quarantined {pruned.height} snaks on deleted properties", flush=True)
            atomic_sink_parquet(quarantine_rows(pruned).lazy(), quarantine_path)
    return tmp_path


def place_built_claims(built: Path, claim_pq: Path, lookup_pq: Path) -> None:
    """A chunk whose claims the split built (claims_build): its claims partition file is
    the one normalise_claims_direct would write, so it is moved into place, after the label
    lookup the claims call would give (an official dump's claims carry no labels: empty).
    No snaks are quarantined: a chunk with any to prune keeps its claims as JSON."""
    if not built.exists():
        raise RuntimeError(f"{built.parent.parent.name}/{built.name} has no claims column and "
                           f"no claims file ({built}): split again")
    atomic_sink_parquet(lookup_to_long(pl.DataFrame()).lazy(), lookup_pq)
    claim_pq.parent.mkdir(parents=True, exist_ok=True)
    try:
        built.replace(claim_pq)
    except OSError:  # another file system: copied, then renamed into place
        tmp = claim_pq.with_suffix(".tmp")
        shutil.copyfile(built, tmp)
        tmp.replace(claim_pq)
        built.unlink()


def process(
    data_dir: Path,
    output_dir: Path,
    repo_id: str,
    state_dir: Path,
    chunk_idx: int | None = None,
):
    """Flatten the source files from `data_dir` and store in `output_dir`.

    Args:
        data_dir: Root directory for HuggingFace dataset cache and other source files.
                  A temporary directory will be created directly beneath this path.
        output_dir: Destination directory where the 5 separate tables will be written,
                    each stored in a subdirectory named after the `Table` enum value.
        repo_id: HuggingFace dataset repository ID in the format 'user/dataset'.
        state_dir: Directory containing per-file state JSONL files.
        chunk_idx: If set, only process files in the specific chunk.
    """
    tmp_dir = data_dir / "tmp"
    # A release's chunks are split from its dump into data_dir (see dump.py); otherwise
    # they are downloaded from the source repo into its local mirror
    source_dir = data_dir if RELEASE else _hf_dl_subdir(data_dir, repo_id=repo_id) / REMOTE_REPO_PATH
    assert source_dir.exists(), f"Source directory doesn't exist: {source_dir!s}"

    all_state = (
        get_all_state(state_dir) if chunk_idx is None else get_chunk_state(state_dir, chunk_idx)
    )

    for pq_path in sorted(source_dir.glob(chunk_glob(chunk_idx))):
        if file_at_or_past(pq_path.name, Step.PROCESS, all_state):
            print(f"Skipping {pq_path.name} (already processed)")
            continue

        print(f"Processing {pq_path.name}", flush=True)
        # Only the columns this frame is used for: genson reads labels, descriptions,
        # aliases and claims from the file itself, so the claims JSON (most of a
        # chunk) is not decoded here
        # A part whose labels, descriptions, aliases and links the split built has none of
        # their columns (partitioning.fused): they are placed, not normalised
        in_file = pl.read_parquet_schema(pq_path)
        built_terms = not any(c in in_file for c in fused.FUSED.values())
        columns = ["id", *([] if built_terms else ["sitelinks"]), *(["entity"] if RELEASE else [])]
        df = pl.read_parquet(pq_path, columns=columns)
        total = n_ids(df)

        def tbl_pq(tbl: Table) -> Path:
            return output_dir / tbl / pq_path.name

        label_pq, desc_pq, alias_pq, link_pq, claim_pq, lookup_pq = map(
            tbl_pq,
            [Table.LABEL, Table.DESC, Table.ALIAS, Table.LINKS, Table.CLAIMS, Table.CLAIMS_LABELS],
        )

        # A prior interrupted attempt at this chunk may have left a .tmp beside a table
        # whose real file already exists (so the code below never touches it again).
        for dst in (label_pq, desc_pq, alias_pq, link_pq, claim_pq, lookup_pq):
            dst.with_suffix(".tmp").unlink(missing_ok=True)

        if built_terms:
            # Their tables as the split wrote them, already in their partitions' rows (one
            # file each, its row groups by key; the group's close partitions them), each
            # id-checked as a processed file is: a built table has no row for an entity
            # with an empty cell, which the split counted (fused.check_built_ids)
            for t in fused.FUSED:
                fused.place_one_file(pq_path.parent / t, output_dir / t, pq_path.name)
                fused.check_built_ids(total, t, output_dir / t, pq_path.name)

        # Process labels
        if built_terms:
            pass
        elif not label_pq.exists():
            normalise_map_direct(pq_path, label_pq, key="labels")
        if not built_terms:
            check_ids(total, pl.scan_parquet(label_pq), table="labels")

            # Process descriptions
            if not desc_pq.exists():
                normalise_map_direct(pq_path, desc_pq, key="descriptions")
            check_ids(total, pl.scan_parquet(desc_pq), table="descs")

            # Process aliases
            if not alias_pq.exists():
                normalise_map_direct(pq_path, alias_pq, key="aliases", lists=True)
            check_ids(total, pl.scan_parquet(alias_pq), table="aliases")

        # A release's entities' own fields, from the `entity` column (see dump.py)
        if RELEASE:
            entity_pq = tbl_pq(Table.ENTITIES)
            entity_pq.with_suffix(".tmp").unlink(missing_ok=True)
            if not entity_pq.exists():
                entities = df.select(
                    "id", pl.col("entity").str.json_decode(ENTITY_SCHEMA)
                ).unnest("entity")
                atomic_sink_parquet(entities.lazy(), entity_pq)
            check_ids(total, pl.scan_parquet(entity_pq), table="entities")

        # Process links
        if built_terms:
            pass
        elif link_pq.exists():
            links = pl.read_parquet(link_pq)
            check_ids(total, links, table="links")
        else:
            links = normalise_sitelinks(df)
            atomic_sink_parquet(links.lazy(), link_pq)
            check_ids(total, links, table="links")

        # Claims are complex nested JSON. Dump them to disk as we go to resume easily
        tmp_batch_store = tmp_dir / pq_path.stem
        if not (claim_pq.exists() and lookup_pq.exists()):
            # Claims get very large so cache intermediate parquets to
            # "data/tmp/chunk_000-of-n/" dir, as files named "batch-1-of-5.parquet" etc
            built = pq_path.parent / CLAIMS_DIR / pq_path.name
            if built.exists() or "claims" not in in_file:
                place_built_claims(built, claim_pq, lookup_pq)
            else:
                tmp_claims = normalise_claims_direct(
                    pq_path, claim_pq, lookup_pq, QUARANTINE_DIR / pq_path.name, total
                )
                tmp_claims.replace(claim_pq)
        if CLEAN_UP_TMP and tmp_batch_store.exists():
            shutil.rmtree(tmp_batch_store)
            print(f"Cleaned up {tmp_batch_store}", flush=True)
        # The claims' ids were checked before their file was written (normalise_claims_direct,
        # or the split's claims_build.write_claims)
        update_state(Path(pq_path.name), Step.PROCESS, state_dir)
        if CLEAN_UP_LOCAL:
            pq_path.unlink()
            print(f"Deleted source {pq_path.name}", flush=True)

    print("Processing complete!", flush=True)
