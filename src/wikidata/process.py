import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl
from deepdiff import DeepDiff
from huggingface_hub import HfFileSystem
from polars_genson import (
    avro_to_polars_schema,
    normalise_from_parquet,
    read_parquet_metadata,
    schema_to_dict,
)

from .config import (
    CLEAN_UP_LOCAL,
    RELEASE,
    QUARANTINE_DIR,
    REMOTE_REPO_PATH,
    Table,
    chunk_glob,
)
from .pull import _hf_dl_subdir
from .state import Step, file_at_or_past, get_all_state, update_state

CLEAN_UP_TMP = False
repo_id = "philippesaade/wikidata"
hf_fs = HfFileSystem()


SITELINK_SCHEMA = pl.Struct({"site": pl.String, "title": pl.String})


def _map_schema(key: str, lists: bool = False) -> pl.Schema:
    """Expected schema for a map of language code to string (labels, descriptions)
    or to a list of strings (aliases)."""
    value_type = pl.List(pl.String) if lists else pl.String
    return pl.Schema(
        pl.Struct({key: pl.List(pl.Struct({"key": pl.String, "value": value_type}))})
    )


def normalise_map_direct(
    input_path: Path,
    output_path: Path,
    *,
    key: str,
    lists: bool = False,
) -> pl.DataFrame:
    """Normalise JSON map, reading schema from metadata and validating against expected."""
    with TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / output_path.name
        normalise_from_parquet(
            input_path=input_path,
            column=key,
            output_path=tmp_path,
            output_column=key,
            ndjson=True,
            wrap_root=key,
            map_threshold=0,
            typed=True,
            keep_columns=["id"],
        )

        # Read inferred schema from metadata
        metadata = read_parquet_metadata(tmp_path)
        avro_schema_json = metadata["genson_avro_schema"]
        inferred_schema = pl.Struct(avro_to_polars_schema(avro_schema_json))
        print(f"Inferred Schema: {inferred_schema}", flush=True)

        # Compare against expected
        expected_schema = _map_schema(key, lists=lists)
        d1 = schema_to_dict(expected_schema)
        d2 = schema_to_dict(pl.Schema(inferred_schema))
        if d1 != d2:
            diff = DeepDiff(d1, d2, ignore_order=True)
            if not is_acceptable_diff(diff):
                print(f"Schema mismatch in {key} for {input_path.name}:", flush=True)
                print(diff, flush=True)
                raise SystemExit(
                    f"Schema mismatch - update expected schema for {key}: {list(diff.keys())}"
                )

        result = pl.read_parquet(tmp_path).unnest(key)

    return result


def normalise_sitelinks(df: pl.DataFrame) -> pl.DataFrame:
    """Normalise JSON Map of site codes (e.g. 'enwiki') to {site,title} Records."""
    # Forced to a map whatever the number of distinct sites (the default needs over 20).
    # A forced map's record values come back as JSON strings, so decode them here.
    raw = pl.Struct(
        {"sitelinks": pl.List(pl.Struct({"key": pl.String, "value": pl.String}))}
    )
    links = df.genson.normalise_json(
        "sitelinks",
        ndjson=True,
        wrap_root="sitelinks",
        force_field_types={"sitelinks": "map"},
        decode=raw,
        max_builders=100,
    )
    decode_value = pl.element().struct.with_fields(
        pl.field("value").str.json_decode(SITELINK_SCHEMA)
    )
    links = links.with_columns(pl.col("sitelinks").list.eval(decode_value))
    # normalise_json gives one row per input row, so the ids line up
    return pl.concat([df.select("id"), links], how="horizontal")


def atomic_sink_parquet(lf: pl.LazyFrame, dst: Path) -> None:
    """Sink to a temp file then rename into place, so a Ctrl+C or crash mid-write can
    never leave a truncated file at `dst` for a later run's resume check to trust."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp")
    lf.sink_parquet(tmp)
    tmp.replace(dst)


def n_ids(fr: pl.DataFrame | pl.LazyFrame) -> int:
    """Count the unique IDs (we expect them *all* to be preserved)."""
    return fr.lazy().select(pl.col("id").n_unique()).collect().item()


def check_ids(total: int, fr: pl.DataFrame | pl.LazyFrame, *, table: str) -> None:
    """Halt if a table lost (or gained) entity IDs relative to its source file."""
    if (n := n_ids(fr)) != total:
        raise RuntimeError(f"ID loss in {table}: {total} source ids --> {n}")


KV_SCHEMA = pl.List(pl.Struct({"key": pl.String, "value": pl.String}))
DV_SCHEMA = pl.Struct(
    {
        "id": pl.String,
        "datavalue__string": pl.String,
        "precision": pl.Struct(
            {
                "precision__integer": pl.Int64,
                "precision__number": pl.Float64,
            }
        ),
        "text": pl.String,
        "language": pl.String,
        "amount": pl.String,
        "unit": pl.String,
        "upperBound": pl.String,
        "lowerBound": pl.String,
        "time": pl.String,
        "timezone": pl.Int64,
        "before": pl.Int64,
        "after": pl.Int64,
        "calendarmodel": pl.String,
        "latitude": pl.Struct(
            {
                "latitude__number": pl.Float64,
                "latitude__integer": pl.Int64,
            }
        ),
        "longitude": pl.Struct(
            {
                "longitude__number": pl.Float64,
                "longitude__integer": pl.Int64,
            }
        ),
        "altitude": pl.Null,
        "globe": pl.String,
    }
)


def _claims_schema(snak: pl.Struct) -> pl.Schema:
    groups = pl.List(pl.Struct({"key": pl.String, "value": pl.List(snak)}))
    statement = pl.Struct(
        {
            "mainsnak": snak,
            "rank": pl.String,
            "references": pl.List(groups),
            "qualifiers": groups,
        }
    )
    claims = pl.List(pl.Struct({"key": pl.String, "value": pl.List(statement)}))
    return pl.Schema({"claims": claims})


SNAK_FIELDS = {"property": pl.String, "datavalue": DV_SCHEMA, "datatype": pl.String}
# A release from an official dump keeps each snak's type ("value", "somevalue" for an
# unknown value, "novalue"), which the philippesaade copy had dropped (see dump.py)
if RELEASE:
    SNAK_FIELDS["snaktype"] = pl.String
claims_schema = _claims_schema(pl.Struct(SNAK_FIELDS))

# Snaks on deleted properties (P450, P4003) are pruned by normalise_from_parquet: a
# record holding one of these fields is removed, and a mainsnak takes its statement
# with it. A datatype lookup that failed leaves the datavalue as {value, error}; a snak
# collapsed to its bare property id is promoted to mainsnak__string (a mainsnak:
# force_scalar_promotion) or value__string (a qualifier or reference snak: wrap_scalars).
# The fields are left out of the inferred schema, so it is checked against claims_schema.
QUARANTINE_FIELDS = {"mainsnak__string", "value__string", "value", "error"}

CLAIMS_INFERENCE_OPTIONS = {
    "ndjson": True,
    "map_threshold": 0,
    "unify_maps": True,
    "force_field_types": {"mainsnak": "record"},
    "force_scalar_promotion": {
        "mainsnak",
        "datavalue",
        "precision",
        "latitude",
        "longitude",
    },
    "no_unify": {"qualifiers"},
}


# Label maps repeated in every claim that mentions an entity, property or unit, keyed by
# their sibling field. They are moved out to a per-chunk lookup table (Table.CLAIMS_LABELS).
LABEL_INVARIANTS = {
    "labels": "id",
    "property-labels": "property",
    "unit-labels": "unit",
}


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
        "value", ndjson=True, wrap_root="labels", map_threshold=0, decode=maps
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
    *,
    key: str = "claims",
) -> tuple[pl.DataFrame, pl.Schema]:
    """Normalise complex nested JSON claims to a typed frame, writing their label maps
    to `lookup_path` (see `LABEL_INVARIANTS`) and their snaks on deleted properties to
    `quarantine_path` (if any, see `QUARANTINE_FIELDS`).

    Returns the claims without those snaks, conformed to the stored `claims_schema` (so
    every chunk has the same schema, with fields it lacks as null), and the schema
    inferred for this chunk (to check against `claims_schema`).
    """
    with TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / output_path.name
        tmp_lookup = Path(tmpdir) / f"lookup_{output_path.name}"
        tmp_pruned = Path(tmpdir) / f"pruned_{output_path.name}"
        normalise_from_parquet(
            input_path=input_path,
            column=key,
            output_path=tmp_path,
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
        )
        inferred = pl.Schema(pl.read_parquet_schema(tmp_path)[key].to_schema())
        result = (
            pl.scan_parquet(
                tmp_path,
                schema={"id": pl.String, key: pl.Struct(claims_schema)},
                # Unknown fields are dropped here but halt the run at the schema check
                cast_options=pl.ScanCastOptions(
                    missing_struct_fields="insert", extra_struct_fields="ignore"
                ),
            )
            .unnest(key)
            .collect()
        )
        atomic_sink_parquet(lookup_to_long(pl.read_parquet(tmp_lookup)).lazy(), lookup_path)
        if (pruned := pl.read_parquet(tmp_pruned)).height:
            print(f"Quarantined {pruned.height} snaks on deleted properties", flush=True)
            atomic_sink_parquet(quarantine_rows(pruned).lazy(), quarantine_path)
    return result, inferred


def is_acceptable_diff(diff: DeepDiff) -> bool:
    """Diff is empty, or the schema is a subset of the one we have stored."""
    if not diff:
        return True

    # Only allow these two diff types, nothing else
    if set(diff.keys()) - {"dictionary_item_removed", "values_changed"}:
        return False

    # For values_changed: new must be subset of old
    for change in diff.get("values_changed", {}).values():
        new_keys = set(change["new_value"].keys())
        old_keys = set(change["old_value"].keys())
        if new_keys - old_keys:
            return False

    return True


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
    ds_dir = _hf_dl_subdir(data_dir, repo_id=repo_id)
    assert ds_dir.exists(), f"Dataset source directory doesn't exist: {ds_dir!s}"

    hf_local_mirror_subpath = f"{REMOTE_REPO_PATH}/{chunk_glob(chunk_idx)}"

    all_state = get_all_state(state_dir)

    for pq_path in sorted(ds_dir.glob(hf_local_mirror_subpath)):
        if file_at_or_past(pq_path.name, Step.PROCESS, all_state):
            print(f"Skipping {pq_path.name} (already processed)")
            continue

        print(f"Processing {pq_path.name}", flush=True)
        df = pl.read_parquet(pq_path)
        total = n_ids(df)

        def tbl_pq(tbl: Table) -> Path:
            return output_dir / tbl / pq_path.name

        label_pq, desc_pq, alias_pq, link_pq, claim_pq, lookup_pq = map(tbl_pq, Table)

        # A prior interrupted attempt at this chunk may have left a .tmp beside a table
        # whose real file already exists (so the code below never touches it again).
        for dst in (label_pq, desc_pq, alias_pq, link_pq, claim_pq, lookup_pq):
            dst.with_suffix(".tmp").unlink(missing_ok=True)

        # Process labels
        if label_pq.exists():
            labels = pl.read_parquet(label_pq)
        else:
            labels = normalise_map_direct(pq_path, label_pq, key="labels")
            atomic_sink_parquet(labels.lazy(), label_pq)
        check_ids(total, labels, table="labels")

        # Process descriptions
        if desc_pq.exists():
            descs = pl.read_parquet(desc_pq)
        else:
            descs = normalise_map_direct(pq_path, desc_pq, key="descriptions")
            atomic_sink_parquet(descs.lazy(), desc_pq)
        check_ids(total, descs, table="descs")

        # Process aliases
        if alias_pq.exists():
            aliases = pl.read_parquet(alias_pq)
        else:
            aliases = normalise_map_direct(pq_path, alias_pq, key="aliases", lists=True)
            atomic_sink_parquet(aliases.lazy(), alias_pq)
        check_ids(total, aliases, table="aliases")

        # Process links
        if link_pq.exists():
            links = pl.read_parquet(link_pq)
        else:
            links = normalise_sitelinks(df)
            atomic_sink_parquet(links.lazy(), link_pq)
        check_ids(total, links, table="links")

        # Claims are complex nested JSON. Dump them to disk as we go to resume easily
        tmp_batch_store = tmp_dir / pq_path.stem
        if claim_pq.exists() and lookup_pq.exists():
            claims = pl.scan_parquet(claim_pq)
        else:
            # Claims get very large so cache intermediate parquets to
            # "data/tmp/chunk_000-of-n/" dir, as files named "batch-1-of-5.parquet" etc
            cn = claim_pq.name
            # cn_idx = int(cn.split("-")[1])
            claims, inferred_claims_schema = normalise_claims_direct(
                pq_path, claim_pq, lookup_pq, QUARANTINE_DIR / pq_path.name
            )
            # Check if schema is equivalent [under permutation] to one we have stored
            d1 = schema_to_dict(claims_schema)
            d2 = schema_to_dict(inferred_claims_schema)
            if d1 != d2:
                diff = DeepDiff(d1, d2, ignore_order=True)
                if not is_acceptable_diff(diff):
                    print(f"Schema mismatch in {cn}:", flush=True)
                    print(diff, flush=True)
                    raise SystemExit(
                        f"Schema mismatch - update DV_SCHEMA for: {list(diff.keys())}"
                    )
            atomic_sink_parquet(claims.lazy(), claim_pq)
        if CLEAN_UP_TMP and tmp_batch_store.exists():
            shutil.rmtree(tmp_batch_store)
            print(f"Cleaned up {tmp_batch_store}", flush=True)
        check_ids(total, claims, table="claims")
        update_state(Path(pq_path.name), Step.PROCESS, state_dir)
        if CLEAN_UP_LOCAL:
            pq_path.unlink()
            print(f"Deleted source {pq_path.name}", flush=True)

    print("Processing complete!", flush=True)
