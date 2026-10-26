#!/usr/bin/env python3
"""Write the Avro schemas a release's tables are decoded against (process.SCHEMAS_DIR):
claims.avsc (process.CLAIMS_AVRO) and labels.avsc, descriptions.avsc, aliases.avsc
(normalise_map_direct).

Each is one schema inferred, with its table's own options (as process infers a chunk's),
from that column of all the given chunk files together: for claims, their label maps
extracted first and nothing pruned. A chunk whose rows all fit a schema is decoded
against it, without inference; any other is inferred and checked as before
(polars-genson's normalise_from_parquet(avro_schema=...)). Rerun it, on more chunks,
when the run logs show chunks falling back ("do not all fit").

The claims schema inferred is then completed against the target (`complete`, by
schema_evolution.complete_avro): every field of claims_schema the chunks did not show is
added, nullable, so a claim holding it fits rather than sending its part to inference. A
field is only ever added, never changed, so what already fitted decodes as before. `--complete` completes the stored
claims.avsc in place, without inferring.

Usage: WIKIDATA_RELEASE=<date> python scripts/avro_schemas.py <chunk files...>
       WIKIDATA_RELEASE=<date> python scripts/avro_schemas.py --complete
"""

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl
import pyarrow as pa
from polars_genson import normalise_from_parquet

from wikidata.schema_evolution import complete_avro
from wikidata.process import (
    CLAIMS_INFERENCE_OPTIONS,
    LABEL_INVARIANTS,
    SCHEMAS_DIR,
    _map_schema,
    claims_schema,
)


def complete(avsc: str) -> str:
    """claims.avsc with every field of the target schema (claims_schema, with any fields
    promoted from a release: schemas/claims.promoted.json) it lacks added, nullable: a
    claim holding such a field fits it, and is decoded as genson's inference would give it
    (a field absent is null either way). Every field claims_schema has is then in the
    schema, but for a promoted integer field beside a float one (an integer latitude goes
    to `latitude__number`, as it does today). The same completion makes each version of a
    release's schema its Avro schema as the release runs (release_schema.ReleaseSchema.avro)."""
    return complete_avro(avsc, claims_schema["claims"])


def inferred(path: Path, key: str, **options) -> str:
    """The Avro schema genson normalises the column `key` against, inferred."""
    stream = normalise_from_parquet(
        path,
        column=key,
        output_path=None,
        output_format="arrow",
        output_column=key,
        wrap_root=key,
        ndjson=True,
        typed=True,
        map_encoding="kv",
        **options,
    )
    return pa.RecordBatchReader.from_stream(stream).schema.metadata[b"genson_avro_schema"].decode()


def main(files: list[Path]) -> None:
    with TemporaryDirectory() as tmp:
        tables = {
            "claims": dict(
                **{k: v for k, v in CLAIMS_INFERENCE_OPTIONS.items() if k != "ndjson"},
                max_builders=100,
                extract_invariants=LABEL_INVARIANTS,
                lookup_output_path=Path(tmp) / "lookup.parquet",
            ),
            "labels": dict(map_threshold=0, schema=_map_schema("labels")),
            "descriptions": dict(map_threshold=0, schema=_map_schema("descriptions")),
            "aliases": dict(map_threshold=0, schema=_map_schema("aliases", lists=True)),
        }
        for key, options in tables.items():
            column = Path(tmp) / f"{key}.parquet"
            pl.concat([pl.read_parquet(f, columns=["id", key]) for f in files]).write_parquet(column)
            dst = SCHEMAS_DIR / f"{key}.avsc"
            schema = inferred(column, key, **options)
            dst.write_text((complete(schema) if key == "claims" else schema) + "\n")
            print(f"{dst}: from {len(files)} chunks", flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["--complete"]:
        dst = SCHEMAS_DIR / "claims.avsc"
        dst.write_text(complete(dst.read_text()) + "\n")
        print(f"{dst}: completed against claims_schema", flush=True)
    else:
        main([Path(f) for f in sys.argv[1:]])
