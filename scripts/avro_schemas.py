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

Usage: WIKIDATA_RELEASE=<date> python scripts/avro_schemas.py <chunk files...>
"""

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl
import pyarrow as pa
from polars_genson import normalise_from_parquet

from wikidata.process import (
    CLAIMS_INFERENCE_OPTIONS,
    LABEL_INVARIANTS,
    SCHEMAS_DIR,
    _map_schema,
)


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
            dst.write_text(inferred(column, key, **options) + "\n")
            print(f"{dst}: from {len(files)} chunks", flush=True)


if __name__ == "__main__":
    main([Path(f) for f in sys.argv[1:]])
