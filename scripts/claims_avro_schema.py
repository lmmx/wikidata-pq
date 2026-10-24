#!/usr/bin/env python3
"""Write the Avro schema a release's claims are decoded against (process.CLAIMS_AVRO).

One schema inferred, with the claims' own options (as normalise_claims_direct infers a
chunk's), from the claims of all the given chunk files together, their label maps
extracted first and nothing pruned. A chunk whose claims all fit it is decoded against
it, without inference; any other is inferred as before (polars-genson's
normalise_from_parquet(avro_schema=...)). Rerun it, on more chunks, when the run logs
show chunks falling back ("do not all fit").

Usage: WIKIDATA_RELEASE=<date> python scripts/claims_avro_schema.py <chunk files...>
"""

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl
import pyarrow as pa
from polars_genson import normalise_from_parquet

from wikidata.process import CLAIMS_AVRO_PATH, CLAIMS_INFERENCE_OPTIONS, LABEL_INVARIANTS


def main(files: list[Path]) -> None:
    with TemporaryDirectory() as tmp:
        claims = Path(tmp) / "claims.parquet"
        frames = [pl.read_parquet(f, columns=["id", "claims"]) for f in files]
        pl.concat(frames).write_parquet(claims)
        stream = normalise_from_parquet(
            claims,
            column="claims",
            output_path=None,
            output_format="arrow",
            output_column="claims",
            wrap_root="claims",
            **CLAIMS_INFERENCE_OPTIONS,
            max_builders=100,
            typed=True,
            map_encoding="kv",
            extract_invariants=LABEL_INVARIANTS,
            lookup_output_path=Path(tmp) / "lookup.parquet",
        )
        schema = pa.RecordBatchReader.from_stream(stream).schema
    CLAIMS_AVRO_PATH.write_text(schema.metadata[b"genson_avro_schema"].decode() + "\n")
    print(f"{CLAIMS_AVRO_PATH}: from {len(files)} chunks", flush=True)


if __name__ == "__main__":
    main([Path(f) for f in sys.argv[1:]])
