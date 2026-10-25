"""A part's entities table written by the split (dump.route_parts), so the chunk file
holds no `entity` column and the pipeline neither reads nor decodes it.

The pipeline's process step decoded a chunk's `entity` JSON (the entities' own fields:
type, datatype, ns, title, pageid, lastrevid, modified) with ENTITY_SCHEMA and sank it as
the chunk's entities file, which partition_chunk then partitioned into its one key. The
split has that JSON in memory as it writes the chunk: here it is decoded by the same
expression and written by the same sink, and the pipeline places the file
(process.place_built). The entities' few scalar fields need no compiled builder: Polars'
decode of them is a small part of the split's time, and is the pipeline's own, so the
rows are its rows exactly (scripts/test_split_claims.py checks them).

WIKIDATA_SPLIT_ENTITIES=json leaves the `entity` column in the chunk file, for the
pipeline to decode as before.
"""

import os
from pathlib import Path

import polars as pl

from .claims_schema import check_ids, n_ids

# Where a part's entities file is, beside the part: `entities/` in its directory
ENTITIES_DIR = "entities"

SPLIT_ENTITIES = os.environ.get("WIKIDATA_SPLIT_ENTITIES", "built")
if SPLIT_ENTITIES not in ("built", "json"):
    raise SystemExit(f"WIKIDATA_SPLIT_ENTITIES is built or json, not {SPLIT_ENTITIES!r}")

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


def entities(part: pl.DataFrame) -> pl.DataFrame:
    """A part's entities rows: its `id`, and its `entity` JSON decoded (ENTITY_SCHEMA)."""
    return part.select("id", pl.col("entity").str.json_decode(ENTITY_SCHEMA)).unnest("entity")


def write_entities(part: pl.DataFrame, dst: Path) -> str | None:
    """Write a part's entities file to `dst` as the pipeline's process step writes it (its
    ids checked first, as there), by way of a temporary file. None once written, else why
    the part keeps its `entity` column (and nothing is written)."""
    if SPLIT_ENTITIES == "json":
        return "not built (WIKIDATA_SPLIT_ENTITIES=json)"
    rows = entities(part)
    check_ids(n_ids(part), rows, table="entities")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp")
    rows.lazy().sink_parquet(tmp)  # (process.atomic_sink_parquet's write)
    tmp.replace(dst)
    return None
