"""Find the claims label lookup rows that make genson's normalise_json panic.

Rebuilds chunk 2's lookup exactly as the pipeline does, saves it as
lookup_chunk_2.parquet, then bisects to the failing rows.

Run from the repo root: uv run python scripts/debugging/bisect_lookup.py
"""

import json
import tempfile

import polars as pl
from polars_genson import normalise_from_parquet

from wikidata.process import CLAIMS_INFERENCE_OPTIONS, LABEL_INVARIANTS

src = "data/huggingface_hub/philippesaade/wikidata/data/chunk_2.parquet"
with tempfile.TemporaryDirectory() as d:
    normalise_from_parquet(
        input_path=src,
        column="claims",
        output_path=f"{d}/c.parquet",
        output_column="claims",
        wrap_root="claims",
        **CLAIMS_INFERENCE_OPTIONS,
        max_builders=100,
        typed=True,
        keep_columns=["id"],
        extract_invariants=LABEL_INVARIANTS,
        lookup_output_path=f"{d}/lk.parquet",
    )
    lk = pl.read_parquet(f"{d}/lk.parquet")
lk.write_parquet("lookup_chunk_2.parquet")
print("lookup rows:", lk.height)


def ok(idx: list[int]) -> bool:
    try:
        lk[idx].genson.normalise_json("value", wrap_root="labels", map_threshold=0)
        return True
    except Exception:
        return False


bad: list[int] = []


def bisect(idx: list[int]) -> None:
    if ok(idx):
        return
    if len(idx) == 1:
        bad.append(idx[0])
        return
    m = len(idx) // 2
    bisect(idx[:m])
    bisect(idx[m:])


bisect(list(range(lk.height)))
print("failing rows:", len(bad))
for i in bad[:10]:
    field, key, value = lk.row(i)
    try:
        parsed = type(json.loads(value)).__name__
    except Exception as e:
        parsed = f"json.loads fails: {e}"
    print(f"{field} {key} ({parsed}) len={len(value)}: {value[:300]!r}")
