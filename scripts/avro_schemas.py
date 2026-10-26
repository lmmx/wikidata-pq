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

The claims schema inferred is then completed against the target (`complete`): every
field of claims_schema the chunks did not show is added, nullable, so a claim holding it
fits rather than sending its part to inference. A field is only ever added, never
changed, so what already fitted decodes as before. `--complete` completes the stored
claims.avsc in place, without inferring.

Usage: WIKIDATA_RELEASE=<date> python scripts/avro_schemas.py <chunk files...>
       WIKIDATA_RELEASE=<date> python scripts/avro_schemas.py --complete
"""

import json
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
    claims_schema,
)

_AVRO_SCALARS = {pl.String: "string", pl.Int64: "int", pl.Float64: "float"}
_INT_SUFFIXES = ("int", "integer", "long")
_FLOAT_SUFFIXES = ("float", "double", "number")


def _suffix(name: str) -> str:
    """A promoted field's type suffix (`latitude__number`: number), as genson reads it."""
    return name.split("__")[-1] if "__" in name else ""


def _is_map(dtype: pl.DataType) -> bool:
    entry = dtype.inner if isinstance(dtype, pl.List) else None
    return isinstance(entry, pl.Struct) and [f.name for f in entry.fields] == ["key", "value"]


def _new(dtype: pl.DataType, name: str, namespace: str):
    """The Avro type of a target field the inferred schema lacks: nullable at every level
    (a field genson never saw), its records named as genson names them."""
    if dtype == pl.Null:
        return ["null"]
    if dtype in _AVRO_SCALARS:
        return ["null", _AVRO_SCALARS[dtype]]
    if isinstance(dtype, pl.Struct):
        record = {"type": "record", "name": name, "namespace": namespace, "fields": []}
        return ["null", _completed(record, dtype)]
    if _is_map(dtype):
        values = _new(dtype.inner.fields[1].dtype, f"{name}_values", f"{namespace}.{name}_types")
        return ["null", {"type": "map", "values": values}]
    if isinstance(dtype, pl.List):
        return ["null", {"type": "array", "items": _new(dtype.inner, name, f"{namespace}.{name}_types")}]
    raise ValueError(f"a target field {name} of {dtype}, with no Avro type")


def _completed(record: dict, dtype: pl.Struct) -> dict:
    """An Avro record with the target struct's fields it lacks added (see `complete`)."""
    have = {f["name"]: f for f in record["fields"]}
    target = {f.name: f.dtype for f in dtype.fields}
    namespace = f"{record['namespace']}.{record['name']}_types"
    for name, t in target.items():
        if name in have:
            have[name]["type"] = _complete(have[name]["type"], t, name, namespace)
            continue
        # Never a promoted integer field beside a float one: genson's strict decode gives an
        # integer to the float field (`__number`) only where its record has no integer
        # field, so adding one would move every integer the schema already took
        siblings = [n for n in [*have, *target] if n.split("__")[0] == name.split("__")[0]]
        if _suffix(name) in _INT_SUFFIXES and any(_suffix(n) in _FLOAT_SUFFIXES for n in siblings):
            continue
        record["fields"].append({"name": name, "type": _new(t, name, namespace)})
        have[name] = record["fields"][-1]
    return record


def _complete(avro, dtype: pl.DataType, name: str, namespace: str):
    """An inferred Avro type completed against its target dtype: each record's missing
    fields added, recursively. Its nullability and its types are kept as inferred."""
    if isinstance(avro, list):
        return [b if b == "null" else _complete(b, dtype, name, namespace) for b in avro]
    if isinstance(avro, str):
        return avro
    kind = avro["type"]
    if kind == "record":
        return _completed(avro, dtype)
    if kind == "map":
        values = dtype.inner.fields[1].dtype
        avro["values"] = _complete(avro["values"], values, f"{name}_values", namespace)
    elif kind == "array":
        avro["items"] = _complete(avro["items"], dtype.inner, name, namespace)
    return avro


def complete(avsc: str) -> str:
    """claims.avsc with every field of the target schema (claims_schema) it lacks added,
    nullable: a claim holding such a field fits it, and is decoded as genson's inference
    would give it (a field absent is null either way). Every field claims_schema has is
    then in the schema, but for a promoted integer field beside a float one (an integer
    latitude goes to `latitude__number`, as it does today)."""
    root = json.loads(avsc)
    (claims,) = [f for f in root["fields"] if f["name"] == "claims"]
    claims["type"] = _complete(claims["type"], claims_schema["claims"], "claims", root["namespace"])
    return json.dumps(root, separators=(",", ":"))


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
