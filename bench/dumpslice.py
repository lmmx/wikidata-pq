"""The benchmark's chunk files as a slice of a Wikidata JSON dump: each row turned back
into the dump's entity line (dump.entity_row undone), the lines in the files' order, the
dump's own framing (`[`, an entity a line, comma-separated, `]`), compressed by bzip2 as
the dump is. dump.split then reshapes each line into exactly the row it came from, so
splitting and routing the slice reproduces the files (pipeline.split_stage checks it).

Usage: dumpslice.py OUT.json.bz2 FILE.parquet...
"""

import bz2
import sys
from pathlib import Path

import orjson
import pyarrow.parquet as pq


def _snak(snak: dict) -> dict:
    """dump._snak undone: the datavalue's value and type as its `datavalue`."""
    out = {k: v for k, v in snak.items() if k not in ("datavalue", "datavalue_type")}
    if "datavalue" in snak:
        out["datavalue"] = {"value": snak["datavalue"], "type": snak.get("datavalue_type")}
    return out


def _snaks(by_property: dict) -> dict:
    return {p: [_snak(s) for s in snaks] for p, snaks in by_property.items()}


def _statement(statement: dict) -> dict:
    out = dict(statement)
    out["mainsnak"] = _snak(statement["mainsnak"])
    if "qualifiers" in statement:
        out["qualifiers"] = _snaks(statement["qualifiers"])
    if "references" in statement:
        out["references"] = [{**ref, "snaks": _snaks(ref.get("snaks") or {})} for ref in statement["references"]]
    return out


def entity_line(row: dict) -> bytes:
    """A chunk row as the dump's entity line: dump.entity_row undone."""
    terms = lambda m: {lang: {"language": lang, "value": v} for lang, v in m.items()}  # noqa: E731
    aliases = {lang: [{"language": lang, "value": v} for v in vs] for lang, vs in orjson.loads(row["aliases"]).items()}
    own = orjson.loads(row["entity"])
    entity = {
        **({"type": own.pop("type")} if "type" in own else {}),
        "id": row["id"],
        "labels": terms(orjson.loads(row["labels"])),
        "descriptions": terms(orjson.loads(row["descriptions"])),
        "aliases": aliases,
        "claims": {p: [_statement(s) for s in ss] for p, ss in orjson.loads(row["claims"]).items()},
        "sitelinks": orjson.loads(row["sitelinks"]),
        **own,
    }
    return orjson.dumps(entity)


def make(dst: Path, files: list[str]) -> None:
    lines = [entity_line(r) for f in files for r in pq.read_table(f).to_pylist()]
    body = b"[\n" + b",\n".join(lines) + b"\n]\n"
    tmp = dst.with_suffix(".tmp")
    tmp.write_bytes(bz2.compress(body, 9))
    tmp.replace(dst)


if __name__ == "__main__":
    make(Path(sys.argv[1]), sys.argv[2:])
