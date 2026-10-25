"""The benchmark's chunk files as a slice of a Wikidata JSON dump: each row turned back
into the dump's entity line (dump.entity_row undone), the lines in the files' order, the
dump's own framing (`[`, an entity a line, comma-separated, `]`), compressed by bzip2 as
the dump is. dump.split then reshapes each line into exactly the row it came from, so
splitting and routing the slice reproduces the files (pipeline.split_stage checks it).
The same lines are also written gzip-compressed beside it (OUT.json.gz), at level 9 as
the gz dump is: the first 48 MiB of wikidata-20261020-all.json.gz decompress to 347 MB,
which gzip -9 compresses to within 0.01% of those 48 MiB (pigz -9, used when installed,
to within 0.3%).

Usage: dumpslice.py OUT.json.bz2 FILE.parquet...
"""

import bz2
import gzip
import shutil
import subprocess
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


def gz_path(dst: Path) -> Path:
    """The gz slice beside the bz2 slice `dst`."""
    return dst.with_suffix(".gz")


def make(dst: Path, files: list[str]) -> None:
    """Write the slice as bz2 at `dst` and as gz beside it (gz_path), its lines streamed
    to lbzip2 and pigz (all cores; bz2 and gzip, a core each, without them), a file at a
    time, so the slice is never held whole."""
    fallback = {"lbzip2": lambda f: bz2.BZ2File(f, "wb", compresslevel=9),
                "pigz": lambda f: gzip.GzipFile(fileobj=f, mode="wb", compresslevel=9, mtime=0)}
    sinks = []
    for path, tool in ((dst, "lbzip2"), (gz_path(dst), "pigz")):
        out = open(path.with_name(path.name + ".tmp"), "wb")
        proc = subprocess.Popen([tool, "-9", "-c"], stdin=subprocess.PIPE, stdout=out) if shutil.which(tool) else None
        sinks.append((path, out, proc, proc.stdin if proc else fallback[tool](out)))

    def write(b: bytes) -> None:
        for *_, sink in sinks:
            sink.write(b)

    write(b"[\n")
    first = True
    for f in files:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=1000):
            for row in batch.to_pylist():
                write((b"" if first else b",\n") + entity_line(row))
                first = False
    write(b"\n]\n")
    for path, out, proc, sink in sinks:
        sink.close()
        if proc and proc.wait():
            raise RuntimeError(f"{proc.args[0]} exited with {proc.returncode}")
        out.close()
        path.with_name(path.name + ".tmp").replace(path)


if __name__ == "__main__":
    make(Path(sys.argv[1]), sys.argv[2:])
