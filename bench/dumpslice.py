"""The benchmark's chunk files as a slice of a Wikidata JSON dump: each row turned back
into the dump's entity line (dump.entity_row undone), the lines in the files' order, the
dump's own framing (`[`, an entity a line, comma-separated, `]`), compressed by bzip2 as
the dump is. dump.split then reshapes each line into exactly the row it came from, so
splitting and routing the slice reproduces the files (pipeline.split_stage checks it).
Each line is written as the dumps write theirs (Wikibase's PHP json_encode), not as orjson
does: every non-ASCII character escaped as \\uXXXX (a surrogate pair beyond the BMP), `/`
as `\\/`, and a float in PHP's shortest form, exponential below 1e-4 and from 1e17
(`2.77778e-5`, `1.0e+20`, where orjson writes `0.0000277778` and `1e+20`); since bench
11.0.0 (before it, orjson's escaping: each line's bytes the split's reshape copies were
the dumps' less their escapes, about 13% of a line's bytes).
The same lines are also written gzip-compressed beside it (OUT.json.gz), at level 9 as
the gz dump is: the first 48 MiB of wikidata-20261020-all.json.gz decompress to 347 MB,
which gzip -9 compresses to within 0.01% of those 48 MiB (pigz -9, used when installed,
to within 0.3%).

The same lines in orjson's form, bench 10.0.0's slice, are written beside them as
OUT.orjson.json.bz2 (orjson_path), for the split's figures on both, on the same runs.

Usage: dumpslice.py OUT.json.bz2 FILE.parquet...
"""

import bz2
import gzip
import hashlib
import json
import re
import shutil
import subprocess
import sys
from decimal import Decimal
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
    """A chunk row as the dump's entity line, as the dumps escape it (dump_escaped)."""
    return dump_escaped(orjson_line(row))


def orjson_line(row: dict) -> bytes:
    """A chunk row as the dump's entity line in orjson's form: dump.entity_row undone."""
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


# A float, its PHP form held in a string while the line is re-encoded (dump_escaped): a NUL
# and F, then the form, then a NUL, which no cell of the data holds (the split's parity
# would show it)
_FLOAT = re.compile(r'"\\u0000F([-+.0-9e]+)\\u0000"')


def php_float(token: str) -> str:
    """A JSON float as PHP's json_encode writes it (serialize_precision -1: php_gcvt in mode
    0): its shortest digits, in exponential form (`d.ddde-5`, `d.0e+20`) when its decimal
    point is more than 3 places left of its first digit or more than 17 right of it, else
    in fixed form. An integral float in fixed form, which PHP would write as an integer
    (`52`, read back as an int, not the float the cell holds), is left as orjson wrote it."""
    x = float(token)
    if x != x or x in (float("inf"), float("-inf")) or x == int(x) and abs(x) < 1e17:
        return token
    digits, exp = Decimal(repr(abs(x))).normalize().as_tuple()[1:]
    d = "".join(map(str, digits))
    point = len(d) + exp  # (the decimal point's place after the first digit's: x = 0.d × 10^point)
    sign = "-" if x < 0 else ""
    if point < -3 or point > 17:
        e = point - 1
        return f"{sign}{d[0]}.{d[1:] or '0'}e{'-' if e < 0 else '+'}{abs(e)}"
    if point <= 0:
        return f"{sign}0.{'0' * -point}{d}"
    return f"{sign}{d[:point]}.{d[point:]}"


def dump_escaped(line: bytes) -> bytes:
    """An orjson line as the dumps write it: its strings with every non-ASCII character as
    \\uXXXX and `/` as `\\/`, its floats as PHP writes them (php_float), its keys in
    their order, no whitespace. (`/` is only ever in a string: no JSON token holds one.)"""
    entity = json.loads(line, parse_float=lambda t: f"\0F{php_float(t)}\0")
    text = json.dumps(entity, ensure_ascii=True, separators=(",", ":")).replace("/", "\\/")
    return _FLOAT.sub(r"\1", text).encode()


def gz_path(dst: Path) -> Path:
    """The gz slice beside the bz2 slice `dst`."""
    return dst.with_suffix(".gz")


def orjson_path(dst: Path) -> Path:
    """Beside the bz2 slice `dst`, the same lines in orjson's form, bz2 too: bench
    10.0.0's slice, which the split also runs on, apart (pipeline.split_gz), so the
    change to the dumps' escaping shows on the same runs."""
    return dst.with_name(dst.name.removesuffix(".json.bz2") + ".orjson.json.bz2")


def make(dst: Path, files: list[str]) -> None:
    """Write the slice as bz2 at `dst` and as gz beside it (gz_path), and in orjson's form
    as bz2 (orjson_path), its lines streamed to lbzip2 and pigz (all cores; bz2 and gzip,
    a core each, without them), a file at a time, so the slice is never held whole."""
    fallback = {"lbzip2": lambda f: bz2.BZ2File(f, "wb", compresslevel=9),
                "pigz": lambda f: gzip.GzipFile(fileobj=f, mode="wb", compresslevel=9, mtime=0)}
    sinks = []
    for path, tool, form in ((dst, "lbzip2", "dump"), (gz_path(dst), "pigz", "dump"),
                             (orjson_path(dst), "lbzip2", "orjson")):
        out = open(path.with_name(path.name + ".tmp"), "wb")
        proc = subprocess.Popen([tool, "-9", "-c"], stdin=subprocess.PIPE, stdout=out) if shutil.which(tool) else None
        sinks.append((path, out, proc, proc.stdin if proc else fallback[tool](out), form))

    def write(b: dict[str, bytes]) -> None:
        for *_, sink, form in sinks:
            sink.write(b[form])

    write({"dump": b"[\n", "orjson": b"[\n"})
    sep = b""
    for f in files:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=1000):
            for row in batch.to_pylist():
                line = orjson_line(row)
                write({"dump": sep + dump_escaped(line), "orjson": sep + line})
                sep = b",\n"
    write({"dump": b"\n]\n", "orjson": b"\n]\n"})
    for path, out, proc, sink, _ in sinks:
        sink.close()
        if proc and proc.wait():
            raise RuntimeError(f"{proc.args[0]} exited with {proc.returncode}")
        out.close()
        path.with_name(path.name + ".tmp").replace(path)


def sharded_path(dst: Path) -> Path:
    """Beside the bz2 slice `dst`, its entities as a dump of 8 shards (`sharded`)."""
    return dst.with_name(dst.name.removesuffix(".json.bz2") + ".sharded.json.bz2")


def _shard(entity_id: str, factor: int) -> int:
    """Wikibase's dump shard of an id (DumpGenerator::idMatchesShard): its sha1's first 4
    bytes mod the sharding factor."""
    return int.from_bytes(hashlib.sha1(entity_id.encode()).digest()[:4], "big") % factor


def _lines(src: Path):
    """The entity lines of a bz2 slice (as `make` writes it), without their commas."""
    proc = subprocess.Popen(["lbzip2", "-dc", str(src)] if shutil.which("lbzip2") else ["bzip2", "-dc", str(src)],
                            stdout=subprocess.PIPE)
    for line in proc.stdout:
        line = line.rstrip(b",\n")
        if line.startswith(b"{"):
            yield line
    if proc.wait():
        raise RuntimeError(f"{proc.args[0]} exited with {proc.returncode} on {src}")


_ID = re.compile(rb'"id":"([A-Z])(\d+)"')


def sharded(src: Path, dst: Path, renumber: bool = True, factor: int = 8, start: int = 10_000) -> dict:
    """The slice at `src` as a dump of `factor` shards one after another (bench 17.13.0,
    for the split's sweep reader; docs/journal/2026-10-26-progressive-finalise.md), one
    bz2 stream written by lbzip2 (bzip2 without it), as Wikimedia's dump job writes its
    shards: each shard's lines in the slice's order, so each file's rows keep theirs.
    The slice's own ids put each of its files in one shard (c0 in shard 0, r50 in 3, s30
    in 2, s90 in 7), so with `renumber` its Q ids are new: the i-th Q line goes to shard
    i mod `factor` and takes the next number from `start` up that sha1 puts in that shard
    (its `"id"` and `"title"`), so the shards are sweeps over one id range, as the dump's
    are. Its rows are otherwise the slice's. Returns the lines in each shard."""
    nexts = {t: start for t in range(factor)}

    def number(t: int) -> int:
        n = nexts[t]
        while _shard(f"Q{n}", factor) != t:
            n += 1
        nexts[t] = n + 1
        return n

    tool = ["lbzip2", "-9", "-c"] if shutil.which("lbzip2") else ["bzip2", "-9", "-c"]
    tmp = dst.with_name(dst.name + ".tmp")
    counts = [0] * factor
    with open(tmp, "wb") as out:
        proc = subprocess.Popen(tool, stdin=subprocess.PIPE, stdout=out)
        proc.stdin.write(b"[\n")
        sep = b""
        for t in range(factor):  # (a pass over the slice a shard: it is never held whole)
            q = 0
            for line in _lines(src):
                m = _ID.search(line, 0, 256)
                letter, old = m.group(1), m.group(2)
                if renumber and letter == b"Q":
                    mine, q = q % factor == t, q + 1
                    if not mine:
                        continue
                    new = b"Q%d" % number(t)
                    line = line.replace(b'"id":"Q' + old + b'"', b'"id":"' + new + b'"', 1).replace(
                        b'"title":"Q' + old + b'"', b'"title":"' + new + b'"', 1)
                elif _shard((letter + old).decode(), factor) != t:
                    continue
                proc.stdin.write(sep + line)
                sep = b",\n"
                counts[t] += 1
        proc.stdin.write(b"\n]\n")
        proc.stdin.close()
        if proc.wait():
            raise RuntimeError(f"{tool[0]} exited with {proc.returncode}")
    tmp.replace(dst)
    return {"lines": counts}


if __name__ == "__main__":
    make(Path(sys.argv[1]), sys.argv[2:])
