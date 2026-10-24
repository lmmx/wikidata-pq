"""dump.reshape (compiled) against dump.entity_row (orjson and Python), line by line: the
same cells byte for byte, the same mismatched count and fields, or both raising. On lines
made to probe the edges (numbers orjson rewrites, escapes, the shapes the reshape changes,
the wrong types entity_row rejects), and on any dump files given (`.json` lines, or a
`.json.bz2` such as bench/dumpslice.py makes).

    .venv/bin/python scripts/test_reshape.py [dump.json ...]
"""

import bz2
import random
import sys

import orjson
import polars as pl

from wikidata import dump

SNAK = {"snaktype": "value", "property": "P31", "hash": "h"}


def entity(**parts) -> dict:
    return {"type": "item", "id": "Q1", **parts, "lastrevid": 7}


def statement(**over) -> dict:
    return {
        "mainsnak": {**SNAK, "datavalue": {"value": {"entity-type": "item", "numeric-id": 5, "id": "Q5"}, "type": "wikibase-entityid"}, "datatype": "wikibase-item"},
        "type": "statement",
        "id": "Q1$x",
        "rank": "normal",
        **over,
    }


def text(raw: str) -> bytes:
    return raw.encode()


def floats() -> bytes:
    """An entity of doubles from across their range, written as Python, PHP and the dumps
    write them (shortest digits, a trailing .0, an upper-case E, an explicit +)"""
    rng = random.Random(0)
    texts = []
    for _ in range(5000):
        x = rng.uniform(1, 10) * 10.0 ** rng.randint(-320, 307) * rng.choice((1, -1))
        r = repr(x)
        texts += [r, r.replace("e", "E"), r.replace("e+", "e"), f"{x:.17g}"]
    texts += ["1.0e-5", "1.0e-6", "1.0e+15", "1e15", "1e16", "9007199254740993", "123456789012345678901234567890"]
    return f'{{"type":"item","id":"Q1","x":[{",".join(texts)}]}}'.encode()


EDGES = [
    # numbers: the floats orjson rewrites, big and small, ints past 64 bits signed
    text('{"type":"item","id":"Q1","claims":{"P625":[{"mainsnak":{"snaktype":"value","property":"P625","datavalue":{"value":{"latitude":1.0e-5,"longitude":-1.0e-8,"altitude":null,"precision":2.77778e-5,"globe":"x"},"type":"globecoordinate"}},"type":"statement"}]}}'),
    floats(),
    text('{"type":"item","id":"Q1","x":[0,-0,1,-1,9223372036854775807,9223372036854775808,18446744073709551615,-9223372036854775808,0.1,1e300,-0.0,1E2,5e-324,123456789.123456789,1.7976931348623157e308]}'),
    # strings: PHP-style escapes, control characters, DEL, U+2028, an emoji's surrogate pair, quotes
    text('{"type":"item","id":"Q1","labels":{"en":{"language":"en","value":"a\\/b \\u00e9 \\u0000\\u0001\\u001f\\u007f\\u2028 \\ud83d\\ude00 \\"q\\" \\\\ \\t\\n\\r\\b\\f"}}}'),
    # the shapes reshaped: empty maps written as [], terms and aliases, a mismatched term
    orjson.dumps(entity(labels=[], descriptions=[], aliases=[], sitelinks=[], claims=[])),
    orjson.dumps(entity(labels={"en": {"language": "en", "value": "x"}, "de": {"language": "en", "value": "y"}, "fr": {"language": "fr", "value": "z", "extra": 1}},
                        aliases={"en": [{"language": "en", "value": "a"}, {"language": "de", "value": "b"}]})),
    orjson.dumps(entity(sitelinks={"enwiki": {"site": "enwiki", "title": "T", "badges": []}, "dewiki": {"title": "U", "site": "dewiki", "url": "u"}})),
    # statements: qualifiers ([] and a map), references with and without snaks, snaks without datavalues, datavalues without value or type
    orjson.dumps(entity(claims={"P31": [statement(qualifiers=[], references=[{"hash": "r", "snaks": [], "snaks-order": []}, {"hash": "s"}]),
                                        statement(**{"qualifiers": {"P2": [{**SNAK, "datavalue": {"value": "v", "type": "string"}}, {"snaktype": "novalue", "property": "P2"}]}, "qualifiers-order": ["P2"]}),
                                        statement(mainsnak={**SNAK, "datavalue": {}}),
                                        statement(mainsnak={"snaktype": "somevalue", "property": "P31"})]})),
    # an entity whose own fields come before and after its parts
    orjson.dumps({"pageid": 1, "ns": 0, "title": "Q1", "id": "Q1", "type": "item", "labels": {}, "modified": "2026-01-01T00:00:00Z"}),
    # what entity_row rejects (so both must raise) or reshapes in place (left to it)
    orjson.dumps(entity(labels={"en": "bare"})),
    orjson.dumps(entity(labels={"en": {"language": "en"}})),
    orjson.dumps(entity(claims={"P31": [{"type": "statement"}]})),
    orjson.dumps(entity(claims={"P31": [statement(mainsnak={**SNAK, "datavalue": None})]})),
    orjson.dumps(entity(claims={"P31": [statement(references=["r"])]})),
    orjson.dumps(entity(claims={"P31": [statement(mainsnak={**SNAK, "datavalue_type": "old", "datavalue": {"value": 1, "type": "t"}})]})),
    orjson.dumps(entity(claims={"P31": {"not": "a list"}})),
    orjson.dumps(entity(sitelinks={"enwiki": "a string"})),
    text('{"type":"item","id":"Q1","labels":{"en":{"language":"en","value":"\\ud800"}}}'),
    text('{"type":"item","id":"Q1","x":1e400}'),
    text('{"type":"item","id":7}'),
    text('{"type":"item","id":"Q1",}'),
]


def python(line: bytes):
    """The row as write_chunk made it before dump.reshape: entity_row's, in a String frame"""
    mismatched, fields = [0], {"entity": set(), "sitelink": set()}
    row = dump.entity_row(orjson.loads(line), mismatched, fields)
    frame = pl.DataFrame([row], schema={c: pl.String for c in dump.COLUMNS}, orient="row")
    return frame.row(0), mismatched[0], fields


def compiled(line: bytes):
    frame, mismatched, fields = dump.reshape([line])
    return frame.row(0), mismatched, fields


def check(lines: list[bytes], what: str) -> int:
    failed = 0
    for i, line in enumerate(lines):
        try:
            want = python(line)
        except Exception as e:
            want = type(e)
        try:
            got = compiled(line)
        except Exception as e:
            got = type(e)
        same = (isinstance(want, type) and isinstance(got, type)) or want == got
        if not same:
            failed += 1
            print(f"FAIL {what} line {i}: {line[:120]!r}\n  entity_row: {str(want)[:300]}\n  compiled:   {str(got)[:300]}")
    return failed


def whole(lines: list[bytes], what: str) -> int:
    """A file's lines as one chunk: dump.reshape's frame against entity_row's rows"""
    mismatched, fields = [0], {"entity": set(), "sitelink": set()}
    rows = [dump.entity_row(orjson.loads(line), mismatched, fields) for line in lines]
    frame, got_mismatched, got_fields = dump.reshape(lines)
    differing = sum(a != b for a, b in zip(frame.iter_rows(), rows)) + abs(frame.height - len(rows))
    ok = not differing and got_mismatched == mismatched[0] and got_fields == fields
    print(f"{what}: {len(lines):,} entities, {differing} rows differing, mismatched {got_mismatched} "
          f"(entity_row {mismatched[0]}), fields {'equal' if got_fields == fields else 'DIFFER'}")
    return 0 if ok else 1


def dump_lines(path: str) -> list[bytes]:
    opener = bz2.open if path.endswith(".bz2") else open
    with opener(path, "rb") as f:
        lines = [line.rstrip(b",\r\n") for line in f if line.startswith(b"{")]
    try:
        orjson.loads(lines[-1])
    except orjson.JSONDecodeError:
        lines = lines[:-1]  # a slice cut short
    return lines


# the edges entity_row reshapes, as one chunk (the others raise)
failed = check(EDGES, "edge") + whole(EDGES[:9], "the edges as a chunk")
for path in sys.argv[1:]:
    failed += whole(dump_lines(path), path)
print("FAILED" if failed else "all passed")
sys.exit(1 if failed else 0)
