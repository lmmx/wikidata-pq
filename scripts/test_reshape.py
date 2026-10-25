"""dump.reshape (compiled) against dump.entity_row (orjson and Python), line by line: the
same cells byte for byte, the same mismatched count and fields, or both raising. On lines
made to probe the edges (numbers orjson rewrites, escapes, the shapes the reshape changes,
the wrong types entity_row rejects, what the compiled scan leaves to entity_row), on
those lines mutated at random (seeded), and on any dump files given (`.json` lines, or a
`.json.bz2` such as bench/dumpslice.py makes), each as it is and as the dumps escape
their strings (every non-ASCII character as \\uXXXX, and / as \\/: bench/dumpslice.py
writes orjson's escaping instead).

    .venv/bin/python scripts/test_reshape.py [dump.json ...]
"""

import bz2
import json
import random
import sys

import orjson
import polars as pl
from wikidata_reshape import entity_rows

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


def snak_line(snak: str) -> bytes:
    return text('{"type":"item","id":"Q1","claims":{"P1":[{"mainsnak":%s,"type":"statement"}]}}' % snak)


DV = '"datavalue":{"value":"v\\/w","type":"string"}'
# What the compiled scan writes span by span, or leaves to entity_row (which reshapes or
# rejects it): either way the same row, or both raising
SCAN_EDGES = [
    # whitespace between tokens (none in the dumps)
    text(' { "type" : "item", "id":"Q1" , "labels" : { "en" : {"language":"en", "value":"x"} } ,"claims":{"P1":[ {"mainsnak":{"snaktype":"value"}} ]}}\t'),
    # escapes in keys, the id and a term's language; escapes everywhere a value is copied
    text('{"type":"item","id":"Q1","labels":{"e\\u006e":{"language":"en","value":"x"}}}'),
    text('{"type":"item","id":"Q\\u0031"}'),
    text('{"type":"item","id":"Q1","labels":{"en":{"language":"\\u0065n","value":"x"}}}'),
    text('{"type":"i\\/tem","id":"Q1","title":"\\u00e9\\ud83d\\ude00\\u0000\\u001F\\u007f\\u2028\\"\\\\\\b\\f\\n\\r\\t","sitelinks":{"enwiki":{"site":"enwiki","title":"A\\/B \\u00e9","badges":["Q\\u0031"]}},'
         '"aliases":{"en":[{"language":"en","value":"\\u00e9"},{"language":"en","value":"\\/"}]},'
         '"claims":{"P1":[{"mainsnak":{"snaktype":"value","property":"P1","datavalue":{"value":{"text":"\\u00e9\\/","language":"fr"},"type":"monolingual\\u0074ext"},"datatype":"x"},"type":"statement","id":"Q1$\\u00e9"}]}}'),
    # escapes orjson.loads refuses
    text('{"type":"item","id":"Q1","x":"\\x"}'),
    text('{"type":"item","id":"Q1","x":"\\u12"}'),
    text('{"type":"item","id":"Q1","x":"\\u12G4"}'),
    text('{"type":"item","id":"Q1","x":"\\ud800\\u0041"}'),
    text('{"type":"item","id":"Q1","x":"\\ud800\\n"}'),
    text('{"type":"item","id":"Q1","x":"\\udc00"}'),
    text('{"type":"item","id":"Q1","x":"\\ud800"}'),
    text('{"type":"item","id":"Q1","x":"\\'),
    # raw control characters, DEL, invalid UTF-8, a string left open
    b'{"type":"item","id":"Q1","x":"a\x01b"}',
    b'{"type":"item","id":"Q1","x":"a\x7fb\xc3\xa9"}',
    b'{"type":"item","id":"Q1","x":"\xff"}',
    b'{"type":"item","id":"Q1","x":"\xc0\xaf"}',
    b'{"type":"item","id":"Q1","x":"\xed\xa0\x80"}',
    b'{"type":"item","id":"Q1","x":"abc}',
    # numbers: JSON's grammar, and the integers at 64 bits' edges
    *(text('{"type":"item","id":"Q1","x":%s}' % n) for n in [
        "01", "-01", "1.", ".5", "-", "1e", "1e+", "+1", "1.5e+3", "0.000001", "1E-7", "-0", "-0.0e0", "0e5",
        "999999999999999999", "-999999999999999999", "9999999999999999999", "-9999999999999999999",
        "9223372036854775807", "9223372036854775808", "-9223372036854775808", "-9223372036854775809",
        "18446744073709551615", "18446744073709551616", "100000000000000000000", "1" * 400, "1e309", "-1e-400",
        "[1,-0,2.50,1e2,0]", "{\"a\":-0}",
    ]),
    # literals
    text('{"type":"item","id":"Q1","x":[true,false,null]}'),
    text('{"type":"item","id":"Q1","x":tru}'),
    text('{"type":"item","id":"Q1","x":True}'),
    text('{"type":"item","id":"Q1","x":nul}'),
    # content after the entity, an entity left open, empty, not an object
    text('{"type":"item","id":"Q1"} x'),
    text('{"type":"item","id":"Q1"}}'),
    text('{"type":"item","id":"Q1"'),
    text('{}'),
    text('[]'),
    text('{"id":"Q1"}'),
    # repeated keys (orjson keeps the last value, in the first's place)
    text('{"type":"item","id":"Q1","labels":{},"labels":{"en":{"language":"en","value":"x"}}}'),
    text('{"type":"item","id":"Q1","x":1,"x":2}'),
    text('{"type":"item","id":"Q1","id":"Q2"}'),
    text('{"type":"item","id":"Q1","labels":{"en":{"language":"en","value":"x","value":"y"}}}'),
    text('{"type":"item","id":"Q1","labels":{"en":{"language":"en","language":"en","value":"y"}}}'),
    snak_line('{"datavalue":{"value":1,"value":2,"type":"t"}}'),
    snak_line('{"datavalue":{"value":1},"datavalue":{"value":2}}'),
    text('{"type":"item","id":"Q1","claims":{"P1":[{"mainsnak":{},"mainsnak":{}}]}}'),
    # nesting: within what orjson writes, and past it
    text('{"type":"item","id":"Q1","x":%s1%s}' % ("[" * 120, "]" * 120)),
    text('{"type":"item","id":"Q1","x":%s1%s}' % ("[" * 200, "]" * 200)),
    text('{"type":"item","id":"Q1","x":%s1%s}' % ("[" * 300, "]" * 300)),
    text('{"type":"item","id":"Q1","x":%s}' % ('{"a":' * 150 + "1" + "}" * 150)),
    # a snak's datavalue first, between, last, alone; without value or type; with other fields; not an object
    snak_line('{%s,"snaktype":"value","property":"P1"}' % DV),
    snak_line('{"snaktype":"value",%s,"property":"P1"}' % DV),
    snak_line('{"snaktype":"value","property":"P1",%s}' % DV),
    snak_line('{%s}' % DV),
    snak_line('{"datavalue":{}}'),
    snak_line('{"datavalue":{"type":"t","other":[1,{}],"value":{"a":"\\u00e9"}},"x":1}'),
    snak_line('{"datavalue":{"value":"\\ud800","type":"t"}}'),
    snak_line('{"datavalue":"s"}'),
    snak_line('{"datavalue":null}'),
    snak_line('{"datavalue":[]}'),
    snak_line('{"datavalue_type":"t","snaktype":"value"}'),
    snak_line('{"datavalue_type":"t",%s}' % DV),
    snak_line('{}'),
    snak_line('"s"'),
    snak_line('[]'),
    # statements: qualifiers and references in every shape entity_row takes or refuses
    text('{"type":"item","id":"Q1","claims":{"P1":[{"mainsnak":{},"qualifiers":"q","references":[{},{"snaks":[]},{"hash":"h"},{"snaks":{"P2":[{%s}]},"snaks-order":["P2"]}]}]}}' % DV),
    text('{"type":"item","id":"Q1","claims":{"P1":[{"mainsnak":{},"qualifiers":{"P2":[{%s},{"snaktype":"novalue"}],"P3":[]}}]}}' % DV),
    text('{"type":"item","id":"Q1","claims":{"P1":[{"mainsnak":{},"qualifiers":{"P2":{}}}]}}'),
    text('{"type":"item","id":"Q1","claims":{"P1":[{"mainsnak":{},"references":""}]}}'),
    text('{"type":"item","id":"Q1","claims":{"P1":[{"mainsnak":{},"references":{}}]}}'),
    text('{"type":"item","id":"Q1","claims":{"P1":[{"mainsnak":{},"references":[{"snaks":{"P2":"s"}}]}]}}'),
    text('{"type":"item","id":"Q1","claims":{"P1":[{"mainsnak":{},"references":[{"snaks":{},"snaks":{}}]}]}}'),
    text('{"type":"item","id":"Q1","claims":{"P1":[{"mainsnak":"m"}]}}'),
    text('{"type":"item","id":"Q1","claims":{"P1":[]},"sitelinks":[],"labels":"x","descriptions":7,"aliases":null}'),
    text('{"type":"item","id":"Q1","claims":{"P1":{}}}'),
    text('{"type":"item","id":"Q1","claims":{"P1":"s"}}'),
    # terms: value first, extra fields, a language not a string or missing
    text('{"type":"item","id":"Q1","labels":{"en":{"value":"x","language":"en"},"de":{"value":"y","language":"de","x":[1]},"fr":{"value":"z","language":7},"it":{"value":"w"}}}'),
    text('{"type":"item","id":"Q1","aliases":{"en":{}}}'),
    text('{"type":"item","id":"Q1","aliases":{"en":"s"}}'),
    text('{"type":"item","id":"Q1","aliases":{"en":[]}}'),
    text('{"type":"item","id":"Q1","aliases":{"en":[{"value":"a"},"b"]}}'),
    # sitelinks: a link not an object
    text('{"type":"item","id":"Q1","sitelinks":{"enwiki":["a"]}}'),
    text('{"type":"item","id":"Q1","sitelinks":{"enwiki":{}}}'),
]


def mutants(lines: list[bytes], n: int) -> list[bytes]:
    """`n` lines, each one of `lines` with a byte or two replaced, inserted or deleted
    (seeded), from the bytes that matter to the scan"""
    rng = random.Random(1)
    alphabet = list(b'"\\{}[],:0-9eE.+- u/abfnrtx\x00\x1f\x7f') + [0xc3, 0xa9, 0xed, 0xff]
    out = []
    for _ in range(n):
        line = bytearray(rng.choice(lines))
        for _ in range(rng.choice((1, 1, 2))):
            i = rng.randrange(len(line) + 1)
            op = rng.randrange(3)
            if op == 0 and i < len(line):
                line[i] = rng.choice(alphabet)
            elif op == 1:
                line.insert(i, rng.choice(alphabet))
            elif i < len(line):
                del line[i]
        out.append(bytes(line))
    return out


def dump_escaped(line: bytes) -> bytes:
    """A line as the dumps escape its strings: each non-ASCII character as \\uXXXX, and
    / as \\/ (a / is only ever in a string)"""
    return json.dumps(json.loads(line), ensure_ascii=True, separators=(",", ":")).replace("/", "\\/").encode()


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
    one, three = entity_rows(lines), entity_rows(lines, threads=3)
    threads = [[bytes(b) for b in col] for col in one[0]] == [[bytes(b) for b in col] for col in three[0]] and one[1:] == three[1:]
    print(f"{what}: {len(lines):,} entities, {differing} rows differing, mismatched {got_mismatched} "
          f"(entity_row {mismatched[0]}), fields {'equal' if got_fields == fields else 'DIFFER'}, "
          f"{len(one[4]):,} left to entity_row, on 3 threads {'the same' if threads else 'DIFFERENT'}")
    return 0 if ok and threads else 1


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
failed += check(SCAN_EDGES, "scan edge")
failed += check([dump_escaped(line) for line in EDGES[:9]], "edge, as the dumps escape it")
fuzzed = mutants([line for line in EDGES + SCAN_EDGES if len(line) < 2000], 20_000)
failed += check(fuzzed, "mutant")
print(f"{len(EDGES) + len(SCAN_EDGES)} edges and {len(fuzzed):,} mutants checked")
for path in sys.argv[1:]:
    lines = dump_lines(path)
    failed += whole(lines, path)
    failed += whole([dump_escaped(line) for line in lines], f"{path}, as the dumps escape it")
print("FAILED" if failed else "all passed")
sys.exit(1 if failed else 0)
