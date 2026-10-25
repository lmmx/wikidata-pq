"""The claims built from the dump lines in the reshape's scan (claims_build.reshape_built,
reshape/src/claims_arrow.rs RawBuild) against the text path they replace: the reshape's
claims text (entity_rows, with dump.entity_row for the lines it leaves), routed by
scholarly.is_scholarly's regex, and decoded part by part (claims_build.claims_arrays).
For every chunk, either the line path leaves it to the text path (reshape_built gives
None), or:

- the text path reshapes every line itself (none left to entity_row);
- every other column is the text path's, byte for byte;
- each row's set is the one the regex gives it;
- each set's claims are the arrays the text path decodes from its text, and the text
  path decodes them all (no part of a built chunk is a misfit there).

The dump files' chunks are checked in the split's batches of claims, and the first in
batches of 64 KiB.

On each dump file given (`.json` lines, or a `.json.bz2` such as bench/dumpslice.py makes),
cut into chunks; on lines made to probe the edges, the routing's above all (each a chunk
of its own, then all as one chunk); and on lines of claims mutated at random (seeded),
each a chunk of its own.

    WIKIDATA_RELEASE=20261012 .venv/bin/python scripts/test_claims_from_line.py [dump.json.bz2 ...]
"""

import bz2
import collections
import os
import random
import sys

os.environ.setdefault("WIKIDATA_RELEASE", "20261012")

import numpy as np  # noqa: E402
import orjson  # noqa: E402
import polars as pl  # noqa: E402
import pyarrow as pa  # noqa: E402
from wikidata_reshape import entity_rows, entity_rows_built  # noqa: E402

from wikidata import claims_build, dump  # noqa: E402
from wikidata.scholarly import SCHOLARLY_CLASSES, is_scholarly  # noqa: E402

COLUMNS = dump.COLUMNS
CLAIMS = COLUMNS.index("claims")


def text_path(lines: list[bytes]):
    """The text path: (its columns, as bytes per cell; the regex's mask; each set's claims
    table or the misfit, by set name; how many lines were left to entity_row)."""
    columns, _, _, _, left = entity_rows(lines)
    arrays = {
        name: pa.Array.from_buffers(pa.large_string(), len(lines), [None, pa.py_buffer(o), pa.py_buffer(d)])
        for name, (d, o) in zip(COLUMNS, columns)
    }
    frame = pl.from_arrow(pa.table(arrays))
    if left:
        counted = [0]
        fields = {"entity": set(), "sitelink": set()}
        rows = [dump.entity_row(orjson.loads(lines[i]), counted, fields) for i in left]
        rows = pl.DataFrame(rows, schema={c: pl.String for c in COLUMNS}, orient="row")
        frame = frame.with_columns(frame[c].scatter(left, rows[c]) for c in COLUMNS)
    mask = frame.select(is_scholarly(pl.col("claims"))).to_series()
    text = claims_build.text_of(frame)
    sets = {}
    for name, m in (("scholar", mask), ("main", ~mask)):
        if not m.any():
            continue
        rows = pa.array(m.arg_true().cast(pl.Int64).to_numpy())
        table, why = claims_build.claims_arrays(frame.get_column("id").filter(m), text, rows)
        sets[name] = table.column("claims").combine_chunks() if table is not None else why
    cells = [frame.get_column(c).to_list() for c in COLUMNS]
    return cells, mask.to_list(), sets, len(left)


def line_path(lines: list[bytes]):
    """The line path: None (the text path's), or (its columns' cells, its flags, each
    set's claims, by set name)."""
    built = claims_build.reshape_built(lines)
    if built is None:
        return None
    (columns, _, _, _, left), b = built
    assert not left
    cells = []
    for d, o in columns:
        offsets = np.frombuffer(o, dtype=np.int64)
        data = bytes(d)
        cells.append([data[offsets[i]:offsets[i + 1]].decode() for i in range(len(offsets) - 1)])
    return cells, b.scholarly.to_list(), b.claims


def why_left(lines: list[bytes]) -> str:
    return entity_rows_built(lines, claims_build.PLAN, list(SCHOLARLY_CLASSES), claims_build.BATCH_TEXT_BYTES)[2]


def check(lines: list[bytes], what: str, stats: collections.Counter) -> int:
    """0 if the line path is the text path's on `lines` (or leaves them to it), else 1."""
    try:
        built = line_path(lines)
    except Exception as e:  # noqa: BLE001
        print(f"FAIL {what}: the line path raised {e!r}")
        return 1
    if built is None:
        stats[f"text path: {why_left(lines)}"] += 1
        return 0
    try:
        cells, mask, sets, left = text_path(lines)
    except Exception as e:  # noqa: BLE001 (entity_row raising, say)
        print(f"FAIL {what}: built, but the text path raised {e!r}")
        return 1
    stats["built"] += 1
    stats["rows"] += len(lines)
    stats["scholarly rows"] += sum(mask)
    bcells, bmask, bclaims = built
    if left:
        print(f"FAIL {what}: built, but the text path left {left} lines to entity_row")
        return 1
    for c, name in enumerate(COLUMNS):
        if c != CLAIMS and bcells[c] != cells[c]:
            print(f"FAIL {what}: column {name} differs")
            return 1
    if bmask != mask:
        diff = [i for i, (a, b) in enumerate(zip(bmask, mask)) if a != b]
        print(f"FAIL {what}: the sets of rows {diff[:5]} differ from the regex's")
        return 1
    for name in ("main", "scholar"):
        n = sum(m == (name == "scholar") for m in mask)
        got = bclaims[name].combine_chunks()
        if len(got) != n:
            print(f"FAIL {what}: {name} has {len(got)} rows, not {n}")
            return 1
        if not n:
            continue
        want = sets[name]
        if isinstance(want, str):
            print(f"FAIL {what}: built {name}, which the text path leaves as JSON ({want})")
            return 1
        if not (got.type == want.type and got.equals(want)):
            print(f"FAIL {what}: {name}'s claims differ from the text path's")
            return 1
    return 0


# Routing and reshaping at the edges. A statement whose mainsnak's value is `value`
SNAK = '{"snaktype":"value","property":"%s","datatype":"wikibase-item","datavalue":%s}'
ITEM = '{"value":{"entity-type":"item","numeric-id":%s,"id":"%s"},"type":"wikibase-entityid"}'


def line(claims: str, qid: str = "Q1") -> bytes:
    return ('{"type":"item","id":"%s","labels":{"en":{"language":"en","value":"x"}},"claims":%s,"lastrevid":7}'
            % (qid, claims)).encode()


def p31(value: str, prop: str = "P31", extra: str = "") -> bytes:
    return line('{"%s":[{"mainsnak":%s,"type":"statement","id":"Q1$a","rank":"normal"%s}]}'
                % (prop, SNAK % (prop, value), extra))


SCHOLARLY = ITEM % ("13442814", "Q13442814")
EDGES = [
    p31(SCHOLARLY),
    p31(ITEM % ("5", "Q5")),
    p31(SCHOLARLY, prop="P279"),
    # the value id's forms: escaped, past 18 digits, a fraction, negative, other digits
    p31(ITEM % ("13442814", "Q\\u00313442814")),
    p31(ITEM % ("13442814", "Q13442814\\u0031")),
    p31(ITEM % ("1344281400000000000000", "Q13442814")),
    p31(ITEM % ("13442814.0", "Q13442814")),
    p31(ITEM % ("-13442814", "Q13442814")),
    p31(ITEM % ("13442814", "Q١٢")),
    p31(ITEM % ("13442814", "Q13442814x")),
    # the value's members in another order, or another before them
    p31('{"value":{"numeric-id":13442814,"entity-type":"item","id":"Q13442814"},"type":"wikibase-entityid"}'),
    p31('{"value":{"id":"Q13442814","entity-type":"item","numeric-id":13442814},"type":"wikibase-entityid"}'),
    p31('{"type":"wikibase-entityid","value":{"entity-type":"item","numeric-id":13442814,"id":"Q13442814"}}'),
    p31('{"value":{"entity-type":"\\u0069tem","numeric-id":13442814,"id":"Q13442814"},"type":"wikibase-entityid"}'),
    p31('{"value":{"entity-type":"property","numeric-id":13442814,"id":"Q13442814"},"type":"wikibase-entityid"}'),
    # the snak's other fields: braces, escapes, the property escaped, the datavalue first
    line('{"P31":[{"mainsnak":{"snaktype":"val{ue","property":"P31","datatype":"wikibase-item","datavalue":%s},"type":"statement"}]}' % SCHOLARLY),
    line('{"P31":[{"mainsnak":{"snaktype":"value","property":"P31","datatype":"x}y","datavalue":%s},"type":"statement"}]}' % SCHOLARLY),
    line('{"P31":[{"mainsnak":{"snaktype":"value","property":"P\\u00331","datatype":"wikibase-item","datavalue":%s},"type":"statement"}]}' % SCHOLARLY),
    line('{"P31":[{"mainsnak":{"snaktype":"value","property":"P31","datatype":"wikibase\\u002ditem","datavalue":%s},"type":"statement"}]}' % SCHOLARLY),
    line('{"P31":[{"mainsnak":{"datavalue":%s,"snaktype":"value","property":"P31","datatype":"wikibase-item"},"type":"statement"}]}' % SCHOLARLY),
    line('{"P31":[{"mainsnak":{"snaktype":"value","datatype":"wikibase-item","datavalue":%s,"property":"P31"},"type":"statement"}]}' % SCHOLARLY),
    line('{"P31":[{"mainsnak":{"snaktype":"value","property":"P31","datatype":"wikibase-item","datavalue":{"value":"Q13442814","type":"string"}},"type":"statement"}]}'),
    line('{"P31":[{"mainsnak":{"snaktype":"somevalue","property":"P31","datatype":"wikibase-item"},"type":"statement"}]}'),
    # a P31 in a qualifier, in a reference, in a second statement, in a second property
    line('{"P1":[{"mainsnak":%s,"type":"statement","qualifiers":{"P31":[%s]}}]}' % (SNAK % ("P1", ITEM % (5, "Q5")), SNAK % ("P31", SCHOLARLY))),
    line('{"P1":[{"mainsnak":%s,"type":"statement","references":[{"hash":"h","snaks":{"P31":[%s]}}]}]}' % (SNAK % ("P1", ITEM % (5, "Q5")), SNAK % ("P31", SCHOLARLY))),
    line('{"P31":[{"mainsnak":%s,"type":"statement"},{"mainsnak":%s,"type":"statement"}]}' % (SNAK % ("P31", ITEM % (5, "Q5")), SNAK % ("P31", SCHOLARLY))),
    line('{"P1":[{"mainsnak":%s,"type":"statement"}],"P31":[{"mainsnak":%s,"type":"statement"}]}' % (SNAK % ("P1", ITEM % (5, "Q5")), SNAK % ("P31", SCHOLARLY))),
    # the reshape's changes of shape: maps as [] or null, references without snaks
    line('[]'), line('{}'), line('null'),
    p31(SCHOLARLY, extra=',"qualifiers":[]'), p31(SCHOLARLY, extra=',"qualifiers":null'), p31(SCHOLARLY, extra=',"qualifiers":{}'),
    p31(SCHOLARLY, extra=',"references":[]'), p31(SCHOLARLY, extra=',"references":[{}]'),
    p31(SCHOLARLY, extra=',"references":[{"hash":"h"}]'), p31(SCHOLARLY, extra=',"references":[{"hash":"h","snaks":[]}]'),
    p31(SCHOLARLY, extra=',"references":[{"hash":"h","snaks":{},"snaks-order":[]}]'),
    p31(SCHOLARLY, extra=',"references":null'), p31(SCHOLARLY, extra=',"references":[null]'),
    p31(SCHOLARLY, extra=',"qualifiers":{"P2":[{"snaktype":"novalue","property":"P2","hash":"q"}]},"qualifiers-order":["P2"]'),
    # datavalues: none, empty, without a value or a type, with another member, beside a datavalue_type
    p31('{}'), p31('{"value":{"entity-type":"item","numeric-id":13442814,"id":"Q13442814"}}'),
    p31('{"type":"wikibase-entityid"}'), p31('null'), p31('"x"'),
    p31('{"value":{"entity-type":"item","numeric-id":13442814,"id":"Q13442814"},"type":"wikibase-entityid","extra":1}'),
    line('{"P31":[{"mainsnak":{"snaktype":"value","property":"P31","datavalue_type":"x","datavalue":%s},"type":"statement"}]}' % SCHOLARLY),
    line('{"P31":[{"mainsnak":{"snaktype":"value","property":"P31","datavalue":%s,"datavalue_type":"x"},"type":"statement"}]}' % SCHOLARLY),
    line('{"P31":[{"mainsnak":{"snaktype":"value","property":"P31","datavalue":%s,"datavalue":%s},"type":"statement"}]}' % (SCHOLARLY, SCHOLARLY)),
    # statements without a mainsnak, with one that is not an object, repeated keys
    line('{"P31":[{"type":"statement"}]}'), line('{"P31":[{}]}'), line('{"P31":[{"mainsnak":null}]}'),
    line('{"P31":[{"mainsnak":{}}]}'), line('{"P31":["x"]}'), line('{"P31":{}}'), line('{"P31":null}'),
    line('{"P31":[{"mainsnak":%s,"type":"statement","type":"statement"}]}' % (SNAK % ("P31", SCHOLARLY))),
    line('{"P31":[],"P31":[]}'),
    # numbers and strings: floats the reshape rewrites, escapes, the reshape's \u checks
    line('{"P625":[{"mainsnak":{"snaktype":"value","property":"P625","datatype":"globe-coordinate","datavalue":{"value":{"latitude":1.0e-5,"longitude":-3.8209166666666667,"altitude":null,"precision":2.77778e-5,"globe":"http:\\/\\/www.wikidata.org\\/entity\\/Q2"},"type":"globecoordinate"}},"type":"statement"}]}'),
    line('{"P625":[{"mainsnak":{"snaktype":"value","property":"P625","datatype":"globe-coordinate","datavalue":{"value":{"latitude":12,"longitude":-0,"altitude":null,"precision":1E2,"globe":"g"},"type":"globecoordinate"}},"type":"statement"}]}'),
    line('{"P1476":[{"mainsnak":{"snaktype":"value","property":"P1476","datatype":"monolingualtext","datavalue":{"value":{"text":"\\u00e9\\ud83d\\ude00\\"\\\\\\/\\n\\u0000","language":"fr"},"type":"monolingualtext"}},"type":"statement"}]}'),
    line('{"P1":[{"mainsnak":{"snaktype":"value","property":"P1","datatype":"string","datavalue":{"value":"\\ud800","type":"string"}},"type":"statement"}]}'),
    line('{"P1":[{"mainsnak":{"snaktype":"value","property":"P1","datatype":"string","datavalue":{"value":"\\u+041","type":"string"}},"type":"statement"}]}'),
    line('{"P1":[{"mainsnak":{"snaktype":"value","property":"P1","datatype":"string","datavalue":{"value":"\\x","type":"string"}},"type":"statement"}]}'),
    line('{"P1":[{"mainsnak":{"snaktype":"value","property":"P1","datatype":"quantity","datavalue":{"value":{"amount":"+1","unit":"1","upperBound":1e400},"type":"quantity"}},"type":"statement"}]}'),
    line('{"P1":[{"mainsnak":{"snaktype":"value","property":"P1","datatype":"time","datavalue":{"value":{"time":"+2001","timezone":-0,"before":0,"after":01,"precision":11,"calendarmodel":"c"},"type":"time"}},"type":"statement"}]}'),
    line('{"P\\u0031":[{"mainsnak":%s,"type":"statement"}]}' % (SNAK % ("P1", ITEM % (5, "Q5")))),
    line('{"P31":[{"mainsnak":%s, "type":"statement"}]}' % (SNAK % ("P31", SCHOLARLY))),
    line('{"P31":[{"mainsnak":%s,"type":true}]}' % (SNAK % ("P31", SCHOLARLY))),
    # a line the reshape leaves to entity_row, and one without claims
    b'{"type":"item","id":"Q1","labels":{"en":"bare"},"claims":{}}',
    b'{"type":"item","id":"Q1","labels":{}}',
]


def mutants(lines: list[bytes], n: int) -> list[bytes]:
    """Each of `n` lines one of `lines` with a byte, a span or a token changed (seeded)."""
    rng = random.Random(7)
    tokens = [b"{", b"}", b"[", b"]", b",", b":", b'"', b"\\", b"null", b"{}", b"[]", b"0", b"-0", b"1e5",
              b"\\u0031", b"\\/", b"P31", b"Q13442814", b'"datavalue"', b'"mainsnak"', b'"snaks"', b" "]
    out = []
    for _ in range(n):
        s = bytearray(rng.choice(lines))
        start = s.index(b'"claims"')
        i = rng.randrange(start, len(s))
        kind = rng.randrange(3)
        if kind == 0:
            del s[i:i + rng.randint(1, 8)]
        elif kind == 1:
            s[i:i] = rng.choice(tokens)
        else:
            s[i:i + rng.randint(1, 4)] = rng.choice(tokens)
        out.append(bytes(s))
    return out


def dump_lines(path: str) -> list[bytes]:
    opener = bz2.open if path.endswith(".bz2") else open
    with opener(path, "rb") as f:
        return [ln.rstrip(b",\r\n") for ln in f if ln.startswith(b"{")]


def main(paths: list[str]) -> None:
    failed = 0
    stats = collections.Counter()
    for path in paths:
        lines = dump_lines(path)
        for start in range(0, len(lines), 2000):
            failed += check(lines[start:start + 2000], f"{path} lines {start}-", stats)
        print(f"{path}: {len(lines):,} lines, {dict(stats)}")
        # and in batches of 64 KiB of claims (a set's batches cut, rows moved between sets
        # across their ends), on the file's first chunks
        batch, claims_build.BATCH_TEXT_BYTES = claims_build.BATCH_TEXT_BYTES, 64 * 1024
        small = collections.Counter()
        for start in range(0, min(len(lines), 6000), 2000):
            failed += check(lines[start:start + 2000], f"{path} lines {start}- in small batches", small)
        claims_build.BATCH_TEXT_BYTES = batch
        print(f"{path}, in batches of 64 KiB: {dict(small)}")
    if paths:
        # the files' lines shuffled (seeded), so the sets' rows interleave as they do in a
        # release's chunks (the bench's chunks hold one set's rows, or two runs of them):
        # rows moved between the sets, in the split's batches and in batches of 64 KiB
        every = [ln for path in paths for ln in dump_lines(path)]
        mixed = random.Random(3).sample(every, min(len(every), 6000))
        batch = claims_build.BATCH_TEXT_BYTES
        for size in (batch, 64 * 1024):
            claims_build.BATCH_TEXT_BYTES = size
            shuffled = collections.Counter()
            for start in range(0, len(mixed), 2000):
                failed += check(mixed[start:start + 2000], f"shuffled lines {start}- in batches of {size}", shuffled)
            print(f"{len(mixed):,} of the lines shuffled, in batches of {size:,} bytes: {dict(shuffled)}")
        claims_build.BATCH_TEXT_BYTES = batch
    edges = collections.Counter()
    for i, e in enumerate(EDGES):
        failed += check([e], f"edge {i}", edges)
        failed += check([EDGES[0], e, EDGES[1]], f"edge {i} among others", edges)
    failed += check(EDGES, "the edges as a chunk", edges)
    print(f"{len(EDGES)} edges: {dict(edges)}")
    fuzz = collections.Counter()
    for i, m in enumerate(mutants(EDGES[:-2], 20_000)):
        failed += check([m], f"mutant {i}: {m[:300]!r}", fuzz)
    print(f"20,000 mutants: {fuzz['built']:,} built, {20_000 - fuzz['built']:,} left to the text path")
    if failed:
        raise SystemExit(f"{failed} failed")
    print("all passed")


if __name__ == "__main__":
    main(sys.argv[1:])
