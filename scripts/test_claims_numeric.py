"""A globe-coordinate's latitude, longitude and precision in `__number` on every path
(claims_numeric.canonical_coordinates), and nothing else changed:

- **each path gives the same arrays** (schema and values) for each case: the split's build
  (claims_build.claims_arrays), genson's strict decode against claims.avsc, and genson's
  inference (the fallback), each made canonical as the pipeline makes it. Each case at a
  mainsnak, a qualifier and a reference, alone in its part (the thinnest part, whose
  inference has an integer field where its own values are integers) and beside a
  release's context (test_claims_schema.CONTEXT);
- **the cases:** whole-number, fractional, negative, 0 and null latitudes, longitudes and
  precisions, each beside a time (its precision an integer) at the same place;
- **the values:** each globe-coordinate's latitude, longitude and precision is
  `{__number: float(v), __integer: null}` (or a null record for a null), the float
  exactly the JSON's number; each time's precision `{__integer: v, __number: null}`;
- **only those cells change:** each path's arrays before the rule and after it are the
  same everywhere else, and where a cell moved its number is the same;
- **the files:** the split's claims partition file (write_claims from the text,
  write_built from built arrays) is normalise_claims_direct's, with claims.avsc and
  with the fallback;
- **exactness:** an integer a Float64 does not hold exactly (|v| > 2^53) halts every path
  rather than being rounded; 2^53 itself moves exactly; an integer beside a float halts;
- **arrays with offsets** (slices) come out as their slice of the whole's;
- **the bench's parity rule** (bench/parity.py `canonical`, made apart in Polars) gives
  each path's statements as this rule writes them, and leaves those unchanged.

    WIKIDATA_RELEASE=20261012 .venv/bin/python scripts/test_claims_numeric.py
"""

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("WIKIDATA_RELEASE", "20261012")
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))

import orjson  # noqa: E402
import polars as pl  # noqa: E402
import pyarrow as pa  # noqa: E402

from test_claims_build import claims, genson_claims, q_snak, same_partition, statement  # noqa: E402
from test_claims_schema import CONTEXT, RETRIEVED, STATED_IN, coord, reference  # noqa: E402
from wikidata import claims_build, process  # noqa: E402
from parity import canonical  # noqa: E402
from wikidata.claims_numeric import FIELDS, GLOBE, NotExact, canonical_coordinates  # noqa: E402
from wikidata.partitioning.claims import statements  # noqa: E402

TIME = {k: v for k, v in q_snak(property="P585").items() if k != "hash"}  # (precision 9)
EXACT = 2**53


def at(place: str, snak: dict) -> bytes:
    """A part's claims cell with `snak` (and a time) at `place`, P625's statement."""
    if place == "mainsnak":
        return orjson.dumps({"P625": [statement(mainsnak=snak)], "P585": [statement(mainsnak=TIME)]})
    if place == "qualifier":
        q = {k: v for k, v in snak.items()}
        return claims(statement(qualifiers={"P625": [q], "P585": [TIME]}, **{"qualifiers-order": ["P625", "P585"]}))
    return claims(statement(references=[reference(snak, TIME), reference(STATED_IN, RETRIEVED)]))


VALUES = {
    "whole numbers": (47, 8, 1),
    "fractions": (47.049845, 8.30308, 2.777777777777778e-08),
    "negatives": (-33, -70, 10),
    "negative fractions": (-33.5, -70.25, 0.5),
    "zeros": (0, 0, 0),
    "float zeros": (0.0, -0.0, 0.0),
    "a null precision": (12, -3.5, None),
    "the extremes": (-90, 180, 360),
    "an integer beside a fraction": (52, 13.25, 0.0001),
    "2^53": (EXACT, -EXACT, EXACT),
}
CASES = {f"{place}: {name}": at(place, coord(*v)) for place in ("mainsnak", "qualifier", "reference")
         for name, v in VALUES.items()}
# A part of several, the same snak whole in one row and fractional in the next
CASES["mainsnak: whole and fractional rows"] = at("mainsnak", coord(47, 8, 1))
INEXACT = {f"{place}: {f} 2^53 + 1": at(place, coord(*[EXACT + 1 if g == f else 1.5 for g in FIELDS]))
           for place in ("mainsnak", "qualifier", "reference") for f in FIELDS}


def snaks_of(row: dict):
    """A row's snaks (to_pylist's), each with its place."""
    for group in row["claims"] or []:
        for s in group["value"] or []:
            yield "mainsnak", s["mainsnak"]
            for g in s.get("qualifiers") or []:
                for q in g["value"] or []:
                    yield "qualifier", q
            for r in s.get("references") or []:
                for g in r["snaks"] or []:
                    for q in g["value"] or []:
                        yield "reference", q


def json_snaks(text: bytes):
    for prop, sts in orjson.loads(text).items():
        for s in sts:
            yield "mainsnak", s["mainsnak"]
            for q in (s.get("qualifiers") or {}).values():
                yield from (("qualifier", x) for x in q)
            for r in s.get("references") or []:
                for q in r["snaks"].values():
                    yield from (("reference", x) for x in q)


def check_values(table: pa.Table, texts: list[bytes]) -> int:
    """Each globe-coordinate's numbers in __number, exactly its JSON's; each time's
    precision in __integer. Returns the globe-coordinates seen."""
    seen = 0
    for row, text in zip(table.to_pylist(), texts):
        got, want = list(snaks_of(row)), list(json_snaks(text))
        # (the arrays' snaks in their properties' order, as the JSON's)
        assert [(p, s["property"]) for p, s in got] == [(p, s["property"]) for p, s in want]
        for (place, s), (_, j) in zip(got, want):
            dv, jv = s["datavalue"], j.get("datavalue")
            if s["datavalue_type"] == GLOBE:
                seen += 1
                for f in FIELDS:
                    v = jv[f]
                    if v is None:
                        assert dv[f] is None, f"{place} {f}: {dv[f]} for a null"
                        continue
                    num = dv[f][f"{f}__number"]
                    assert dv[f][f"{f}__integer"] is None, f"{place} {f}: an integer {dv[f]}"
                    assert type(num) is float, f"{place} {f}: {num} for {v}"
                    if type(v) is int:  # (exactly; a fraction is as genson's reader reads it, not always the nearest double)
                        assert num == v and str(num) == str(float(v)), f"{place} {f}: {num} for {v}"
                    else:
                        assert abs(num - v) <= 4e-16 * abs(v) and str(num)[0] == str(v)[0], f"{place} {f}: {num} for {v}"
            elif s["datavalue_type"] == "time":
                assert dv["precision"] == {"precision__integer": jv["precision"], "precision__number": None}, dv
    return seen


def only_moved(before: pa.Table, after: pa.Table) -> int:
    """`after` is `before` but for globe-coordinates' numbers moved to __number (the same
    number) and records of nulls made null. Returns the cells changed."""
    changed = 0
    assert before.schema.equals(after.schema, check_metadata=True)
    for a, b in zip(before.to_pylist(), after.to_pylist()):
        sa, sb = list(snaks_of(a)), list(snaks_of(b))
        assert len(sa) == len(sb)
        for (_, x), (_, y) in zip(sa, sb):
            if x["datavalue_type"] != GLOBE or x["datavalue"] is None:
                assert x == y
                continue
            xd, yd = dict(x["datavalue"]), dict(y["datavalue"])
            for f in FIELDS:
                u, v = xd.pop(f), yd.pop(f)
                if u == v:
                    continue
                changed += 1
                ints, nums = (u or {}).get(f"{f}__integer"), (u or {}).get(f"{f}__number")
                if ints is None and nums is None:
                    assert v is None, (u, v)
                else:
                    assert nums is None and v == {f"{f}__number": float(ints), f"{f}__integer": None} or \
                        v == {f"{f}__integer": None, f"{f}__number": float(ints)}, (u, v)
            assert xd == yd and {k: w for k, w in x.items() if k != "datavalue"} == {
                k: w for k, w in y.items() if k != "datavalue"}
    return changed


def inferred(path: Path) -> pa.Table:
    saved, process.CLAIMS_AVRO = process.CLAIMS_AVRO, None
    try:
        return genson_claims(path)
    finally:
        process.CLAIMS_AVRO = saved


def direct(path: Path, d: Path, total: int, avro: bool = True) -> Path:
    """normalise_claims_direct's claims file for the part at `path`."""
    saved = process.CLAIMS_AVRO
    process.CLAIMS_AVRO = saved if avro else None
    try:
        return process.normalise_claims_direct(path, d / "claims" / path.name, d / "lookup.parquet", d / "q.parquet", total)
    finally:
        process.CLAIMS_AVRO = saved


def check_case(name: str, texts: list[bytes], d: Path) -> tuple[int, int]:
    path = d / "part.parquet"
    part = pl.DataFrame({"id": [f"Q{i}" for i in range(1, 1 + len(texts))], "claims": [t.decode() for t in texts]})
    part.write_parquet(path)
    built, why = claims_build.claims_arrays(part["id"], claims_build.text_of(part))
    assert built is not None, f"{name!r}: not built ({why})"
    raw = {"built": built, "strict": genson_claims(path), "fallback": inferred(path)}
    canon = {k: canonical_coordinates(t) for k, t in raw.items()}
    for k, t in canon.items():
        assert t.schema.equals(canon["built"].schema), f"{name!r}: {k}'s schema"
        assert t.equals(canon["built"]), f"{name!r}: {k} is not the build's"
        assert canonical_coordinates(t).equals(t), f"{name!r}: {k} not canonical once made so"
    seen = check_values(canon["built"], texts)
    for k in raw:  # (the bench's rule, apart: each path's statements by it are the rule's, which it leaves)
        want = pl.from_arrow(statements(canon[k])).rechunk()  # (one chunk: see parity.canonical)
        assert canonical(pl.from_arrow(statements(raw[k])).rechunk().lazy())[0].collect().equals(want), f"{name!r}: parity's rule ({k})"
        assert canonical(want.lazy())[0].collect().equals(want), f"{name!r}: parity's rule changes the rule's ({k})"
    changed = {k: only_moved(raw[k], canon[k]) for k in raw}
    # The files: the split's (from text, and from built arrays) against the pipeline's
    total = process.n_ids(part)
    files = {"strict": direct(path, d / "s", total), "fallback": direct(path, d / "f", total, avro=False)}
    for k, f in files.items():
        for writer in ("text", "built"):
            mine = d / f"{writer}.parquet"
            if writer == "text":
                assert claims_build.write_claims(part["id"], claims_build.text_of(part), None, mine) is None
            else:
                claims_build.write_built(part["id"], built.column("claims"), mine)
            assert same_partition(mine, f), f"{name!r}: the split's file ({writer}) is not normalise_claims_direct's ({k})"
    return seen, changed["fallback"] + changed["built"]


def check_inexact(d: Path) -> None:
    for name, text in INEXACT.items():
        path = d / "part.parquet"
        part = pl.DataFrame({"id": ["Q1"], "claims": [text.decode()]})
        part.write_parquet(path)
        halted = []
        built, why = claims_build.claims_arrays(part["id"], claims_build.text_of(part))
        tables = {"strict": genson_claims(path), "fallback": inferred(path), **({"built": built} if built else {})}
        for k, t in tables.items():
            ints = any(f"__integer': {EXACT + 1}" in str(s["datavalue"]) for row in t.to_pylist() for _, s in snaks_of(row))
            if not ints:  # (a float field alone: decoded to a float by genson, as before; see the journal)
                continue
            try:
                canonical_coordinates(t)
            except NotExact:
                halted.append(k)
            else:
                raise AssertionError(f"{name!r} ({k}): an inexact integer moved")
        for k, f in (("fallback", False), ("strict", True)):
            if k in halted:
                try:
                    direct(path, d / k, 1, avro=f)
                except NotExact:
                    continue
                raise AssertionError(f"{name!r}: normalise_claims_direct ({k}) wrote it")
        if "built" in halted:
            try:
                claims_build.write_claims(part["id"], claims_build.text_of(part), None, d / "x.parquet")
            except NotExact:
                pass
            else:
                raise AssertionError(f"{name!r}: write_claims wrote it")
        print(f"inexact {name!r}: halted where an integer was held ({', '.join(halted) or 'none'})")
        if "precision" in name:  # (an integer field there on every path)
            assert set(halted) == set(tables), f"{name!r}: not halted on every path ({halted})"


def check_both() -> None:
    """An integer with a float beside it (which no path writes) halts."""
    t = claims_build.SCHEMA
    snak = t.field("claims").type.value_type.field("value").type.value_type.field("mainsnak").type
    row = {"claims": [{"key": "P625", "value": [{"mainsnak": {
        "snaktype": "value", "property": "P625", "datavalue_type": GLOBE, "datatype": "globe-coordinate",
        "datavalue": {"latitude": {"latitude__number": 1.5, "latitude__integer": 1}}}}]}], "id": "Q1"}
    table = pa.Table.from_pylist([row], schema=t)
    assert table.column("claims").type.value_type.field("value").type.value_type.field("mainsnak").type == snak
    try:
        canonical_coordinates(table)
    except NotExact:
        print("an integer beside a float: halted")
        return
    raise AssertionError("an integer beside a float moved")


def check_slices(d: Path) -> None:
    texts = [CASES["mainsnak: whole numbers"], CASES["reference: fractions"], CASES["qualifier: negatives"],
             CASES["reference: zeros"], CASES["mainsnak: a null precision"]]
    part = pl.DataFrame({"id": [f"Q{i}" for i in range(len(texts))], "claims": [t.decode() for t in texts]})
    built, _ = claims_build.claims_arrays(part["id"], claims_build.text_of(part))
    whole = canonical_coordinates(built)
    for start, n in ((1, 3), (2, 3), (4, 1), (0, 0)):
        assert canonical_coordinates(built.slice(start, n)).equals(whole.slice(start, n)), (start, n)
    # (a table with nothing to move is handed back as it is)
    floats = pl.DataFrame({"id": ["Q1"], "claims": [CASES["reference: fractions"].decode()]})
    t, _ = claims_build.claims_arrays(floats["id"], claims_build.text_of(floats))
    assert canonical_coordinates(t) is t
    print("slices: each its slice of the whole's; a table with nothing to move unchanged")


def main() -> None:
    globes = changed = 0
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        for name, text in CASES.items():
            shapes = {"alone": [text], "with context": [text, *CONTEXT]}
            if "whole and fractional" in name:
                shapes = {"two rows": [text, at("mainsnak", coord(47.5, 8.5, 1.5))]}
            for shape, texts in shapes.items():
                s, c = check_case(f"{name} ({shape})", texts, d)
                globes, changed = globes + s, changed + c
            print(f"case {name!r}: every path the same, alone and with context; its files the same")
        check_inexact(d)
        check_both()
        check_slices(d)
    print(f"{len(CASES)} cases, {globes} globe-coordinates checked, {changed} cells moved from the paths' arrays")


if __name__ == "__main__":
    main()
