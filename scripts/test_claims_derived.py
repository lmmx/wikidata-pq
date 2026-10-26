#!/usr/bin/env python3
"""Test the claims' derived leaves (src/wikidata/claims_derived.py): each datavalue's
`entity-type` and `numeric-id` left out of the claims partition and group files where
they are exactly a function of its `id`, and put back by finalise.

End to end (the split, both sets' runs and their finalises' local stages, each a process
of its own, on a dump slice made of the chunk files given, as scripts/test_schema_evolution.py
runs them, whose scenarios and stand-in Hub this uses):

- **keep:** WIKIDATA_CLAIMS_DERIVED=keep, every leaf written, as before: the reference.
- **drop:** the default. Every file published is keep's, byte for byte; every file
  uploaded is too, but the claims group files, each without all six derived leaves (the
  slice's values are all derivable) and named so in its metadata, and smaller.
- **mixed:** each claims row group or file written with all its leaves, without the
  numeric-ids alone, or without all six, by its first id (a hook in every process): a
  file's row groups of several forms written again in one (claims_build.RowGroups), a
  group's partition files of several forms glued in one (push.core._harmonise), and the
  build branch's claims group files of several forms, each read as its own in finalise.
  Every file published is keep's, byte for byte.
- **mixed, killed:** the same, each run killed once at a kill point and run again until it
  completes: a row group file written again in its file's form, a group's file of other
  leaves written in the group's form, and the sort putting back a file's leaves. Every
  file published is keep's, byte for byte.

- **a new entity type mid-release:** a main snak's item value made a MediaInfo one
  (`M5`, mediainfo, 5: a type the rule does not give) in one entity of the slice: its
  claims file, and its group's, keep the main snak's two leaves (the others dropped), and
  every file published is keep's of the same slice, byte for byte. (A new field
  mid-release, a datavalue's among them, is test_schema_evolution.py's: its scenarios
  run with the leaves dropped, and publish native's files.)

On the claims published by keep (each a full claims file, as a partition file is laid out):

- **derive:** the entity type and number of ids of every form (and of none);
- **round trip:** each file without its leaves (reduce), written as the split writes one
  (write_table), read back by pyarrow and pq-glue, and its leaves put back (restore): the
  file's rows and schema exactly; its plan's footer estimate (sort_by_id.footer_size) the
  full file's; its schema checked as its full schema less the leaves it names
  (check_form, release_schema.check);
- **checked, not assumed:** a value changed in one row (a qualifier's numeric-id, a
  reference's entity-type) keeps that leaf in the file, and only that one; a value under a
  null datavalue, which Parquet does not store, does not;
- **refused:** a schema less a leaf it does not name, or naming a leaf it has, or a leaf
  that is not derived; rows without their leaves given to conform_table;
- **a file's row groups:** a row group whose values allow fewer leaves than those before
  it has those written again with the leaf put back (RowGroups), the file one form and
  its rows the table's.

Usage: python scripts/test_claims_derived.py MAIN.parquet SCHOLAR.parquet [scenario ...]
       (e.g. bench-data's r50.parquet and s30.parquet)
"""

import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "bench"))
os.environ.setdefault("WIKIDATA_RELEASE", "20261012")

from test_schema_evolution import HOOKS, Scenario, check, clean, same_published, slice_lines  # noqa: E402

MIX = r'''
import hashlib
if os.environ.get("MIX_DERIVED"):
    from wikidata import claims_build, claims_derived, claims_refs
    claims_build.ARROW_ROW_GROUP_ROWS = 1_500  # (several row groups a claims file, of several forms)
    _reduce = claims_derived.reduce
    def reduce(t):
        """Each table's form by its first id: every leaf, the numeric-ids dropped alone, or all."""
        if not t.num_rows:
            return _reduce(t)
        k = int(hashlib.sha256(t.column("id")[0].as_py().encode()).hexdigest(), 16) % 3
        if k == 0:
            return t
        if k == 1:
            return claims_derived.drop(t, {l for l in claims_derived.matching(t) if l.endswith("numeric-id")})
        return _reduce(t)
    claims_derived.reduce = claims_refs.reduce = reduce
    _one = claims_build.RowGroups._one_form
    def one_form(self):
        print("[test] row groups of several forms written in one", flush=True)
        _one(self)
        if hook == "one_form":
            tick()
    claims_build.RowGroups._one_form = one_form
if hook == "restore_read":
    from wikidata import claims_derived, sort_by_id
    _restore = claims_derived.restore
    def restore(*a, **k):
        tick()
        return _restore(*a, **k)
    claims_derived.restore = restore
'''

KILLS = {
    "split: a file's row groups written again in its form": {"split": ("one_form", 1)},
    "run: mid merge, a file of other leaves written in the group's form": {"main": ("upgrade_file", 1)},
    "finalise: the sort putting back a file's leaves": {"finalise": ("restore_read", 1)},
}


def forms(s: Scenario) -> list[frozenset]:
    """Each claims group file's leaves dropped, on the stand-in Hub's build branches."""
    import pyarrow.parquet as pq

    from wikidata import claims_derived

    out = []
    for p in sorted((s.root / "fakehub").rglob("*.parquet")):
        schema = pq.read_schema(p)
        if "datavalue" in schema.names:
            out.append(claims_derived.dropped(schema))
    return out


def logs(s: Scenario) -> str:
    return "".join(p.read_text() for p in s.root.glob("*.log"))


def new_type(lines: list[bytes]) -> tuple[list[bytes], str]:
    """The slice with the first main snak's item value from line 1,500 on made MediaInfo's."""
    import orjson

    from dumpslice import dump_escaped

    out = list(lines)
    for k in range(1500, len(lines)):
        e = orjson.loads(lines[k])
        for ss in e.get("claims", {}).values():
            for st in ss:
                dv = st["mainsnak"].get("datavalue")
                if dv and dv.get("type") == "wikibase-entityid" and dv["value"].get("entity-type") == "item":
                    dv["value"] = {"entity-type": "mediainfo", "numeric-id": 5, "id": "M5"}
                    out[k] = dump_escaped(orjson.dumps(e))
                    return out, e["id"]
    raise SystemExit("no item value to change")


def end_to_end(tmp: Path, lines: list[bytes], run) -> tuple[bool, dict | None]:
    failed = False
    keep = Scenario(tmp, "keep", lines, extra={"WIKIDATA_CLAIMS_DERIVED": "keep"})
    codes = keep.sequential()
    ref = keep.finalise({})
    failed |= not check(not any(codes.values()) and clean(ref) and not any(forms(keep)),
                        "keep: every chunk complete, every claims group file with its derived leaves")
    if run("drop"):
        s = Scenario(tmp, "drop", lines)
        codes = s.sequential()
        got = s.finalise({})
        six = forms(s)
        differ = {f for n in ref for f in set(ref[n]["uploaded"]) | set(got[n]["uploaded"])
                  if ref[n]["uploaded"].get(f) != got[n]["uploaded"].get(f)}
        ok = not any(codes.values()) and clean(got) and same_published(got, ref)
        ok &= all(set(ref[n]["uploaded"]) == set(got[n]["uploaded"]) for n in ref)
        ok &= bool(differ) and all(f.startswith("claims/") for f in differ)
        ok &= bool(six) and all(len(f) == 6 for f in six)
        size = lambda sc: sum(p.stat().st_size for p in (sc.root / "fakehub").rglob("*.parquet")  # noqa: E731
                              if "datavalue" in __import__("pyarrow.parquet").parquet.read_schema(p).names)
        a, b = size(keep), size(s)
        failed |= not check(ok, f"drop: keep's files published, byte for byte; uploaded but the {len(six)} claims group "
                            f"files, each without its six derived leaves ({a:,} -> {b:,} bytes, {(b - a) / a:+.1%})")
    if run("mixed"):
        s = Scenario(tmp, "mixed", lines, extra={"MIX_DERIVED": "1"})
        codes = s.sequential()
        got = s.finalise({})
        f = forms(s)
        text = logs(s)
        ok = not any(codes.values()) and clean(got) and same_published(got, ref) and len(set(f)) > 1
        ok &= "[test] row groups of several forms written in one" in text and "written with derived leaves put back" in text
        failed |= not check(ok, f"mixed: keep's files published, byte for byte, from claims group files of {len(set(f))} "
                            "forms (a file's row groups and a group's files of several written in one)")
    if run("killed"):
        for kill, hooks in KILLS.items():
            s = Scenario(tmp, f"killed-{list(KILLS).index(kill)}", lines, extra={"MIX_DERIVED": "1"})
            for name in ("split", "main", "scholar"):
                for attempt in range(6):
                    k = hooks.get(name) if attempt == 0 else None
                    if not s.wait({name: s.start(name, k)})[name]:
                        break
                    if not k:
                        raise SystemExit(f"FAIL killed {kill}: {name} failed\n{s.log(name)}")
                else:
                    raise SystemExit(f"FAIL killed {kill}: {name} not complete after six runs")
            got = s.finalise({}, kill=hooks.get("finalise"))
            text = logs(s) + "".join((s.root / f"finalise-{k}.log").read_text() for k in ("main", "scholar"))
            killed = "[test] killed" in text
            ok = clean(got) and same_published(got, ref) and killed
            failed |= not check(ok, f"mixed, killed at {kill}, run again: keep's files published, byte for byte"
                                + ("" if killed else " (NOT KILLED)"))
    if run("newtype"):
        changed, eid = new_type(lines)
        k = Scenario(tmp, "newtype-keep", changed, extra={"WIKIDATA_CLAIMS_DERIVED": "keep"})
        codes = k.sequential()
        want = k.finalise({})
        s = Scenario(tmp, "newtype", changed)
        codes |= {f"drop {n}": c for n, c in s.sequential().items()}
        got = s.finalise({})
        f = forms(s)
        main = {"datavalue.entity-type", "datavalue.numeric-id"}
        ok = not any(codes.values()) and clean(got) and same_published(got, want)
        ok &= any(not (x & main) and len(x) == 4 for x in f) and any(len(x) == 6 for x in f)
        failed |= not check(ok, f"a new entity type mid-release ({eid}: an M5 mediainfo value): its group's claims file "
                            "keeps the main snak's two leaves, the others drop all six; keep's files published, byte for byte")
    return failed, keep


def units(keep: Scenario) -> bool:
    import numpy as np
    import pq_glue
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    from wikidata import claims_build, claims_derived as cd, release_schema, sort_by_id
    from wikidata.config import Table
    from wikidata.partitioning.core import write_table
    from wikidata.schema_evolution import conform_table

    failed = False
    ids = pa.array(["Q1", "P31", "L7", "L7-F2", "L7-S10", "E12", None, "Q", "X1", "Q01", "L1-F", "Q" + "1" * 19,
                    "Q" + "9" * 18, "M5", "q1", "Q1 ", "L1-X2", "E1-F1", "", "Q-1", "Q１"], pa.large_string())
    want_type = ["item", "property", "lexeme", "form", "sense", "entity-schema"] + [None] * 3 + ["item"] + [None] * 2 + \
        ["item"] + [None] * 8
    want_num = [1, 31, 7] + [None] * 6 + [1, None, None, 999999999999999999] + [None] * 8
    e, n = cd.derive(ids)
    e2, n2 = cd.derive(ids.slice(3, 5))
    failed |= not check(e.to_pylist() == want_type and n.to_pylist() == want_num and e.type == pa.large_string()
                        and n.type == pa.int64() and e2.to_pylist() == want_type[3:8] and n2.to_pylist() == want_num[3:8],
                        "derive: the entity type and number of ids of every form, and none of any other")

    full = release_schema.CLAIMS.file_schema(0)
    claims = sorted(p for p in keep.root.rglob("*.parquet")
                    if {"hub", "claims"} <= set(p.parts) and ".cache" not in p.parts)
    table = pa.concat_tables([pq.read_table(p) for p in claims]).replace_schema_metadata()
    work = keep.root / "units"
    work.mkdir(exist_ok=True)
    ok = table.schema.equals(full) and table.num_rows > 10_000
    leaves = cd.matching(table)
    ok &= leaves == frozenset(cd.derivable(full)) and len(leaves) == 6
    write_table(table, work / "full.parquet")
    write_table(cd.reduce(table), work / "reduced.parquet")
    r = pq.read_table(work / "reduced.parquet")
    g = pq_glue.read_parquet(work / "reduced.parquet")
    g = g.replace_schema_metadata(r.schema.metadata)
    back = cd.restore(r, full)
    ok &= cd.dropped(r.schema) == leaves and back.equals(pq.read_table(work / "full.parquet"))
    ok &= back.schema.equals(full) and cd.restore(g, full).equals(back)
    ok &= cd.restore(r.slice(1000, 5000), full).equals(back.slice(1000, 5000))
    ok &= sort_by_id.footer_size([work / "reduced.parquet"], Table.CLAIMS) == sort_by_id.footer_size([work / "full.parquet"], Table.CLAIMS)
    ok &= sort_by_id.footer_size([work / "reduced.parquet"]) == sort_by_id.footer_size([work / "full.parquet"])
    ok &= cd.check_form(r.schema, full, "reduced") == leaves and release_schema.CLAIMS.check(r.schema, "reduced", 0) == 0
    a, b = (work / "full.parquet").stat().st_size, (work / "reduced.parquet").stat().st_size
    failed |= not check(ok, f"round trip: {table.num_rows:,} statements without their six leaves ({a:,} -> {b:,} bytes, "
                        f"{(b - a) / a:+.1%}), read back by pyarrow and pq-glue and put back: the file's rows and schema, "
                        "its plan's footer estimate, its schema checked")

    def changed(path: str, leaf: str) -> pa.Table:
        """`table` with the first non-null value of a leaf changed (a numeric-id + 1, an
        entity-type another)."""
        name = path.split(".", 1)[0]
        col = table.column(name).combine_chunks()

        def edit(arr, at):
            if at == path:
                kids = {arr.type.field(i).name: arr.field(i) for i in range(arr.type.num_fields)}
                v = kids[leaf]
                valid = np.flatnonzero(pc.is_valid(v).to_numpy(zero_copy_only=False))
                parent_ok = pc.is_valid(arr).to_numpy(zero_copy_only=False)
                i = int(next(j for j in valid if parent_ok[j]))
                vals = v.to_pylist()
                vals[i] = vals[i] + 1 if leaf == "numeric-id" else "property"
                kids[leaf] = pa.array(vals, v.type)
                return pa.StructArray.from_arrays(list(kids.values()), fields=list(arr.type), mask=pc.is_null(arr) if arr.null_count else None)
            if pa.types.is_large_list(arr.type):
                return pa.LargeListArray.from_arrays(arr.offsets, edit(arr.values, f"{at}.list.element"), type=arr.type,
                                                     mask=pc.is_null(arr) if arr.null_count else None)
            kids = []
            for i in range(arr.type.num_fields):
                f = arr.type.field(i)
                sub = f"{at}.{f.name}"
                kids.append(edit(arr.field(i), sub) if path == sub or path.startswith(sub + ".") else arr.field(i))
            return pa.StructArray.from_arrays(kids, fields=list(arr.type), mask=pc.is_null(arr) if arr.null_count else None)

        return table.set_column(table.schema.get_field_index(name), table.schema.field(name), edit(col, name))

    q = "qualifiers.list.element.value.list.element.datavalue"
    ref = "references.list.element.snaks.list.element.value.list.element.datavalue"
    ok = True
    for site, leaf in ((q, "numeric-id"), (ref, "entity-type")):
        t = changed(site, leaf)
        m = cd.matching(t)
        red = cd.reduce(t)
        write_table(red, work / "changed.parquet")
        back = cd.restore(pq.read_table(work / "changed.parquet"), full)
        ok &= m == leaves - {f"{site}.{leaf}"} and back.equals(t) and not back.equals(table)
    # A new entity type (MediaInfo), and an entity value without an id (as old dumps wrote
    # them): the leaves of that site kept
    col = table.column("datavalue").combine_chunks()
    kids = {col.type.field(i).name: col.field(i) for i in range(col.type.num_fields)}
    at = int(np.flatnonzero(pc.equal(kids["entity-type"], "item").to_numpy(zero_copy_only=False))[0])
    for new_id, new_type_, new_num in (("M5", "mediainfo", 5), (None, "item", 42)):
        k2 = dict(kids)
        for name, v in (("id", new_id), ("entity-type", new_type_), ("numeric-id", new_num)):
            vals = k2[name].to_pylist()
            vals[at] = v
            k2[name] = pa.array(vals, k2[name].type)
        dv = pa.StructArray.from_arrays(list(k2.values()), fields=list(col.type), mask=pc.is_null(col) if col.null_count else None)
        t = table.set_column(table.schema.get_field_index("datavalue"), table.schema.field("datavalue"), dv)
        write_table(cd.reduce(t), work / "newtype.parquet")
        ok &= cd.matching(t) == leaves - {"datavalue.entity-type", "datavalue.numeric-id"}
        ok &= cd.restore(pq.read_table(work / "newtype.parquet"), full).equals(t)
    # A value under a null datavalue: not stored, so not compared
    col = table.column("datavalue").combine_chunks()
    null_at = int(np.flatnonzero(pc.is_null(col).to_numpy(zero_copy_only=False))[0])  # (a somevalue's)
    kids = {col.type.field(i).name: col.field(i) for i in range(col.type.num_fields)}
    vals = kids["entity-type"].to_pylist()
    vals[null_at] = "item"
    kids["entity-type"] = pa.array(vals, pa.large_string())
    hidden = pa.StructArray.from_arrays(list(kids.values()), fields=list(col.type), mask=pc.is_null(col))
    t = table.set_column(table.schema.get_field_index("datavalue"), table.schema.field("datavalue"), hidden)
    ok &= cd.matching(t) == leaves
    failed |= not check(ok, "checked, not assumed: a value changed in one row keeps that leaf alone (a qualifier's "
                        "numeric-id, a reference's entity-type), a MediaInfo value or one without an id its site's "
                        "two, the rows put back exactly; a value under a null datavalue, not stored, is not compared")

    refused = 0
    tag = lambda s, ls: s.with_metadata({cd.KEY: __import__("json").dumps(sorted(ls)).encode()})  # noqa: E731
    one = "datavalue.numeric-id"
    cases = [
        tag(cd.reduced_schema(full, {one}), set()),  # less a leaf it does not name (untagged: the full check)
        tag(cd.reduced_schema(full, {one}), {one, "datavalue.entity-type"}),  # naming a leaf it has
        tag(cd.reduced_schema(full, {one}), {"datavalue.id"}),  # a leaf that is not derived
        tag(full, {one}),  # naming a leaf it has (all of them)
    ]
    for s in cases:
        try:
            release_schema.CLAIMS.check(s, "case", 0)
        except RuntimeError:
            refused += 1
    try:
        cd.reduced_schema(full, {"datavalue.id"})
    except ValueError:
        refused += 1
    try:
        conform_table(r, full)
    except ValueError:
        refused += 1
    failed |= not check(refused == 6, f"refused: {refused} of 6 (a schema less a leaf it does not name, naming one it "
                        "has, or one not derived; rows without their leaves to conform)")

    # A file's row groups: the third allows one leaf fewer
    bad = changed(q, "numeric-id")
    at = next(i for i in range(table.num_rows) if not bad.slice(i, 1).equals(table.slice(i, 1)))
    rows = pa.concat_tables([table.slice(0 if at >= 8_000 else at + 1, 8_000), bad.slice(at, 4_000)]).combine_chunks()
    claims_build.ARROW_ROW_GROUP_ROWS = 4_000
    out = claims_build.RowGroups(work / "rg.rowgroups")
    for batch in rows.to_batches(max_chunksize=2_500):
        out.add(pa.Table.from_batches([batch]))
    forms_before = list(out.forms)
    out.close(work / "rg.parquet")
    f = pq.ParquetFile(work / "rg.parquet")
    back = cd.restore(pq.read_table(work / "rg.parquet"), full)
    ok = f.metadata.num_row_groups == 3 and cd.dropped(f.schema_arrow) == leaves - {f"{q}.numeric-id"}
    write_table(rows, work / "rg-full.parquet")
    ok &= back.equals(pq.read_table(work / "rg-full.parquet")) and len(set(forms_before)) == 2 and len(set(out.forms)) == 1
    failed |= not check(ok, "a file's row groups: one allowing a leaf fewer than those before it has them written "
                        "again with it, the file one form, its rows the table's")
    return failed


def main() -> None:
    files = [a for a in sys.argv[1:] if a.endswith(".parquet")]
    wanted = [a for a in sys.argv[1:] if not a.endswith(".parquet")]
    run = lambda s: not wanted or s in wanted  # noqa: E731
    with TemporaryDirectory(dir=os.environ.get("TEST_TMP"), delete=not os.environ.get("KEEP")) as tmp:
        tmp = Path(tmp)
        (tmp / "hooks").mkdir()
        (tmp / "hooks" / "sitecustomize.py").write_text(HOOKS + MIX)
        lines, _ = slice_lines(files[0], files[1], inject=False)
        failed, keep = end_to_end(tmp, lines, run)
        if run("units"):
            failed |= units(keep)
    print("FAILED" if failed else "all passed", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
