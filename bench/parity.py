"""Whether two kept runs of pipeline.py (KEEP=1) published the same data, compared by
fingerprint, so a run can be checked against an earlier one's outputs without them.

A run's fingerprint:

- **Published files:** for each table and key, finalise's sorted part files read in
  order: the row count and two sums of position-seeded row hashes (as
  `compact._fingerprint`), so any changed, missing, extra or reordered row changes it, and
  the SHA-256 of the Arrow schema with its metadata. What a release publishes, exactly.
- **Group files:** for each table and partition key, the key's staged files: the row count
  and two sums of row hashes without positions (the same rows, in any order), and the
  schema's SHA-256. Their order is the intermediate layout, which the published files'
  order (each key stably sorted by id) checks wherever it shows: a change in the order
  of an id's rows changes the published files. Whether their order differs is reported.
- **Quarantine:** each file's SHA-256.
- **Audit sidecars:** each column's SHA-256, the run directory stripped from `path`, and
  the key's directory too (a key's file of a chunk, `{table}/{key}/{chunk}`, and the one
  file of the chunk the split builds its keys' row groups in, `{table}/{chunk}`, name the
  same rows), but `file_size`: a partition file's bytes depend on how its writer laid out
  the same rows.
- **The claims, by the coordinate rule** (`canonical`, bench 17.5.0): each claims key's
  published and group files fingerprinted again with every globe-coordinate's latitude,
  longitude and precision in `__number` and a record of nulls there null (the rule of
  wikidata/claims_numeric.py, made here apart from it, in Polars), and the cells the rule
  would move counted (`moved`). A commit before the rule and one after it differ only in
  those cells if their claims are the same by the rule, the commit after's already are
  (the rule moves none of its cells), and nothing else differs: parity then passes, the
  differing keys and the cells listed apart (`intended`). Once both commits keep the rule
  this is the plain comparison, as the rule changes neither's files.
- **The claims' group files without their derived leaves** (bench 17.14.0): each claims
  key's group files fingerprinted again (rows, unordered and ordered, and schema) with
  every datavalue's `entity-type` and `numeric-id` left out, wherever a struct has them
  beside an `id` (made here in pyarrow, apart from wikidata/claims_derived.py, each file
  rewritten so to a temporary file). A commit whose claims group files are without those
  leaves (they are a function of the `id`, put back by finalise) and one with them differ
  only there: where the exact comparison fails, parity passes if the two runs are the
  same with the claims group files compared so, and everything else exactly (the
  published files above all, byte for byte in their rows and schema). The report then
  lists the differing group keys (`derived_leaves`).

The row hashes are Polars', so fingerprints compare only when made with the same Polars
version, which each holds.

Usage: parity.py fingerprint <run.keep> <out.json>
       parity.py compare <a.json> <b.json>   (prints a JSON report; exits 1 if not the same)
"""

import hashlib
import json
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

MAY_DIFFER = {"file_size"}
sha = lambda b: hashlib.sha256(b).hexdigest()
GLOBE = "globecoordinate"
# The leaves a datavalue's `id` spells out (wikidata/claims_derived.py), and their files'
# metadata key naming those a file is without
DERIVED = ("entity-type", "numeric-id")
DERIVED_KEY = b"wikidata.derived_leaves"
NUMBERS = ("latitude", "longitude", "precision")


def _snak_rule(snak, dtype):
    """A snak (an expression of a struct of `dtype`) by the coordinate rule: each
    globe-coordinate's latitude, longitude and precision in `__number` (an integer as a
    Float64), its `__integer` null, and a record of nulls null; and an expression of
    each number's cells the rule moves there ({name: count}, summed by the caller)."""
    import polars as pl

    fields = {f.name: f.dtype for f in dtype.fields}
    if "datavalue_type" not in fields or "datavalue" not in fields:
        return snak, {}
    dv_types = {f.name: f.dtype for f in fields["datavalue"].fields}
    dv = snak.struct.field("datavalue")
    globe = (snak.struct.field("datavalue_type") == GLOBE).fill_null(False) & dv.is_not_null()
    out, moved = [], {}
    for name in (n for n in NUMBERS if n in dv_types):
        x = dv.struct.field(name)
        ints, nums = x.struct.field(f"{name}__integer"), x.struct.field(f"{name}__number")
        # (no literals: Polars 2.0 panics on a struct of a literal and a column of several
        # chunks, as a scan of several row groups gives; a `when` without `otherwise` is null)
        as_number = pl.struct([
            (pl.coalesce(nums, ints.cast(pl.Float64)) if f.name == f"{name}__number" else
             pl.when(ints.is_null()).then(ints) if f.name == f"{name}__integer" else x.struct.field(f.name)).alias(f.name)
            for f in dv_types[name].fields])
        here = globe & x.is_not_null()
        empty = ints.is_null() & nums.is_null()
        out.append(pl.when(~(here & empty)).then(pl.when(here & ints.is_not_null()).then(as_number).otherwise(x))
                   .alias(name))
        moved[f"{name}: an integer"] = here & ints.is_not_null()
        moved[f"{name}: a record of nulls"] = here & empty
    rule = pl.when(globe).then(snak.struct.with_fields(dv.struct.with_fields(out).alias("datavalue"))).otherwise(snak)
    return rule, moved


def _groups_rule(groups, dtype):
    """A list of {key, value: [snak]} (qualifiers, a reference's snaks) by the rule, and the
    cells it moves there, a count a row."""
    import polars as pl

    snak = {f.name: f.dtype for f in dtype.inner.fields}["value"].inner
    rule, moved = _snak_rule(pl.element(), snak)
    snaks = lambda e: groups.list.eval(pl.element().struct.field("value").list.eval(e).list.sum()).list.sum()  # noqa: E731
    return (groups.list.eval(pl.element().struct.with_fields(pl.element().struct.field("value").list.eval(rule))),
            {k: snaks(e) for k, e in moved.items()})


def canonical(lf):
    """A claims partition frame (one row per statement, its mainsnak's fields as columns)
    by the coordinate rule; and the cells the rule moves, an expression each, by place.

    Run on a scan with the streaming engine (as `fingerprint` runs it), or on a frame of one
    chunk: Polars 2.0 panics on a `when` over struct columns of several chunks."""
    import polars as pl

    schema = lf.collect_schema()
    if "datavalue" not in schema or "datavalue_type" not in schema:
        return lf, {}
    columns, moved = {}, {}
    main = pl.Struct({k: schema[k] for k in ("datavalue", "datavalue_type")})
    rule, m = _snak_rule(pl.struct("datavalue", "datavalue_type"), main)
    columns["datavalue"] = rule.struct.field("datavalue")
    moved |= {f"mainsnak {k}": e.sum() for k, e in m.items()}
    if "qualifiers" in schema:
        columns["qualifiers"], m = _groups_rule(pl.col("qualifiers"), schema["qualifiers"])
        moved |= {f"qualifier {k}": e.sum() for k, e in m.items()}
    if "references" in schema:
        snaks = {f.name: f.dtype for f in schema["references"].inner.fields}["snaks"]
        rule, m = _groups_rule(pl.element().struct.field("snaks"), snaks)
        columns["references"] = pl.col("references").list.eval(pl.element().struct.with_fields(rule.alias("snaks")))
        refs = lambda e: pl.col("references").list.eval(e).list.sum()  # noqa: E731
        moved |= {f"reference {k}": refs(e).sum() for k, e in m.items()}
    return lf.with_columns(e.alias(k) for k, e in columns.items()), moved


def _strip_type(t):
    """A type without its structs' DERIVED fields beside an `id`."""
    import pyarrow as pa

    if pa.types.is_list(t) or pa.types.is_large_list(t):
        f = t.value_field
        inner = f.with_type(_strip_type(f.type))
        return pa.large_list(inner) if pa.types.is_large_list(t) else pa.list_(inner)
    if not pa.types.is_struct(t):
        return t
    names = {t.field(i).name for i in range(t.num_fields)}
    drop = set(DERIVED) if "id" in names else set()
    return pa.struct([t.field(i).with_type(_strip_type(t.field(i).type)) for i in range(t.num_fields)
                      if t.field(i).name not in drop])


def _strip(arr, t):
    """An array as `_strip_type`'s type of its own: the same values, the fields left out."""
    import pyarrow as pa
    import pyarrow.compute as pc

    mask = pc.is_null(arr) if arr.null_count else None
    if pa.types.is_list(arr.type) or pa.types.is_large_list(arr.type):
        cls = pa.LargeListArray if pa.types.is_large_list(arr.type) else pa.ListArray
        offsets = arr.offsets
        start, end = offsets[0].as_py(), offsets[-1].as_py()  # (from_arrays takes offsets from 0)
        values = arr.values.slice(start, end - start)
        return cls.from_arrays(pc.subtract(offsets, offsets[0]), _strip(values, t.value_type), type=t, mask=mask)
    if not pa.types.is_struct(arr.type):
        return arr
    keep = [t.field(i).name for i in range(t.num_fields)]
    children = [_strip(arr.field(n), t.field(n).type) for n in keep]
    return pa.StructArray.from_arrays(children, fields=list(t), mask=mask)


def stripped(files: list[Path], tmp: Path) -> list[Path]:
    """Each file rewritten without its derived leaves (`_strip`), a row group at a time, its
    schema's metadata but DERIVED_KEY kept: the same rows and schema whether a file had
    the leaves or was written without them."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    out = []
    for i, src in enumerate(files):
        pf = pq.ParquetFile(src)
        meta = {k: v for k, v in (pf.schema_arrow.metadata or {}).items() if k != DERIVED_KEY} or None
        schema = pa.schema([f.with_type(_strip_type(f.type)) for f in pf.schema_arrow], metadata=meta)
        dst = tmp / f"{i:05d}.parquet"
        with pq.ParquetWriter(dst, schema) as w:
            for g in range(pf.metadata.num_row_groups):
                rg = pf.read_row_group(g)
                cols = [pa.chunked_array([_strip(c, f.type) for c in rg.column(f.name).chunks], f.type) for f in schema]
                w.write_table(pa.Table.from_arrays(cols, schema=schema))
        out.append(dst)
    return out


def group_files(root: Path) -> dict:
    """table/key -> its group files, in group order."""
    out = defaultdict(list)
    for p in sorted((root / "staging").rglob("*.parquet")):
        group, table, key = p.parts[-4], p.parts[-3], p.parts[-2]
        out[f"{table}/{key}"].append((group, p))
    return {k: [p for _, p in sorted(v)] for k, v in out.items()}


def fingerprint(root: Path) -> dict:
    import polars as pl
    import pyarrow.parquet as pq

    def rows(files: list[Path], ordered: bool = True, rule: bool = False) -> list[int]:
        lf = pl.scan_parquet(files)
        if rule:
            lf = canonical(lf)[0]
        if ordered:
            lf = lf.with_row_index("_row")
        row = pl.struct(pl.all())
        df = (
            lf.select(pl.len().alias("n"), row.hash(seed=0).sum().alias("h0"), row.hash(seed=1).sum().alias("h1"))
            .collect(engine="streaming")
        )
        return [int(df["n"][0]), int(df["h0"][0]), int(df["h1"][0])]

    schema = lambda f: sha(pq.read_schema(f).serialize().to_pybytes())
    final = defaultdict(list)
    for p in sorted((root / "final").rglob("*.parquet")):
        final[str(p.parent.relative_to(root / "final"))].append(p)

    def strip(d: pl.DataFrame) -> pl.DataFrame:
        d = d.with_columns(pl.col("path").str.replace(r"^.*/results/", "results/"))
        key = d.columns[3]  # (path, num_rows, file_size, key, min_id, max_id)
        drop = lambda p, k: p if (parts := p.rsplit("/", 2))[1:2] != [k] else f"{parts[0]}/{parts[2]}"
        return d.with_columns(pl.Series("path", [drop(p, k) for p, k in zip(d["path"], d[key])], pl.String))

    sidecars = {}
    for p in sorted((root / "audit").rglob("*.parquet")):
        d = strip(pl.read_parquet(p))
        sidecars[str(p.relative_to(root / "audit"))] = {
            "height": d.height,
            "columns": {c: sha(json.dumps(d[c].to_list(), default=str).encode()) for c in d.columns},
        }
    def moved(files: list[Path]) -> dict:
        exprs = canonical(pl.scan_parquet(files))[1]
        if not exprs:
            return {}
        df = pl.scan_parquet(files).select(**exprs).collect(engine="streaming")
        return {k: int(v) for k, v in df.row(0, named=True).items()}

    claims_keys = {k: fs for k, fs in group_files(root).items() if k.split("/")[0] == "claims"}
    claims_final = {k: fs for k, fs in final.items() if k.split("/")[0] == "claims"}
    derived = {"keys": {}, "keys_ordered": {}}
    with tempfile.TemporaryDirectory(dir=root) as tmp:
        for k, fs in claims_keys.items():
            d = Path(tmp) / k.replace("/", "_")
            d.mkdir()
            fs = stripped(fs, d)
            derived["keys"][k] = rows(fs, ordered=False) + [schema(fs[0])]
            derived["keys_ordered"][k] = rows(fs)
    return {
        "derived": derived,
        "canonical": {
            "keys": {k: rows(fs, ordered=False, rule=True) for k, fs in claims_keys.items()},
            "keys_ordered": {k: rows(fs, rule=True) for k, fs in claims_keys.items()},
            "final": {k: rows(fs, rule=True) for k, fs in claims_final.items()},
            "moved": {k: moved(fs) for k, fs in claims_final.items()},
        },
        "polars": pl.__version__,
        "keys": {k: rows(fs, ordered=False) + [schema(fs[0])] for k, fs in group_files(root).items()},
        "keys_ordered": {k: rows(fs) for k, fs in group_files(root).items()},
        "final": {k: rows(fs) + [schema(fs[0])] for k, fs in final.items()},
        "quarantine": {
            str(p.relative_to(root)): sha(p.read_bytes()) for p in sorted((root / "quarantine").rglob("*.parquet"))
        },
        "sidecars": sidecars,
    }


def without_derived(fp: dict) -> dict:
    """A fingerprint with its claims keys' group files as they are without their derived
    leaves (the published files as they are)."""
    d = fp["derived"]
    return {**fp, "keys": {**fp["keys"], **d["keys"]}, "keys_ordered": {**fp["keys_ordered"], **d["keys_ordered"]}}


def by_rule(fp: dict) -> dict:
    """A fingerprint with its claims keys' rows as the coordinate rule makes them (each
    file's schema kept: the rule changes no type)."""
    c = fp["canonical"]
    keep = lambda raw, new, n: {k: (new[k] + v[n:] if k in new else v) for k, v in raw.items()}  # noqa: E731
    return {**fp, "keys": keep(fp["keys"], c["keys"], 3), "keys_ordered": keep(fp["keys_ordered"], c["keys_ordered"], 3),
            "final": keep(fp["final"], c["final"], 3)}


def compare(a: dict, b: dict) -> dict:
    """The comparison (`exact`), and where it fails, the same by the coordinate rule (see
    the module's docstring)."""
    report = exact(a, b)
    if report["ok"]:
        return report
    if "derived" in a and "derived" in b:
        derived = exact(without_derived(a), without_derived(b))
        if derived["ok"]:
            return {**derived, "derived_leaves": {
                "rule": "the claims group files compared without each datavalue's entity-type and numeric-id "
                        "(a function of its id, put back by finalise: claims_derived.py)",
                "keys_differing": report["keys_differing"], "keys_reordered": report["keys_reordered"],
                "every_other_value": "the same: the published files exactly, every other table, file and sidecar"}}
    if "canonical" not in a or "canonical" not in b:
        return report
    rule = exact(by_rule(a), by_rule(b))
    # (the commit after keeps the rule: by it, its files are as they are, and no cell moves)
    kept = by_rule(b)["keys"] == b["keys"] and by_rule(b)["final"] == b["final"] and not any(
        n for m in b["canonical"]["moved"].values() for n in m.values())
    if not (rule["ok"] and kept):
        return {**report, "by_the_coordinate_rule": {"ok": rule["ok"], "after_keeps_it": kept}}
    moved = lambda fp: {k: {c: n for c, n in m.items() if n} for k, m in fp["canonical"]["moved"].items()  # noqa: E731
                        if any(m.values())}
    return {**rule, "intended": {
        "rule": "a globe-coordinate's latitude, longitude and precision in __number (claims_numeric.py)",
        "keys_differing": report["keys_differing"], "published_differing": report["published_differing"],
        "keys_reordered": report["keys_reordered"],
        "cells_moved_before": moved(a), "cells_moved_after": moved(b),
        "every_other_value": "the same: the claims the same by the rule, every other table, file and sidecar as before"}}


def exact(a: dict, b: dict) -> dict:
    keys_differing = sorted(k for k in set(a["keys"]) | set(b["keys"]) if a["keys"].get(k) != b["keys"].get(k))
    final_differing = sorted(k for k in set(a["final"]) | set(b["final"]) if a["final"].get(k) != b["final"].get(k))
    order_differing = sorted(
        k for k in set(a["keys_ordered"]) & set(b["keys_ordered"])
        if a["keys_ordered"][k] != b["keys_ordered"][k] and k not in keys_differing
    )
    sidecar_diff = Counter()
    for rel in set(a["sidecars"]) & set(b["sidecars"]):
        x, y = a["sidecars"][rel], b["sidecars"][rel]
        table = rel.split("/")[0]
        if x["height"] != y["height"] or set(x["columns"]) != set(y["columns"]):
            sidecar_diff[f"{table}: shape"] += 1
            continue
        for c in x["columns"]:
            if x["columns"][c] != y["columns"][c]:
                sidecar_diff[f"{table}.{c}"] += 1
    report = {
        "keys": len(b["keys"]),
        "keys_per_table": dict(Counter(k.split("/")[0] for k in b["keys"])),
        "keys_differing": keys_differing,
        "published_keys": len(b["final"]),
        "published_differing": final_differing,
        "keys_reordered": order_differing,
        "quarantine_files": len(b["quarantine"]),
        "quarantine_same": a["quarantine"] == b["quarantine"],
        "sidecars": len(b["sidecars"]),
        "sidecars_same_set": set(a["sidecars"]) == set(b["sidecars"]),
        "sidecar_columns_differing": dict(sidecar_diff),
    }
    report["ok"] = (
        not keys_differing
        and not final_differing
        and len(b["final"]) > 0
        and report["quarantine_same"]
        and report["sidecars_same_set"]
        and all(k.split(".")[-1] in MAY_DIFFER for k in sidecar_diff)
    )
    return report


def main() -> None:
    if sys.argv[1] == "fingerprint":
        Path(sys.argv[3]).write_text(json.dumps(fingerprint(Path(sys.argv[2]))))
    else:
        a, b = (json.loads(Path(p).read_text()) for p in sys.argv[2:4])
        report = compare(a, b)
        print(json.dumps(report))
        sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
