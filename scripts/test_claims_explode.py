#!/usr/bin/env python3
"""Test the claims explodes in Arrow (partitioning/claims.py) against Polars' own.

`explode` is checked against `DataFrame.explode(col, empty_as_null=True)` on null and empty
lists, null elements, sliced, filtered and multi-chunk frames, frames read from Parquet,
and no rows. `claims_base` is checked against the Polars expression it replaced, on
synthetic claims read from Parquet: the same frame (values, nulls, row order, schema), and
byte-identical files from `partition_parquet`. `statements`, which process() lays the
claims out with, is checked against `claims_base`: on claims its Arrow flatten takes (one
batch, several, sliced) and on claims with null and empty lists, which go through
`claims_base`; and the claims file it writes, placed as the partition (`place_unsplit`),
against the file and sidecar `partition_parquet(arrow=True)` writes. Its claims file written with pyarrow
(`arrow=True`) is checked against Polars' sink: the same data, Arrow schema, row groups and
sidecar rows. Given a processed claims file, it checks those too. Runs in a temporary
directory.

Usage: python scripts/test_claims_explode.py [processed claims file]
"""

import os
import random
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from wikidata.config import UNSPLIT_COL, UNSPLIT_KEY
from wikidata.partitioning import partition_parquet, place_unsplit
from wikidata.partitioning.claims import STATEMENT_RENAMES, _statement_columns, claims_base, explode, statements
from wikidata.partitioning.core import write_table

REAL = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else None


def check(name: str, ok: bool) -> None:
    print(f"{'ok  ' if ok else 'FAIL'} {name}", flush=True)
    if not ok:
        sys.exit(1)


def polars_claims_base(lf: pl.LazyFrame) -> pl.LazyFrame:
    """claims_base before the explodes moved to Arrow."""
    return (
        lf.explode("claims", empty_as_null=True)
        .select("id", pl.col("claims").struct.unnest())
        .drop("key")
        .explode("value", empty_as_null=True)
        .select(
            "id",
            pl.col("value").name.map_fields(lambda f: STATEMENT_RENAMES.get(f, f)),
        )
        .unnest("value")
        .unnest("mainsnak")
    )


def same(a: pl.DataFrame, b: pl.DataFrame) -> bool:
    return a.schema == b.schema and a.equals(b, null_equal=True)


SNAK = pl.Struct(
    {
        "snaktype": pl.String,
        "property": pl.String,
        "datavalue": pl.Struct(
            {
                "id": pl.String,
                "amount": pl.String,
                "precision": pl.Struct({"precision__integer": pl.Int64, "precision__number": pl.Float64}),
            }
        ),
    }
)
SNAK_LISTS = pl.List(pl.Struct({"key": pl.String, "value": pl.List(SNAK)}))
STATEMENT = pl.Struct(
    {
        "mainsnak": SNAK,
        "type": pl.String,
        "id": pl.String,
        "rank": pl.String,
        "qualifiers": SNAK_LISTS,
        "references": pl.List(pl.Struct({"hash": pl.String, "snaks": SNAK_LISTS})),
    }
)
CLAIMS = pl.List(pl.Struct({"key": pl.String, "value": pl.List(STATEMENT)}))


def snak(rng: random.Random) -> dict | None:
    if rng.random() < 0.05:
        return None
    kind = rng.choice(["value", "value", "somevalue", "novalue"])
    value = None
    if kind == "value":
        value = {
            "id": rng.choice([None, f"Q{rng.randrange(10**6)}"]),
            "amount": rng.choice([None, str(rng.randrange(1000))]),
            "precision": rng.choice([None, {"precision__integer": rng.randrange(12), "precision__number": None}]),
        }
    return {"snaktype": kind, "property": f"P{rng.randrange(3000)}", "datavalue": value}


def snak_lists(rng: random.Random, most: int) -> list | None:
    """A Map<List<snak>> as kv: None, empty, or a few keys with lists (some empty)."""
    if rng.random() < 0.2:
        return None
    return [
        {"key": f"P{rng.randrange(3000)}", "value": [snak(rng) for _ in range(rng.randrange(3))]}
        for _ in range(rng.randrange(most))
    ]


def statement(rng: random.Random) -> dict:
    return {
        "mainsnak": snak(rng),
        "type": "statement",
        "id": f"Q1${rng.randrange(10**9):x}",
        "rank": rng.choice(["normal", "preferred", "deprecated", None]),
        "qualifiers": snak_lists(rng, 3),
        "references": None
        if rng.random() < 0.3
        else [
            {"hash": f"{rng.randrange(16**8):08x}", "snaks": snak_lists(rng, 3)}
            for _ in range(rng.randrange(3))
        ],
    }


def synthetic_claims(rows: int, seed: int = 0) -> pl.DataFrame:
    """Claims in the processed table's layout: per entity a list of {key, value: list of
    statements}; some entities, properties and lists empty or null."""
    rng = random.Random(seed)
    claims = []
    for _ in range(rows):
        r = rng.random()
        if r < 0.03:
            claims.append(None)  # dropped by prepare_for_partition's drop_nulls
        elif r < 0.08:
            claims.append([])
        else:
            claims.append(
                [
                    {"key": f"P{rng.randrange(3000)}", "value": [statement(rng) for _ in range(rng.randrange(4))]}
                    for _ in range(rng.randrange(1, 6))
                ]
            )
    return pl.DataFrame(
        {"id": [f"Q{i}" for i in range(rows)], "claims": claims}, schema={"id": pl.String, "claims": CLAIMS}
    )


def partition_bytes(lf: pl.LazyFrame, name: str) -> dict[str, bytes]:
    """Partition files and audit sidecar as partition_parquet writes them, by relative path."""
    for d in ("out", "audit"):
        if Path(d).exists():
            for p in sorted(Path(d).rglob("*"), reverse=True):
                p.unlink() if p.is_file() else p.rmdir()
    lf = lf.with_columns(pl.lit(UNSPLIT_KEY).alias(UNSPLIT_COL))
    partition_parquet(by=UNSPLIT_COL, lf=lf, source_name=name, dst_dir=Path("out"), log_dir=Path("audit"))
    return {str(p): p.read_bytes() for d in ("out", "audit") for p in sorted(Path(d).rglob("*.parquet"))}


def arrow_matches_sink(lf: pl.LazyFrame, name: str) -> bool:
    """The claims partition written with pyarrow holds what Polars' sink writes: the same
    files, values (nested nulls and empty lists included), Arrow schema with its
    metadata, row groups, and sidecar rows but for the files' sizes."""
    lf = lf.with_columns(pl.lit(UNSPLIT_KEY).alias(UNSPLIT_COL))
    out = {}
    for arrow in (False, True):
        dst, audit = Path(f"out_{arrow}"), Path(f"audit_{arrow}")
        partition_parquet(by=UNSPLIT_COL, lf=lf, source_name=name, dst_dir=dst, log_dir=audit, arrow=arrow)
        files = sorted(p.relative_to(dst) for p in dst.rglob("*.parquet")) if dst.exists() else []
        sidecar = pl.read_parquet(audit / name).drop("path", "file_size")
        out[arrow] = files, [dst / f for f in files], sidecar
    (files, polars, side), (arrow_files, arrow, arrow_side) = out[False], out[True]
    if files != arrow_files or not side.equals(arrow_side):
        return False
    for a, b in zip(polars, arrow):
        ta, tb = pq.read_table(a), pq.read_table(b)
        ma, mb = pq.read_metadata(a), pq.read_metadata(b)
        groups = [[m.row_group(i).num_rows for i in range(m.num_row_groups)] for m in (ma, mb)]
        if not (ta.schema.equals(tb.schema, check_metadata=True) and ta.equals(tb) and groups[0] == groups[1]):
            return False
        if not same(pl.read_parquet(a), pl.read_parquet(b)):
            return False
    return True


def main() -> None:
    with TemporaryDirectory() as tmp:
        os.chdir(tmp)

        df = pl.DataFrame(
            {
                "id": list("abcdefg"),
                "l": [[{"x": 1, "y": [1, None]}], None, [], [None, {"x": 2, "y": []}], [{"x": None, "y": None}], [], None],
                "z": list(range(7)),
            }
        )
        df.write_parquet("small.parquet")
        frames = {
            "in memory": df,
            "read from Parquet": pl.read_parquet("small.parquet"),
            "sliced": df.slice(1, 4),
            "filtered": df.filter(pl.col("z") % 2 == 0),
            "two chunks": pl.concat([df, df], rechunk=False),
            "no null or empty lists": df.filter(pl.col("l").list.len() > 0),
            "only null and empty lists": df.filter(pl.col("l").list.len().fill_null(0) == 0),
            "no rows": df.clear(),
        }
        for name, frame in frames.items():
            check(f"explode matches Polars: {name}", same(explode(frame, "l"), frame.explode("l", empty_as_null=True)))

        synthetic_claims(3000).write_parquet("claims.parquet")
        lf = pl.scan_parquet("claims.parquet").drop_nulls()
        want = polars_claims_base(lf).collect()
        got = claims_base(lf).collect()
        check(f"claims_base matches Polars on synthetic claims ({got.height:,} rows)", same(got, want))
        claims = lf.collect()["claims"]
        check(
            "synthetic claims include an empty claims list and an empty statements list",
            (claims.list.len() == 0).any() and (claims.explode().struct.field("value").list.len() == 0).any(),
        )
        check(
            "partition_parquet writes byte-identical files from synthetic claims",
            partition_bytes(claims_base(lf), "claims.parquet") == partition_bytes(polars_claims_base(lf), "claims.parquet"),
        )
        check("partition_parquet(arrow=True) writes what the sink does from synthetic claims", arrow_matches_sink(claims_base(lf), "claims.parquet"))
        check("partition_parquet(arrow=True) writes no file for no rows, as the sink", arrow_matches_sink(claims_base(lf).clear(), "none.parquet"))

        # statements: the Arrow flatten on claims it takes, claims_base on the others
        full = pq.read_table("claims.parquet")
        want = claims_base(pl.scan_parquet("claims.parquet").drop_nulls()).collect()
        check("statements matches claims_base on claims with null and empty lists", same(pl.from_arrow(statements(full)), want))
        flat = (
            pl.read_parquet("claims.parquet")
            .drop_nulls()
            .with_columns(pl.col("claims").list.eval(pl.element().filter(pl.element().struct.field("value").list.len() > 0)))
            .filter(pl.col("claims").list.len() > 0)
        )
        flat.write_parquet("flat.parquet")
        table = pq.read_table("flat.parquet")
        want = claims_base(flat.lazy()).collect()
        batches = table.to_batches()
        check("the Arrow flatten takes claims without null or empty lists", all(_statement_columns(b.column("id"), b.column("claims")) is not None for b in batches))
        tables = {
            "one batch": table.combine_chunks(),
            "two batches": pa.concat_tables([table.slice(0, 500), table.slice(500)]),
            "sliced": table.slice(100, 900),
        }
        for name, t in tables.items():
            expected = claims_base(pl.from_arrow(t).lazy()).collect()
            check(f"statements matches claims_base: {name}", same(pl.from_arrow(statements(t)), expected))
        check("statements gives claims_base's Arrow schema", statements(table).schema.equals(want.to_arrow().schema, check_metadata=True))

        # The claims file written by process(), placed as the partition
        Path("processed").mkdir()
        write_table(statements(full), Path("processed/claims.parquet"))
        place_unsplit(Path("processed/claims.parquet"), dst_dir=Path("placed"), log_dir=Path("placed_audit"))
        lf = pl.scan_parquet("claims.parquet").drop_nulls()
        partition_parquet(
            by=UNSPLIT_COL, lf=claims_base(lf).with_columns(pl.lit(UNSPLIT_KEY).alias(UNSPLIT_COL)),
            source_name="claims.parquet", dst_dir=Path("written"), log_dir=Path("written_audit"), arrow=True,
        )
        placed, written = Path("placed") / UNSPLIT_KEY / "claims.parquet", Path("written") / UNSPLIT_KEY / "claims.parquet"
        # (genson's writer, not pyarrow's, since #39: the same rows and Arrow schema, not bytes)
        got, want = pq.read_table(placed), pq.read_table(written)
        check(
            "the placed claims file has partition_parquet(arrow=True)'s rows and Arrow schema",
            got.equals(want) and got.schema.remove_metadata().equals(want.schema.remove_metadata()),
        )
        sidecars = [
            pl.read_parquet(d / "claims.parquet").drop("path", "file_size")
            for d in (Path("placed_audit"), Path("written_audit"))
        ]
        check("its sidecar has the same rows but for the path and file size", sidecars[0].equals(sidecars[1]))
        write_table(statements(full.slice(0, 0)), Path("processed/none.parquet"))
        place_unsplit(Path("processed/none.parquet"), dst_dir=Path("placed"), log_dir=Path("placed_audit"))
        check("a claims file with no rows places no partition file", not (Path("placed") / UNSPLIT_KEY / "none.parquet").exists())

        if REAL is not None:
            lf = pl.scan_parquet(REAL).drop_nulls()
            check(f"partition_parquet(arrow=True) writes what the sink does from {REAL.name}", arrow_matches_sink(claims_base(lf), REAL.name))
            check(
                f"partition_parquet writes byte-identical files from {REAL.name}",
                partition_bytes(claims_base(lf), REAL.name) == partition_bytes(polars_claims_base(lf), REAL.name),
            )
        os.chdir("/")
    print("all passed")


if __name__ == "__main__":
    main()
