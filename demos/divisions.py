"""A country's first-level divisions, from a local copy of the wikidata-pq datasets.

For each division (a German state, a US state, a French region, ...): its capital, its
population and the year it was counted, its area, and its current head of government and
their party. It reads Wikidata the way its own query service does:

- **Hops along statements**: country -> "contains the administrative territorial entity"
  (P150) -> each division's capital (P36), population (P1082), area (P2046) and head of
  government (P6) -> that person's party (P102).
- **Ranks**: a deprecated statement is wrong; of the rest, the preferred ones if any, else
  the normal ones (Wikidata's "truthy" statements).
- **Qualifiers**: a statement with an "end time" (P582) no longer holds; a population's
  "point in time" (P585) says when it was counted, and the latest wins.
- **Units**: an area is an amount and a unit (an item), converted here to km².

Every hop starts from ids already in hand, so each is a lookup in the claims, which are
sorted by id: it reads only the row groups whose id range can hold one of them, a few
MB of the 17 GB. Names come from the labels of the language, then `mul`, then `en`.

    python demos/divisions.py                  # Germany (Q183), in English
    python demos/divisions.py Q183 --lang de   # in German
    python demos/divisions.py Q30              # the United States
    python demos/divisions.py Q142 --lang fr   # France
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl
from tqdm import tqdm

DIVISION = {"P36": "capital", "P1082": "population", "P2046": "area", "P6": "head"}
RANK = {"preferred": 0, "normal": 1}
# km² per unit of area
KM2 = {"Q712226": 1.0, "Q232291": 2.589988110336, "Q35852": 0.01, "Q81292": 1e-6}
# Ids looked up per query: a batch touches up to one row group per id
BATCH = 50

dv = pl.col("datavalue").struct


def qualifier(prop: str, field: str) -> pl.Expr:
    """A field of a statement's first `prop` qualifier value, e.g. its time."""
    return (
        pl.col("qualifiers")
        .list.eval(
            pl.element()
            .filter(pl.element().struct.field("key") == prop)
            .struct.field("value")
            .list.first()
            .struct.field("datavalue")
            .struct.field(field)
        )
        .list.first()
    )


def current(df: pl.DataFrame, one: bool) -> pl.DataFrame:
    """The statements that hold now: not ended, of the best rank for their id and
    property; with `one`, only the latest of those (by point in time or start time)."""
    df = df.filter(pl.col("ended").is_null()).with_columns(
        best=pl.col("rank").replace_strict(RANK, default=len(RANK))
    )
    df = df.filter(pl.col("best") == pl.col("best").min().over("id", "property"))
    if one:
        latest = pl.coalesce("as_of", "start")
        df = df.sort(latest, descending=True, nulls_last=True).unique(
            ["id", "property"], keep="first", maintain_order=True
        )
    return df.drop("best")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("country", nargs="?", default="Q183", help="Item id (Q183)")
    parser.add_argument("--lang", default="en", help="Language code (default en)")
    parser.add_argument("--data", type=Path, default=Path("hub"), help="Local copy")
    args = parser.parse_args()
    data: Path = args.data

    claims = pl.scan_parquet(data / "claims" / "all" / "*.parquet").select(
        "id",
        "property",
        "rank",
        dv.field("id").alias("value"),
        dv.field("amount").str.strip_chars_start("+").cast(pl.Float64).alias("amount"),
        dv.field("unit").str.extract(r"(Q\d+)$").alias("unit"),
        qualifier("P585", "time").alias("as_of"),
        qualifier("P580", "time").alias("start"),
        qualifier("P582", "time").alias("ended"),
    )

    def lookup(ids: list[str], props: list[str], desc: str) -> pl.DataFrame:
        """The non-deprecated `props` statements of the ids, a batch at a time."""
        ids = sorted(set(ids))
        frames = [
            claims.filter(
                pl.col("id").is_in(ids[i : i + BATCH]),
                pl.col("property").is_in(props),
                pl.col("rank") != "deprecated",
            ).collect(engine="streaming")
            for i in tqdm(range(0, len(ids), BATCH), desc=desc, unit="batch")
        ]
        return pl.concat(frames) if frames else claims.clear().collect()

    # Hop 1: the country's divisions, and its own population
    country = current(lookup([args.country], ["P150", "P1082"], "country"), one=False)
    divisions = (
        country.filter(pl.col("property") == "P150")["value"]
        .drop_nulls()
        .unique(maintain_order=True)
        .to_list()
    )
    if not divisions:
        raise SystemExit(f"{args.country} has no current divisions (P150)")

    # Hop 2: each division's capital, population, area and head of government
    facts = current(lookup(divisions, list(DIVISION), "divisions"), one=True)

    def fact(prop: str, *cols: pl.Expr) -> pl.DataFrame:
        return facts.filter(pl.col("property") == prop).select("id", *cols)

    year = pl.col("as_of").str.slice(1, 4).alias("counted")
    table = (
        pl.DataFrame({"id": divisions})
        .join(fact("P36", pl.col("value").alias("capital")), on="id", how="left")
        .join(
            fact("P1082", pl.col("amount").cast(pl.Int64).alias("population"), year),
            on="id",
            how="left",
        )
        .join(
            fact(
                "P2046",
                (pl.col("amount") * pl.col("unit").replace_strict(KM2, default=None))
                .round(0)
                .alias("km²"),
            ),
            on="id",
            how="left",
        )
        .join(fact("P6", pl.col("value").alias("head")), on="id", how="left")
    )

    # Hop 3: each head of government's current party
    heads = table["head"].drop_nulls().to_list()
    party = (
        current(lookup(heads, ["P102"], "heads"), one=True)
        .select(pl.col("id").alias("head"), pl.col("value").alias("party"))
        if heads
        else pl.DataFrame(schema={"head": pl.String, "party": pl.String})
    )
    table = table.join(party, on="head", how="left")

    # Names of every item in the table, from the first of `langs` that has one
    langs = [
        k
        for k in dict.fromkeys([args.lang, "mul", "en"])
        if (data / "labels" / k).is_dir()
    ]
    items = [args.country, *table.select("id", "capital", "head", "party").unpivot()["value"]]
    rank = pl.col("language").replace_strict(langs, range(len(langs)), default=None)
    names = (
        pl.concat([pl.scan_parquet(data / "labels" / k / "*.parquet") for k in langs])
        .filter(pl.col("id").is_in(sorted({i for i in items if i})))
        .sort(rank, maintain_order=True)
        .unique("id", keep="first")
        .select("id", "value")
        .collect(engine="streaming")
    )
    name = dict(names.iter_rows())

    def named(col: str) -> pl.Expr:
        return pl.col(col).replace_strict(name, default=pl.col(col))

    table = table.select(
        named("id").alias("division"),
        named("capital"),
        "population",
        "counted",
        pl.col("km²").cast(pl.Int64),
        pl.when(pl.col("km²") > 0)
        .then(pl.col("population") / pl.col("km²"))
        .round(0)
        .cast(pl.Int64)
        .alias("per km²"),
        named("head"),
        named("party"),
    ).sort("population", descending=True, nulls_last=True)

    total = table["population"].sum()
    own = current(country.filter(pl.col("property") == "P1082"), one=True)
    print(f"\n{name.get(args.country, args.country)}: {table.height} divisions")
    with pl.Config(
        tbl_rows=-1,
        tbl_cols=-1,
        tbl_width_chars=200,
        fmt_str_lengths=40,
        thousands_separator=",",
        tbl_hide_dataframe_shape=True,
        tbl_hide_column_data_types=True,
    ):
        print(table)
    print(f"\nThe divisions' populations add up to {total:,}", end="")
    if own.height:
        row = own.row(0, named=True)
        counted = f" ({row['as_of'][1:5]})" if row["as_of"] else ""
        print(f"; the country's own is {row['amount']:,.0f}{counted}")
    else:
        print()


if __name__ == "__main__":
    main()
