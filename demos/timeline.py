"""The history of a place (or any item) in its own statements, from a local copy of the
wikidata-pq datasets: London (Q84) by default.

Many statements carry a date: in their value (London's "inception", in 47 AD), or in a
qualifier ("point in time" P585, "start time" P580, "end time" P582), such as a mayor's
term or the year of a population count. This lays them out in date order, naming every
property and item, with population counts as a bar chart.

A "significant event" (P793) is often given without a date on London's statement: its
date is on the event's own item (the Great Fire of London's "point in time", 1666), so
those events are looked up too, as one batch.

The claims are sorted by id, so both lookups read only the row groups that can hold the
ids, and the names come from claims_labels in the same way (by `ref`). Names are in the
language, then `mul`, then `en`.

    python demos/timeline.py                # London
    python demos/timeline.py --lang fr      # London, named in French
    python demos/timeline.py Q1741          # Vienna
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

POINT, START, END = "P585", "P580", "P582"
EVENT = "P793"
POPULATION = "P1082"
# Properties of an event's own item that date it, in order of preference
EVENT_DATES = [POINT, START, "P571"]
RANK = {"preferred": 0, "normal": 1}
BAR = 40

dv = pl.col("datavalue").struct


def qualifier(prop: str) -> pl.Expr:
    """The datavalue of a statement's first `prop` qualifier."""
    return (
        pl.col("qualifiers")
        .list.eval(
            pl.element()
            .filter(pl.element().struct.field("key") == prop)
            .struct.field("value")
            .list.first()
            .struct.field("datavalue")
        )
        .list.first()
    )


def date(value: pl.Expr) -> pl.Expr:
    """A time datavalue as text to its precision: `1666`, `1666-09`, `1666-09-02`, `47`
    for AD 47, `c. 1100` for a decade or coarser, `500 BC`."""
    time = value.struct.field("time")
    precision = value.struct.field("precision").struct.field("precision__integer")
    year = time.str.extract(r"^([+-]?\d+)-").cast(pl.Int64)
    year_text = (
        pl.when(year < 0)
        .then(pl.format("{} BC", -year))
        .otherwise(year.cast(pl.String))
    )
    return (
        pl.when(time.is_null())
        .then(None)
        .when((precision >= 11) & (year > 0))
        .then(time.str.slice(1, 10))
        .when((precision == 10) & (year > 0))
        .then(time.str.slice(1, 7))
        .when(precision < 9)
        .then(pl.concat_str(pl.lit("c. "), year_text))
        .otherwise(year_text)
    )


def sort_key(value: pl.Expr) -> pl.Expr:
    """A time datavalue as an orderable string: sign-aware year, then the rest."""
    time = value.struct.field("time")
    year = time.str.extract(r"^([+-]?\d+)-").cast(pl.Int64)
    return pl.format("{}{}", (year + 100_000).cast(pl.String).str.zfill(6), time.str.slice(5))


def best(df: pl.DataFrame, *by: str) -> pl.DataFrame:
    """Rows of the best rank for their `by` (preferred, else normal)."""
    df = df.with_columns(r=pl.col("rank").replace_strict(RANK, default=len(RANK)))
    return df.filter(pl.col("r") == pl.col("r").min().over(*by)).drop("r")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("item", nargs="?", default="Q84", help="Item id (default Q84)")
    parser.add_argument("--lang", default="en", help="Language code (default en)")
    parser.add_argument("--data", type=Path, default=Path("hub"), help="Local copy")
    args = parser.parse_args()
    data: Path = args.data

    def keys(table: str, *wanted: str) -> list[str]:
        return [k for k in dict.fromkeys(wanted) if (data / table / k).is_dir()]

    langs = keys("labels", args.lang, "mul", "en")

    def scan(table: str, *wanted: str) -> pl.LazyFrame:
        return pl.concat(
            [pl.scan_parquet(data / table / k / "*.parquet") for k in keys(table, *wanted)]
        )

    def first(lf: pl.LazyFrame, *by: str) -> pl.LazyFrame:
        """One row per `by`, from the first language in `langs` that has one."""
        rank = pl.col("language").replace_strict(langs, range(len(langs)), default=None)
        return lf.sort(rank, maintain_order=True).unique(
            by, keep="first", maintain_order=True
        )

    claims = pl.scan_parquet(data / "claims" / "all" / "*.parquet").filter(
        pl.col("rank") != "deprecated"
    )

    # Lookup 1: the item's statements, with the dates in their values and qualifiers
    statements = (
        claims.filter(pl.col("id") == args.item)
        .filter(
            (pl.col("datatype") != "monolingualtext")
            | dv.field("language").is_in(langs)
        )
        .select(
            "property",
            "rank",
            dv.field("id").alias("value_id"),
            pl.coalesce(dv.field("text"), dv.field("datavalue__string")).alias("text"),
            dv.field("amount").str.strip_chars_start("+").alias("amount"),
            dv.field("unit").str.extract(r"(Q\d+)$").alias("unit"),
            pl.when(pl.col("datatype") == "time").then("datavalue").alias("own"),
            qualifier(POINT).alias("point"),
            qualifier(START).alias("start"),
            qualifier(END).alias("end"),
        )
        .collect(engine="streaming")
    )
    if statements.is_empty():
        raise SystemExit(f"No statements for {args.item} in {data}")

    # Lookup 2: the dates of undated significant events, from the events' own items
    undated = statements.filter(
        (pl.col("property") == EVENT)
        & pl.all_horizontal(pl.col("point", "start", "own").is_null())
    )
    events = undated["value_id"].drop_nulls().unique().sort().to_list()
    event_dates = (
        best(
            claims.filter(
                pl.col("id").is_in(events),
                pl.col("property").is_in(EVENT_DATES),
                pl.col("datatype") == "time",
            )
            .select("id", "property", "rank", "datavalue")
            .collect(engine="streaming"),
            "id",
            "property",
        )
        .with_columns(
            order=pl.col("property").replace_strict(EVENT_DATES, range(len(EVENT_DATES)))
        )
        .sort("order")
        .unique("id", keep="first")
        .select(pl.col("id").alias("value_id"), pl.col("datavalue").alias("event"))
    )
    statements = statements.join(event_dates, on="value_id", how="left").with_columns(
        when=pl.coalesce("own", "point", "start", "event")
    )

    # Names of the properties, items and units, from claims_labels
    refs = pl.concat(
        [statements["property"], statements["value_id"], statements["unit"]]
    ).drop_nulls().unique()
    names = first(
        scan("claims_labels", *langs).filter(pl.col("ref").is_in(refs.implode())),
        "field",
        "ref",
    ).collect(engine="streaming")

    def name(field: str, col: str) -> pl.DataFrame:
        return names.filter(pl.col("field") == field).select(
            pl.col("ref").alias(col), pl.col("label").alias(f"{col}_name")
        )

    dated = (
        statements.filter(pl.col("when").is_not_null())
        .join(name("property-labels", "property"), on="property", how="left")
        .join(name("labels", "value_id"), on="value_id", how="left")
        .join(name("unit-labels", "unit"), on="unit", how="left")
        .with_columns(
            date=date(pl.col("when")),
            until=date(pl.col("end")),
            key=sort_key(pl.col("when")),
            what=pl.coalesce("property_name", "property"),
            value=pl.coalesce(
                "value_id_name",
                "text",
                pl.when(pl.col("amount").is_not_null()).then(
                    pl.concat_str("amount", "unit_name", separator=" ", ignore_nulls=True)
                ),
                "value_id",
            ),
        )
    )
    population = best(
        dated.filter(pl.col("property") == POPULATION).with_columns(
            pl.col("amount").cast(pl.Float64).cast(pl.Int64)
        ),
        "key",
    ).unique("key", keep="first").sort("key")
    timeline = (
        dated.filter(pl.col("property") != POPULATION)
        .sort("key", "what", "value", nulls_last=True)
        .select("date", "what", "value", "until")
    )

    label = first(scan("labels", *langs).filter(pl.col("id") == args.item), "id")
    desc = first(scan("descriptions", *langs).filter(pl.col("id") == args.item), "id")
    title = label.collect()["value"].to_list() or [args.item]
    about = desc.collect()["value"].to_list()
    print(f"{title[0]} ({args.item}){': ' + about[0] if about else ''}\n")

    print(f"{timeline.height} dated statements, of {statements.height}:\n")
    with pl.Config(
        tbl_rows=-1,
        fmt_str_lengths=50,
        tbl_hide_dataframe_shape=True,
        tbl_hide_column_data_types=True,
    ):
        print(timeline)

    if population.height:
        top = population["amount"].max()
        print(f"\nPopulation, {population.height} counts:\n")
        for row in population.iter_rows(named=True):
            bar = "█" * max(1, round(BAR * row["amount"] / top))
            print(f"{row['date']:>10}  {row['amount']:>12,}  {bar}")


if __name__ == "__main__":
    main()
