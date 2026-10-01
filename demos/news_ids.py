"""The properties for news websites' topic pages (The Guardian topic ID, BBC News topic
ID, ...), from a local copy of the wikidata-pq datasets.

Wikidata classifies its properties: those whose values are a news website's topic pages
are instances of "Wikidata property to identify news website topics" (Q105946994), so an
item holding one is a topic of that site's coverage. Properties for a newspaper's
articles (WSJ article ID), writers (The Atlantic author ID) or archived issues
(Chronicling America) are not, so they are left out: their items are articles, people
and newspapers, not what the news is about.

A property is an entity with statements of its own, and property ids (`P...`) sort
before item ids (`Q...`), so every property's statements are at the start of the sorted
claims, and reading them reads only those row groups. Then a lookup of each property's
outlet ("Wikidata item of this property", P1629) for its country, and how many items use
each property: a pass over the claims' `property` column, filtered to those found. With
`--ids-only`, only the property ids are printed, before that pass, to pass to
demos/bearers.py (see demos/news_topics.sh).

    python demos/news_ids.py
    python demos/news_ids.py --of Q52063969    # properties to identify topics, of any site
    python demos/bearers.py $(python demos/news_ids.py --ids-only)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from classes import INSTANCE, Local, named, show

NEWS_TOPICS = "Q105946994"
SUBJECT, FORMATTER = "P1629", "P1630"
COUNTRY, ORIGIN = "P17", "P495"
# Property ids sort between lexemes (`L...`) and items (`Q...`)
IS_PROPERTY = (pl.col("id") >= "P") & (pl.col("id") < "Q")

dv = pl.col("datavalue").struct


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--of", default=NEWS_TOPICS, help="Class of properties (default Q105946994)"
    )
    parser.add_argument("--ids-only", action="store_true", help="Print only the ids")
    parser.add_argument("--lang", default="en", help="Language code (default en)")
    parser.add_argument("--data", type=Path, default=Path("hub"), help="Local copy")
    args = parser.parse_args()
    local = Local(args.data, args.lang)

    # The properties in the class, with their outlets and formatter URLs: the first
    # row groups only
    about = (
        local.claims.filter(
            IS_PROPERTY, pl.col("property").is_in([INSTANCE, SUBJECT, FORMATTER])
        )
        .select(
            "id",
            "property",
            pl.coalesce(dv.field("id"), dv.field("datavalue__string")).alias("value"),
        )
        .drop_nulls()
        .collect(engine="streaming")
    )
    found = sorted(
        set(
            about.filter(pl.col("property") == INSTANCE, pl.col("value") == args.of)["id"]
        )
    )
    if not found:
        raise SystemExit(f"No properties are instances of {args.of}")
    if args.ids_only:
        print(" ".join(found))
        return
    about = about.filter(pl.col("id").is_in(found))

    def first(prop: str, alias: str) -> pl.DataFrame:
        return (
            about.filter(pl.col("property") == prop)
            .unique("id", keep="first")
            .select("id", pl.col("value").alias(alias))
        )

    outlets = first(SUBJECT, "outlet")
    facts = local.statements(outlets["outlet"].to_list(), [COUNTRY, ORIGIN])
    country = (
        facts.sort("property")
        .unique("id", keep="first")
        .select(pl.col("id").alias("outlet"), pl.col("value").alias("country"))
    )

    # How many items use each: the `property` column, filtered to those found
    print(f"Counting the items using {len(found):,} properties (one pass)...")
    usage = (
        local.claims.filter(pl.col("property").is_in(found))
        .group_by("property")
        .agg(pl.col("id").n_unique().alias("items"))
        .collect(engine="streaming")
        .rename({"property": "id"})
    )

    table = (
        pl.DataFrame({"id": found})
        .join(outlets, on="id", how="left")
        .join(country, on="outlet", how="left")
        .join(first(FORMATTER, "url"), on="id", how="left")
        .join(usage, on="id", how="left")
        .with_columns(pl.col("items").fill_null(0))
        .sort("items", "id", descending=[True, False])
    )
    names = local.names(
        set(found)
        | {args.of}
        | set(table["outlet"].drop_nulls())
        | set(table["country"].drop_nulls())
    )

    print(f"\n{table.height:,} properties are instances of {names.get(args.of, args.of)}:")
    show(
        table.select(
            "id",
            named("id", names).alias("name"),
            "items",
            named("outlet", names),
            named("country", names),
            "url",
        )
    )


if __name__ == "__main__":
    main()
