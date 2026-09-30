"""The properties that hold a newspaper's ids (The Guardian topic ID, WSJ article ID, ...),
from a local copy of the wikidata-pq datasets.

A property is an entity with statements of its own, and property ids (`P...`) sort
before item ids (`Q...`), so every property's statements are at the start of the sorted
claims, and reading them reads only those row groups. A property is kept if:

- the item it is about ("Wikidata item of this property", P1629), the outlet, has a
  kind ("instance of") or a description matching `--outlets` ("newspaper" by default:
  an online newspaper, a daily newspaper, "British daily newspaper", ...), and a
  description not matching `--exclude` (sports newspapers by default, whose ids are for
  players and teams), found by a lookup of those items; or
- its own name matches `--names`, if given (e.g. `(?i)topic ID`: this also finds
  properties with no outlet item, such as BBC News topic ID, and many that are not news);

and it has a "formatter URL" (P1630): its values are ids of pages on the outlet's site.
That leaves out properties about newspapers in general ("issue", "newspaper format").

Then how many items each one is used on: a pass over the claims' `property` column,
filtered to the properties found. With `--ids-only`, only the property ids are printed,
before that pass, to pass to demos/bearers.py (see demos/newspapers.sh).

    python demos/news_ids.py
    python demos/news_ids.py --outlets '(?i)newspaper|news agency|broadcaster'
    python demos/bearers.py $(python demos/news_ids.py --ids-only)
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import polars as pl

from classes import INSTANCE, Local, named, show

SUBJECT, FORMATTER = "P1629", "P1630"
COUNTRY, ORIGIN = "P17", "P495"
OUTLETS = r"(?i)newspaper"
EXCLUDE = r"(?i)\bsports?\b"

# Property ids sort between lexemes (`L...`) and items (`Q...`)
IS_PROPERTY = (pl.col("id") >= "P") & (pl.col("id") < "Q")

dv = pl.col("datavalue").struct


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--outlets", default=OUTLETS, help="Regex on the outlet's kinds and description"
    )
    parser.add_argument(
        "--exclude", default=EXCLUDE, help="Regex on the outlet's description, to leave out"
    )
    parser.add_argument("--names", help="Regex on the property's name")
    parser.add_argument("--ids-only", action="store_true", help="Print only the ids")
    parser.add_argument("--lang", default="en", help="Language code (default en)")
    parser.add_argument("--data", type=Path, default=Path("hub"), help="Local copy")
    args = parser.parse_args()
    local = Local(args.data, args.lang)

    # Every property's outlet item and formatter URL: the first row groups only
    about = (
        local.claims.filter(IS_PROPERTY, pl.col("property").is_in([SUBJECT, FORMATTER]))
        .select(
            "id",
            "property",
            pl.coalesce(dv.field("id"), dv.field("datavalue__string")).alias("value"),
        )
        .collect(engine="streaming")
    )
    subjects = (
        about.filter(pl.col("property") == SUBJECT)
        .select("id", pl.col("value").alias("subject"))
        .drop_nulls()
    )
    formatter = (
        about.filter(pl.col("property") == FORMATTER)
        .unique("id", keep="first")
        .select("id", pl.col("value").alias("url"))
    )

    # What the outlets are, what they are described as, and where they are from: a
    # lookup of their ids
    outlets = subjects["subject"].unique().to_list()
    facts = local.statements(outlets, [INSTANCE, COUNTRY, ORIGIN])
    kind_names = local.names(set(facts.filter(pl.col("property") == INSTANCE)["value"]))
    kinds = (
        facts.filter(pl.col("property") == INSTANCE)
        .with_columns(named("value", kind_names).alias("kind"))
        .group_by("id")
        .agg(pl.col("kind").unique().sort().str.join(", "))
    )
    described = pl.DataFrame(
        list(local.descriptions(set(outlets)).items()),
        schema={"id": pl.String, "description": pl.String},
        orient="row",
    )
    news_subjects = (
        pl.DataFrame({"id": outlets})
        .join(kinds, on="id", how="left")
        .join(described, on="id", how="left")
        .filter(
            pl.col("kind").str.contains(args.outlets).fill_null(False)
            | pl.col("description").str.contains(args.outlets).fill_null(False),
            ~pl.col("description").str.contains(args.exclude).fill_null(False),
        )
        .rename({"id": "subject"})
    )

    # Properties about a matching outlet, or named like one
    property_names = local.names(set(local_property_ids(local)))
    by_subject = subjects.join(news_subjects, on="subject", how="inner")
    by_name = [
        p
        for p, name in property_names.items()
        if args.names and re.search(args.names, name)
    ]
    found = sorted((set(by_subject["id"]) | set(by_name)) & set(formatter["id"]))
    if not found:
        raise SystemExit("No properties found")
    if args.ids_only:
        print(" ".join(found))
        return

    # How many items use each: the `property` column, filtered to those found
    print(f"Counting the items using {len(found):,} properties (one pass)...")
    usage = (
        local.claims.filter(pl.col("property").is_in(found))
        .group_by("property")
        .agg(pl.col("id").n_unique().alias("items"))
        .collect(engine="streaming")
        .rename({"property": "id"})
    )

    country = (
        facts.filter(pl.col("property").is_in([COUNTRY, ORIGIN]))
        .sort("property")
        .unique("id", keep="first")
        .select(pl.col("id").alias("subject"), pl.col("value").alias("country"))
    )
    # A property's outlet: the item that matched, else its first item
    any_subject = subjects.unique("id", keep="first").rename({"subject": "any"})
    table = (
        pl.DataFrame({"id": found})
        .join(by_subject.unique("id", keep="first"), on="id", how="left")
        .join(any_subject, on="id", how="left")
        .with_columns(pl.coalesce("subject", "any").alias("subject"))
        .join(country, on="subject", how="left")
        .join(formatter, on="id", how="left")
        .join(usage, on="id", how="left")
        .with_columns(
            pl.col("items").fill_null(0),
            found_by=pl.when(pl.col("id").is_in(by_subject["id"].implode()))
            .then(pl.lit("its item"))
            .otherwise(pl.lit("its name")),
        )
        .sort("items", "id", descending=[True, False])
    )
    names = property_names | local.names(
        set(table["subject"].drop_nulls()) | set(table["country"].drop_nulls())
    )

    by_item = table.filter(pl.col("found_by") == "its item").height
    print(
        f"\n{table.height:,} properties with a formatter URL: {by_item:,} about a "
        f"matching outlet by their item, {table.height - by_item:,} by their name only"
    )
    show(
        table.select(
            "id",
            named("id", names).alias("name"),
            "items",
            named("subject", names).alias("outlet"),
            "kind",
            "description",
            named("country", names),
            "found_by",
            "url",
        )
    )


def local_property_ids(local: Local) -> list[str]:
    """Every property id: those with any statement, from the first row groups."""
    return (
        local.claims.filter(IS_PROPERTY)
        .select("id")
        .unique()
        .collect(engine="streaming")["id"]
        .to_list()
    )


if __name__ == "__main__":
    main()
