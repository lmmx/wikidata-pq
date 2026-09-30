"""The properties that hold a news or media outlet's ids (The Guardian topic ID, BBC News
topic ID, ...), from a local copy of the wikidata-pq datasets.

A property is an entity with statements of its own, and property ids (`P...`) sort
before item ids (`Q...`), so every property's statements are at the start of the sorted
claims, and reading them reads only those row groups. A property is kept if:

- the item it is about ("Wikidata item of this property", P1629) is an instance of a
  kind whose name matches `--kinds` (newspaper, news website, broadcaster, ...), found
  by a lookup of those items; or
- its own name matches `--names` (e.g. "... topic ID"), for those without such an item.

Then how many items each one is used on: a pass over the claims' `property` column,
filtered to the properties found.

    python demos/news_ids.py
    python demos/news_ids.py --kinds '(?i)magazine' --names '^$'
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import polars as pl

from classes import INSTANCE, Local, named, show

SUBJECT, FORMATTER = "P1629", "P1630"
COUNTRY, ORIGIN = "P17", "P495"
KINDS = r"(?i)news|broadcast|magazine|television (channel|network)|radio station|press agency"
NAMES = r"(?i)\btopic ID\b|\bnews\b"

# Property ids sort between lexemes (`L...`) and items (`Q...`)
IS_PROPERTY = (pl.col("id") >= "P") & (pl.col("id") < "Q")

dv = pl.col("datavalue").struct


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--kinds", default=KINDS, help="Regex on the subject's kinds")
    parser.add_argument("--names", default=NAMES, help="Regex on the property's name")
    parser.add_argument("--lang", default="en", help="Language code (default en)")
    parser.add_argument("--data", type=Path, default=Path("hub"), help="Local copy")
    args = parser.parse_args()
    local = Local(args.data, args.lang)

    # Every property's subject item and formatter URL: the first row groups only
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

    # What the subject items are, and where from: a lookup of their ids
    facts = local.statements(
        subjects["subject"].unique().to_list(), [INSTANCE, COUNTRY, ORIGIN]
    )
    kind_names = local.names(set(facts.filter(pl.col("property") == INSTANCE)["value"]))
    news_subjects = (
        facts.filter(pl.col("property") == INSTANCE)
        .with_columns(named("value", kind_names).alias("kind"))
        .filter(pl.col("kind").str.contains(args.kinds))
        .group_by("id")
        .agg(pl.col("kind").unique().sort().str.join(", "))
        .rename({"id": "subject"})
    )

    # Properties about a news or media outlet, or named like one
    property_names = local.names(set(local_property_ids(local)))
    by_subject = subjects.join(news_subjects, on="subject", how="inner")
    by_name = [p for p, name in property_names.items() if re.search(args.names, name)]
    found = sorted(set(by_subject["id"]) | set(by_name))
    if not found:
        raise SystemExit("No properties found")

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
            found_by=pl.when(pl.col("kind").is_not_null())
            .then(pl.lit("its item"))
            .otherwise(pl.lit("its name")),
        )
        .sort("items", "id", descending=[True, False])
    )
    names = property_names | local.names(
        set(table["subject"].drop_nulls()) | set(table["country"].drop_nulls())
    )

    print(
        f"\n{table.height:,} properties: {by_subject['id'].n_unique():,} about a news or "
        f"media outlet by their item, {len(by_name):,} by their name"
    )
    show(
        table.select(
            "id",
            named("id", names).alias("name"),
            "items",
            named("subject", names).alias("outlet"),
            "kind",
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
