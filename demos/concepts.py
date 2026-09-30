"""Mathematical concepts in Wikidata, from a local copy of the wikidata-pq datasets.

The members of "mathematical concept" (Q24034552): instances of it or of any class below
it by "subclass of", less the numbers (items with a "numeric value", P1181). Then, from
the members' own statements and sitelinks:

- the largest kinds of concept;
- the concepts with articles in the most Wikipedias, and what they are;
- the people with the most concepts named after them ("named after", P138);
- the earliest concepts with a date of discovery ("time of discovery or invention", P575,
  or "inception", P571), and who found them ("discoverer or inventor", P61);
- the fields that study them ("studied by", P2579);
- the defining formulas (P2534) of the best-known ones, in LaTeX.

See demos/classes.py for how the members are found: two passes over the claims, then
lookups of the members' ids.

    python demos/concepts.py
    python demos/concepts.py --lang fr --top 25
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from classes import NUMERIC, Local, joined, named, show

ROOT = "Q24034552"
NAMED_AFTER, DISCOVERER = "P138", "P61"
DISCOVERED, INCEPTION = "P575", "P571"
STUDIED_BY, FORMULA = "P2579", "P2534"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--lang", default="en", help="Language code (default en)")
    parser.add_argument("--top", type=int, default=15, help="Rows per table")
    parser.add_argument("--data", type=Path, default=Path("hub"), help="Local copy")
    args = parser.parse_args()
    top = args.top
    local = Local(args.data, args.lang)

    classes = local.subclasses(ROOT)
    members = local.members(classes["id"].to_list())
    props = [NAMED_AFTER, DISCOVERER, DISCOVERED, INCEPTION, STUDIED_BY, FORMULA, NUMERIC]
    facts = local.statements(members["id"].unique().to_list(), props)
    numbers = facts.filter(pl.col("property") == NUMERIC)["id"].unique()
    members = members.filter(~pl.col("id").is_in(numbers.implode()))
    facts = facts.filter(~pl.col("id").is_in(numbers.implode()))
    ids = members["id"].unique().to_list()
    wikipedias = local.wikipedias(ids)

    kinds = (
        members.group_by("class")
        .agg(pl.len().alias("concepts"))
        .sort("concepts", "class", descending=[True, False])
        .head(top)
    )
    best_known = wikipedias.sort("wikipedias", "id", descending=[True, False]).head(top)
    named_after = (
        facts.filter(pl.col("property") == NAMED_AFTER)
        .join(wikipedias, on="id", how="left")
        .sort("wikipedias", descending=True, nulls_last=True)
        .group_by("value", maintain_order=True)
        .agg(pl.len().alias("concepts"), pl.col("id").head(3).alias("examples"))
        .sort("concepts", "value", descending=[True, False])
        .head(top)
    )
    dated = (
        facts.filter(pl.col("property").is_in([DISCOVERED, INCEPTION]))
        .group_by("id")
        .agg(pl.col("year").min())
        .drop_nulls()
    )
    earliest = dated.sort("year", "id").head(top)
    fields = (
        facts.filter(pl.col("property") == STUDIED_BY)
        .group_by("value")
        .agg(pl.len().alias("concepts"))
        .sort("concepts", "value", descending=[True, False])
        .head(top)
    )
    formulas = (
        facts.filter(pl.col("property") == FORMULA)
        .join(wikipedias, on="id", how="left")
        .sort("wikipedias", "id", descending=[True, False], nulls_last=True)
        .unique("id", keep="first", maintain_order=True)
        .head(top)
    )

    people = facts.filter(pl.col("property").is_in([NAMED_AFTER, DISCOVERER]))
    names = local.names(
        set(kinds["class"])
        | set(best_known["id"])
        | set(people["value"])
        | set(named_after["examples"].explode())
        | set(earliest["id"])
        | set(fields["value"])
        | set(formulas["id"])
    )
    about = local.descriptions(set(best_known["id"]))

    print(
        f"\n{len(ids):,} mathematical concepts, instances of {classes.height:,} classes "
        f"(mathematical concept and those below it); {numbers.len():,} numbers left out"
    )

    print("\nThe largest kinds:")
    show(kinds.select(named("class", names).alias("kind"), "concepts"))

    print("\nIn the most Wikipedias:")
    show(
        best_known.select(
            named("id", names).alias("concept"),
            "wikipedias",
            pl.col("id").replace_strict(about, default="").alias("description")
            if about
            else pl.lit("").alias("description"),
        )
    )

    print("\nNamed after the most concepts:")
    show(
        named_after.select(
            named("value", names).alias("named after"),
            "concepts",
            pl.col("examples")
            .list.eval(pl.element().replace_strict(names, default=pl.element()))
            .list.join(", ")
            .alias("best known")
            if names
            else pl.col("examples").list.join(", ").alias("best known"),
        )
    )

    print(f"\nThe earliest discovered ({dated.height:,} concepts have a date):")
    show(
        earliest.join(
            joined(facts, DISCOVERER, "discovered by", names), on="id", how="left"
        )
        .sort("year", "id")
        .select("year", named("id", names).alias("concept"), "discovered by")
    )

    print("\nThe fields that study the most of them:")
    show(fields.select(named("value", names).alias("studied by"), "concepts"))

    print("\nThe defining formulas of the best known:")
    show(formulas.select(named("id", names).alias("concept"), pl.col("text").alias("formula")))


if __name__ == "__main__":
    main()
