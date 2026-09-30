"""Algorithms in Wikidata, from a local copy of the wikidata-pq datasets.

The members of "algorithm" (Q8366): instances of it or of any class below it by
"subclass of". Then, from the members' own statements and sitelinks:

- the families of algorithm (the classes directly below "algorithm"), with how many
  algorithms are in each, counting the classes below them;
- the time and space complexities of every algorithm that has one (P3752-P3757), the
  data Wikidata holds for comparing sorting, search and graph algorithms;
- the earliest algorithms with a date ("time of discovery or invention", P575, or
  "inception", P571), and their inventors ("discoverer or inventor", P61);
- the people with the most algorithms named after them ("named after", P138);
- the algorithms with articles in the most Wikipedias.

See demos/classes.py for how the members are found: two passes over the claims, then
lookups of the members' ids.

    python demos/algorithms.py
    python demos/algorithms.py --lang de --top 25
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from classes import Local, joined, named, show

ROOT = "Q8366"
NAMED_AFTER, DISCOVERER = "P138", "P61"
DISCOVERED, INCEPTION = "P575", "P571"
COMPLEXITY = {
    "P3753": "best time",
    "P3754": "average time",
    "P3752": "worst time",
    "P3756": "best space",
    "P3757": "average space",
    "P3755": "worst space",
}


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
    ids = members["id"].unique().to_list()
    props = [NAMED_AFTER, DISCOVERER, DISCOVERED, INCEPTION, *COMPLEXITY]
    facts = local.statements(ids, props)
    wikipedias = local.wikipedias(ids)

    # Each class's family: the class directly below the root that it was reached from
    family = {ROOT: ROOT}
    for cls, parent, depth in classes.sort("depth").iter_rows():
        if depth == 1:
            family[cls] = cls
        elif depth > 1:
            family[cls] = family[parent]
    members = members.with_columns(
        family=pl.col("class").replace_strict(family, default=None)
    )
    families = (
        members.group_by("family")
        .agg(pl.col("id").n_unique().alias("algorithms"))
        .sort("algorithms", "family", descending=[True, False])
        .head(top)
    )

    measured = facts.filter(pl.col("property").is_in(list(COMPLEXITY)))
    first_family = members.sort("family").unique("id", keep="first").select("id", "family")
    dated = (
        facts.filter(pl.col("property").is_in([DISCOVERED, INCEPTION]))
        .group_by("id")
        .agg(pl.col("year").min())
        .drop_nulls()
    )
    earliest = dated.sort("year", "id").head(2 * top)
    named_after = (
        facts.filter(pl.col("property") == NAMED_AFTER)
        .group_by("value")
        .agg(pl.len().alias("algorithms"), pl.col("id").sort().head(3).alias("examples"))
        .sort("algorithms", "value", descending=[True, False])
        .head(top)
    )
    best_known = wikipedias.sort("wikipedias", "id", descending=[True, False]).head(top)

    names = local.names(
        set(families["family"])
        | set(measured["id"])
        | set(measured["value"])
        | set(first_family["family"])
        | set(earliest["id"])
        | set(facts.filter(pl.col("property") == DISCOVERER)["value"])
        | set(named_after["value"])
        | set(named_after["examples"].explode())
        | set(best_known["id"])
    )

    print(
        f"\n{len(ids):,} algorithms, instances of {classes.height:,} classes "
        "(algorithm and those below it)"
    )

    print("\nThe largest families (classes directly below algorithm):")
    show(families.select(named("family", names), "algorithms"))

    table = pl.DataFrame({"id": measured["id"].unique()})
    for prop, alias in COMPLEXITY.items():
        table = table.join(joined(facts, prop, alias, names), on="id", how="left")
    table = (
        table.join(first_family, on="id", how="left")
        .with_columns(named("family", names))
        .with_columns(named("id", names).alias("algorithm"))
        .sort("family", "algorithm", nulls_last=True)
        .select("family", "algorithm", *COMPLEXITY.values())
    )
    table = table.select(
        [c for c in table.columns if table[c].null_count() < table.height]
    )
    print(f"\nComplexities ({table.height:,} algorithms have one):")
    show(table)

    print(f"\nThe earliest ({dated.height:,} algorithms have a date):")
    show(
        earliest.join(joined(facts, DISCOVERER, "invented by", names), on="id", how="left")
        .sort("year", "id")
        .select("year", named("id", names).alias("algorithm"), "invented by")
    )

    print("\nNamed after the most algorithms:")
    show(
        named_after.select(
            named("value", names).alias("named after"),
            "algorithms",
            pl.col("examples")
            .list.eval(pl.element().replace_strict(names, default=pl.element()))
            .list.join(", ")
            .alias("for example"),
        )
    )

    print("\nIn the most Wikipedias:")
    show(best_known.select(named("id", names).alias("algorithm"), "wikipedias"))


if __name__ == "__main__":
    main()
