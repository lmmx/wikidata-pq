"""The concepts a WikiProject looks after, and how they fit together, from a local copy of
the wikidata-pq datasets: WikiProject Mathematics (Q8487137) by default.

Items are tied to a WikiProject by "on focus list of Wikimedia project" (P5008) or
"maintained by WikiProject" (P6104), sometimes with a "WikiProject importance scale
rating" (P10714) qualifier. This collects those links, then reads the project's own items:

1. One pass over the claims, filtering on `property`, collects every WikiProject link:
   which projects are largest, and which share the most items with this one. It is the
   only step that reads every row, since the claims are sorted by subject, not value.
2. One lookup of the project's items reads their "instance of" (P31), "subclass of"
   (P279) and "part of" (P361) statements: what kinds of thing they are, and which of
   them the others are subclasses, parts or instances of, within the project. Items
   with a "numeric value" (P1181), such as the numbers WikiProject Mathematics has by
   the hundred thousand (even, odd, prime, ...), are left out.
3. Names and descriptions come from the labels and descriptions, in the language, then
   `mul`, then `en`, for only the ids shown.

    python demos/wikiprojects.py
    python demos/wikiprojects.py Q8487137 --lang de --top 25
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

FOCUS, MAINTAINED = "P5008", "P6104"
IMPORTANCE = "P10714"
INSTANCE, SUBCLASS, PART = "P31", "P279", "P361"
NUMERIC = "P1181"

dv = pl.col("datavalue").struct


def qualifier_id(prop: str) -> pl.Expr:
    """The item id of a statement's first `prop` qualifier."""
    return (
        pl.col("qualifiers")
        .list.eval(
            pl.element()
            .filter(pl.element().struct.field("key") == prop)
            .struct.field("value")
            .list.first()
            .struct.field("datavalue")
            .struct.field("id")
        )
        .list.first()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("project", nargs="?", default="Q8487137", help="WikiProject id")
    parser.add_argument("--lang", default="en", help="Language code (default en)")
    parser.add_argument("--top", type=int, default=15, help="Rows per table")
    parser.add_argument("--data", type=Path, default=Path("hub"), help="Local copy")
    args = parser.parse_args()
    data: Path = args.data
    project, top = args.project, args.top

    claims = pl.scan_parquet(data / "claims" / "all" / "*.parquet").filter(
        pl.col("rank") != "deprecated"
    )
    langs = [
        k
        for k in dict.fromkeys([args.lang, "mul", "en"])
        if (data / "labels" / k).is_dir()
    ]

    def texts(table: str, ids: set[str]) -> dict[str, str]:
        """Each id's value in `table`, from the first of `langs` that has one."""
        keys = [k for k in langs if (data / table / k).is_dir()]
        rank = pl.col("language").replace_strict(keys, range(len(keys)), default=None)
        return dict(
            pl.concat([pl.scan_parquet(data / table / k / "*.parquet") for k in keys])
            .filter(pl.col("id").is_in(sorted(ids)))
            .sort(rank, maintain_order=True)
            .unique("id", keep="first")
            .select("id", "value")
            .collect(engine="streaming")
            .iter_rows()
        )

    # 1. Every WikiProject link
    print("Collecting every WikiProject link from the claims (one pass)...")
    links = (
        claims.filter(pl.col("property").is_in([FOCUS, MAINTAINED]))
        .select(
            "id",
            "property",
            dv.field("id").alias("project"),
            qualifier_id(IMPORTANCE).alias("importance"),
        )
        .drop_nulls("project")
        .collect(engine="streaming")
    )
    projects = (
        links.group_by("project")
        .agg(
            pl.col("id").n_unique().alias("items"),
            (pl.col("property") == FOCUS).sum().alias("focus list"),
            (pl.col("property") == MAINTAINED).sum().alias("maintained"),
        )
        .sort("items", "project", descending=[True, False])
    )
    members = links.filter(pl.col("project") == project)
    items = set(members["id"])
    if not items:
        raise SystemExit(
            f"No items linked to {project}; the largest projects:\n{projects.head(top)}"
        )

    # 2. The project's items: their kinds, and the links among them, numbers left out
    relations = (
        claims.filter(
            pl.col("id").is_in(sorted(items)),
            pl.col("property").is_in([INSTANCE, SUBCLASS, PART, NUMERIC]),
        )
        .select("id", "property", dv.field("id").alias("value"))
        .unique()
        .collect(engine="streaming")
    )
    numbers = set(relations.filter(pl.col("property") == NUMERIC)["id"])
    items -= numbers
    members = members.filter(pl.col("id").is_in(list(items)))
    relations = relations.filter(
        pl.col("id").is_in(list(items)), pl.col("property") != NUMERIC
    ).drop_nulls()
    others = (
        links.filter(pl.col("id").is_in(list(items)), pl.col("project") != project)
        .group_by("project")
        .agg(pl.col("id").n_unique().alias("shared"))
        .sort("shared", "project", descending=[True, False])
        .head(top)
    )
    importance = (
        members.drop_nulls("importance")
        .group_by("importance")
        .len()
        .sort("len", descending=True)
    )
    kinds = (
        relations.filter(pl.col("property") == INSTANCE)
        .group_by("value")
        .agg(pl.col("id").n_unique().alias("items"))
        .sort("items", "value", descending=[True, False])
        .head(top)
    )
    inner = relations.filter(pl.col("value").is_in(list(items)))
    central = (
        inner.group_by("value")
        .agg(
            pl.len().alias("links"),
            (pl.col("property") == SUBCLASS).sum().alias("subclasses"),
            (pl.col("property") == PART).sum().alias("parts"),
            (pl.col("property") == INSTANCE).sum().alias("instances"),
        )
        .sort("links", "value", descending=[True, False])
        .head(top)
    )
    linked = set(inner["id"]) | set(inner["value"])

    # 3. Names and descriptions of what is shown
    shown = (
        {project}
        | set(projects["project"].head(top))
        | set(others["project"])
        | set(importance["importance"])
        | set(kinds["value"])
        | set(central["value"])
    )
    name = texts("labels", shown)
    about = texts("descriptions", set(central["value"]))

    def mapped(col: str, mapping: dict[str, str], default: pl.Expr) -> pl.Expr:
        if not mapping:
            return default
        return pl.col(col).replace_strict(mapping, default=default)

    def named(col: str) -> pl.Expr:
        return mapped(col, name, pl.col(col)).alias(col)

    config = pl.Config(
        tbl_rows=-1,
        fmt_str_lengths=60,
        thousands_separator=",",
        tbl_hide_dataframe_shape=True,
        tbl_hide_column_data_types=True,
    )
    with config:
        print(f"\n{projects.height:,} projects have items linked to them; the largest:")
        print(projects.head(top).select(named("project"), pl.exclude("project")))

        print(
            f"\n{name.get(project, project)} ({project}): {len(items):,} items, "
            f"leaving out {len(numbers):,} numbers (items with a numeric value)"
        )
        by = members.group_by("property").agg(pl.col("id").n_unique().alias("items"))
        for prop, n in by.sort("property").iter_rows():
            what = "on its focus list" if prop == FOCUS else "maintained by it"
            print(f"  {n:,} {what}")
        if importance.height:
            print("\nBy importance rating:")
            print(importance.select(named("importance"), pl.col("len").alias("items")))

        print("\nWhat kinds of thing they are (instance of):")
        print(kinds.select(named("value").alias("kind"), "items"))

        print(
            f"\n{len(linked & items):,} of the items are a subclass, part or instance of "
            f"another, or have one, within the project; the most central:"
        )
        print(
            central.select(
                named("value").alias("concept"),
                "links",
                "subclasses",
                "parts",
                "instances",
                mapped("value", about, pl.lit("")).alias("description"),
            )
        )

        print("\nThe projects that share the most of its items:")
        print(
            others.select(
                named("project"),
                "shared",
                (100 * pl.col("shared") / len(items)).round(1).alias("% of its items"),
            )
        )


if __name__ == "__main__":
    main()
