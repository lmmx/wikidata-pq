"""Helpers for demos about the members of a class (demos/concepts.py, demos/algorithms.py),
from a local copy of the wikidata-pq datasets.

Which items are members of a class is a question about the values of statements, and the
claims are sorted by subject, so it takes two passes over the claims, filtering on
`property`: one for every "subclass of" (P279) link, to find the classes below the root,
and one for the "instance of" (P31) links into those classes. Everything after that is a
lookup of the members' own ids, which reads only the row groups that can hold them.
"""

from __future__ import annotations

import re
from pathlib import Path

import polars as pl

INSTANCE, SUBCLASS = "P31", "P279"
NUMERIC = "P1181"
# Sites named `{code}wiki` that are not a Wikipedia
NOT_WIKIPEDIA = {
    "commonswiki",
    "foundationwiki",
    "incubatorwiki",
    "mediawikiwiki",
    "metawiki",
    "outreachwiki",
    "sourceswiki",
    "specieswiki",
    "wikidatawiki",
    "wikifunctionswiki",
    "wikimaniawiki",
}

dv = pl.col("datavalue").struct


class Local:
    """A local copy of the datasets, read in one language (then `mul`, then `en`)."""

    def __init__(self, data: Path, lang: str) -> None:
        self.data = data
        self.claims = pl.scan_parquet(data / "claims" / "all" / "*.parquet").filter(
            pl.col("rank") != "deprecated"
        )
        self.langs = [
            k
            for k in dict.fromkeys([lang, "mul", "en"])
            if (data / "labels" / k).is_dir()
        ]

    def subclasses(self, root: str) -> pl.DataFrame:
        """The root and every class below it by "subclass of": `id`, `parent` (the
        class it was reached from) and `depth`. One pass over the claims."""
        print(f"Finding the classes below {root} (one pass over the claims)...")
        edges = (
            self.claims.filter(pl.col("property") == SUBCLASS)
            .select("id", dv.field("id").alias("parent"))
            .drop_nulls()
            .unique()
            .collect(engine="streaming")
        )
        found = pl.DataFrame(
            {"id": [root], "parent": [None], "depth": [0]},
            schema={"id": pl.String, "parent": pl.String, "depth": pl.Int32},
        )
        frontier, depth = [root], 0
        while frontier:
            depth += 1
            below = (
                edges.filter(
                    pl.col("parent").is_in(frontier),
                    ~pl.col("id").is_in(found["id"].implode()),
                )
                .unique("id", keep="first", maintain_order=True)
                .with_columns(depth=pl.lit(depth, pl.Int32))
            )
            found = pl.concat([found, below])
            frontier = below["id"].to_list()
        return found

    def members(self, classes: list[str]) -> pl.DataFrame:
        """Every instance of the classes: `id` and `class`. One pass over the claims."""
        print(f"Finding the instances of {len(classes):,} classes (one pass)...")
        return (
            self.claims.filter(
                pl.col("property") == INSTANCE, dv.field("id").is_in(classes)
            )
            .select("id", dv.field("id").alias("class"))
            .unique()
            .collect(engine="streaming")
        )

    def statements(self, ids: list[str], props: list[str]) -> pl.DataFrame:
        """The ids' `props` statements: `id`, `property`, `value` (an item), `text` (a
        string or formula) and `year`. A lookup of the ids."""
        return (
            self.claims.filter(
                pl.col("id").is_in(sorted(ids)), pl.col("property").is_in(props)
            )
            .select(
                "id",
                "property",
                dv.field("id").alias("value"),
                dv.field("datavalue__string").alias("text"),
                dv.field("time")
                .str.extract(r"^([+-]?\d+)-")
                .cast(pl.Int64)
                .alias("year"),
            )
            .unique()
            .collect(engine="streaming")
        )

    def _texts(self, table: str, ids: set[str]) -> dict[str, str]:
        keys = [k for k in self.langs if (self.data / table / k).is_dir()]
        rank = pl.col("language").replace_strict(keys, range(len(keys)), default=None)
        return dict(
            pl.concat(
                [pl.scan_parquet(self.data / table / k / "*.parquet") for k in keys]
            )
            .filter(pl.col("id").is_in(sorted(ids)))
            .sort(rank, maintain_order=True)
            .unique("id", keep="first")
            .select("id", "value")
            .collect(engine="streaming")
            .iter_rows()
        )

    def names(self, ids: set[str]) -> dict[str, str]:
        """Each id's label, from the first language that has one."""
        return self._texts("labels", {i for i in ids if i})

    def descriptions(self, ids: set[str]) -> dict[str, str]:
        """Each id's description, from the first language that has one."""
        return self._texts("descriptions", {i for i in ids if i})

    def wikipedias(self, ids: list[str]) -> pl.DataFrame:
        """How many Wikipedias have an article on each id: `id`, `wikipedias`. A lookup
        of the ids in each Wikipedia's folder of the links."""
        sites = [
            d
            for d in sorted((self.data / "links").iterdir())
            if re.fullmatch(r"[a-z_]+wiki", d.name) and d.name not in NOT_WIKIPEDIA
        ]
        return (
            pl.concat([pl.scan_parquet(d / "*.parquet").select("id") for d in sites])
            .filter(pl.col("id").is_in(sorted(ids)))
            .group_by("id")
            .agg(pl.len().alias("wikipedias"))
            .collect(engine="streaming")
        )


def named(col: str, names: dict[str, str]) -> pl.Expr:
    """The column's ids replaced by their names (the id where there is none)."""
    if not names:
        return pl.col(col)
    return pl.col(col).replace_strict(names, default=pl.col(col)).alias(col)


def joined(facts: pl.DataFrame, prop: str, alias: str, names: dict[str, str]) -> pl.DataFrame:
    """Each id's `prop` values, named and joined with ", ": `id`, `alias`."""
    return (
        facts.filter(pl.col("property") == prop)
        .select("id", named("value", names))
        .group_by("id")
        .agg(pl.col("value").unique().sort().str.join(", ").alias(alias))
    )


def show(df: pl.DataFrame) -> None:
    with pl.Config(
        tbl_rows=-1,
        tbl_cols=-1,
        tbl_width_chars=200,
        fmt_str_lengths=70,
        thousands_separator=",",
        tbl_hide_dataframe_shape=True,
        tbl_hide_column_data_types=True,
    ):
        print(df)
