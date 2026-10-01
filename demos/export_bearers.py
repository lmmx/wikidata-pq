"""The items bearing any of some properties as a Parquet file, one row per item, to explore
in an embedding viewer, from a local copy of the wikidata-pq datasets.

Each row has the item's name and description (and `text`, the two together, to embed),
its kinds ("instance of", and `kind`, the one of them most common among the items, to
colour by), its country, how many Wikipedias have an article on it, and
which of the properties it holds, by the outlet or site each is for ("Wikidata item of
this property", P1629, else the property's own name). For news website topic IDs (see
demos/news_topics_export.sh) that is which news sites have a topic page on it.

One pass over the claims, a file at a time (the files hold consecutive id ranges, so a
file's bearers have their statements in it), then each Wikipedia's folder of the links
(`id` column only), and the labels and descriptions of the bearers.

    python demos/export_bearers.py P3106 P6200 --out demos/output/topics.parquet
    embedding-atlas demos/output/topics.parquet --text text
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl
from tqdm import tqdm

from classes import INSTANCE, Local

SUBJECT = "P1629"
# Where an item is from, in order of preference: country, citizenship, origin
COUNTRY = ["P17", "P27", "P495"]
# Property ids sort between lexemes (`L...`) and items (`Q...`)
IS_PROPERTY = (pl.col("id") >= "P") & (pl.col("id") < "Q")
SEP = "; "

dv = pl.col("datavalue").struct


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("properties", nargs="+", help="Property ids, e.g. P3106")
    parser.add_argument("--out", type=Path, required=True, help="Parquet file to write")
    parser.add_argument("--lang", default="en", help="Language code (default en)")
    parser.add_argument("--data", type=Path, default=Path("hub"), help="Local copy")
    args = parser.parse_args()
    data: Path = args.data
    props = args.properties
    local = Local(data, args.lang)

    # Each property's outlet: the property range of the claims only
    outlet_of = (
        local.claims.filter(
            IS_PROPERTY, pl.col("id").is_in(props), pl.col("property") == SUBJECT
        )
        .select("id", dv.field("id").alias("outlet"))
        .drop_nulls()
        .unique("id", keep="first")
        .collect(engine="streaming")
    )

    # One pass over the claims: the bearers, their kinds and their countries
    files = sorted((data / "claims" / "all").glob("*.parquet"))
    held, facts = [], []
    for f in tqdm(files, desc="claims", unit="file"):
        lf = pl.scan_parquet(f).filter(pl.col("rank") != "deprecated")
        bearers = lf.filter(pl.col("property").is_in(props)).select("id", "property")
        found, about = pl.collect_all(
            [
                bearers.unique(),
                lf.filter(pl.col("property").is_in([INSTANCE, *COUNTRY]))
                .select("id", "property", dv.field("id").alias("value"))
                .drop_nulls()
                .join(bearers.select("id").unique(), on="id", how="semi")
                .unique(),
            ],
            engine="streaming",
        )
        held.append(found)
        facts.append(about)
    held, facts = pl.concat(held), pl.concat(facts)
    ids = held["id"].unique().sort()
    if ids.is_empty():
        raise SystemExit(f"No items have {' or '.join(props)}")
    print(f"{ids.len():,} items; counting their Wikipedias and naming them...")

    wikipedias = local.wikipedias(ids.to_list())
    kind_ids = set(facts.filter(pl.col("property") == INSTANCE)["value"])
    country_ids = set(facts.filter(pl.col("property").is_in(COUNTRY))["value"])
    names = local.names(
        set(ids) | kind_ids | country_ids | set(props) | set(outlet_of["outlet"])
    )
    about = local.descriptions(set(ids))

    def name(col: str) -> pl.Expr:
        return pl.col(col).replace_strict(names, default=pl.col(col))

    sites = (
        pl.DataFrame({"property": props})
        .join(outlet_of.rename({"id": "property"}), on="property", how="left")
        .with_columns(site=pl.coalesce(name("outlet"), name("property")))
        .select("property", "site")
    )
    outlets = (
        held.join(sites, on="property", how="left")
        .group_by("id")
        .agg(
            pl.col("site").unique().sort().str.join(SEP).alias("outlets"),
            pl.col("property").n_unique().alias("n_outlets"),
        )
    )
    instances = facts.filter(pl.col("property") == INSTANCE)
    kinds = (
        instances.with_columns(name("value").alias("kind"))
        .group_by("id")
        .agg(pl.col("kind").unique().sort().str.join(SEP).alias("kinds"))
    )
    # An item's main kind, to colour by: of its kinds, the most common among the items
    common = instances.group_by("value").agg(pl.len().alias("n"))
    main_kind = (
        instances.join(common, on="value")
        .sort("n", "value", descending=[True, False])
        .unique("id", keep="first")
        .select("id", name("value").alias("kind"))
    )
    rank = pl.col("property").replace_strict(COUNTRY, range(len(COUNTRY)))
    country = (
        facts.filter(pl.col("property").is_in(COUNTRY))
        .sort(rank, "value")
        .unique("id", keep="first")
        .select("id", name("value").alias("country"))
    )

    table = (
        pl.DataFrame({"id": ids})
        .with_columns(
            name("id").alias("name"),
            pl.col("id").replace_strict(about, default=None).alias("description")
            if about
            else pl.lit(None, pl.String).alias("description"),
        )
        .join(kinds, on="id", how="left")
        .join(main_kind, on="id", how="left")
        .join(country, on="id", how="left")
        .join(outlets, on="id", how="left")
        .join(wikipedias, on="id", how="left")
        .with_columns(
            pl.col("wikipedias").fill_null(0),
            text=pl.concat_str(
                "name",
                pl.when(pl.col("description").is_not_null())
                .then(pl.concat_str(pl.lit(": "), "description"))
                .when(pl.col("kinds").is_not_null())
                .then(pl.concat_str(pl.lit(" ("), "kinds", pl.lit(")")))
                .otherwise(pl.lit("")),
            ),
        )
        .select(
            "id",
            "name",
            "description",
            "text",
            "kind",
            "kinds",
            "country",
            "outlets",
            "n_outlets",
            "wikipedias",
        )
        .sort("n_outlets", "wikipedias", "id", descending=[True, True, False])
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.write_parquet(args.out)
    print(f"Wrote {table.height:,} rows to {args.out}")
    with pl.Config(tbl_cols=-1, tbl_width_chars=200, fmt_str_lengths=50):
        print(table.head(5))


if __name__ == "__main__":
    main()
