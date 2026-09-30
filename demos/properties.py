"""The most common properties in Wikidata, overall and among the entities with articles in
the most Wikipedias, from a local copy of the wikidata-pq datasets.

1. Every property's cardinality: how many entities have it, and how many statements
   there are. A pass over the claims' `id` and `property` columns, a file at a time:
   the files hold consecutive id ranges, so entities counted per file add up. (An id
   whose statements straddle two files would count twice: at most once per property per
   file boundary, 33 in all.) Deprecated statements are left out, here and below.
2. The most Wikipedia-popular items: the `--popular` items with articles in the most
   Wikipedias, counted over every Wikipedia's folder of the links (`id` column only).
3. Their properties: a lookup of their ids in the claims.

Then: the properties most of the popular items have, next to how common they are
overall, and the properties most over-represented among them ("lift": their share of
popular items over their share of all entities).

    python demos/properties.py
    python demos/properties.py --popular 1000 --top 40 --lang de
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import polars as pl
from tqdm import tqdm

from classes import NOT_WIKIPEDIA, Local, named, show

# Among the popular items, over-represented properties are listed from this share up
MIN_SHARE = 0.25


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--popular", type=int, default=10_000, help="Popular items")
    parser.add_argument("--top", type=int, default=30, help="Rows per table")
    parser.add_argument("--lang", default="en", help="Language code (default en)")
    parser.add_argument("--data", type=Path, default=Path("hub"), help="Local copy")
    args = parser.parse_args()
    data: Path = args.data
    top = args.top
    local = Local(data, args.lang)

    # 1. Every property's cardinality, a file at a time
    files = sorted((data / "claims" / "all").glob("*.parquet"))
    parts, entities = [], 0
    for f in tqdm(files, desc="claims", unit="file"):
        lf = (
            pl.scan_parquet(f)
            .filter(pl.col("rank") != "deprecated")
            .select("id", "property")
        )
        parts.append(
            lf.group_by("property")
            .agg(pl.len().alias("statements"), pl.col("id").n_unique().alias("entities"))
            .collect(engine="streaming")
        )
        entities += lf.select(pl.col("id").n_unique()).collect(engine="streaming").item()
    overall = (
        pl.concat(parts)
        .group_by("property")
        .agg(pl.col("statements", "entities").sum())
        .with_columns(share=pl.col("entities") / entities)
    )

    # 2. The items with articles in the most Wikipedias
    sites = [
        d
        for d in sorted((data / "links").iterdir())
        if re.fullmatch(r"[a-z_]+wiki", d.name) and d.name not in NOT_WIKIPEDIA
    ]
    print(f"Counting articles in {len(sites)} Wikipedias...")
    popular = (
        pl.concat([pl.scan_parquet(d / "*.parquet").select("id") for d in sites])
        .filter(pl.col("id").str.starts_with("Q"))
        .select(pl.col("id").str.slice(1).cast(pl.UInt32))
        .group_by("id")
        .agg(pl.len().alias("wikipedias"))
        .top_k(args.popular, by="wikipedias")
        .collect(engine="streaming")
        .select(pl.format("Q{}", "id").alias("id"), "wikipedias")
        .sort("wikipedias", "id", descending=[True, False])
    )
    n = popular.height

    # 3. Their properties
    among = (
        local.claims.filter(pl.col("id").is_in(popular["id"].implode()))
        .select("id", "property")
        .group_by("property")
        .agg(pl.col("id").n_unique().alias("popular"))
        .collect(engine="streaming")
        .with_columns(popular_share=pl.col("popular") / n)
        .join(overall, on="property", how="left")
        .with_columns(lift=pl.col("popular_share") / pl.col("share"))
    )

    by_entities = overall.sort("entities", "property", descending=[True, False]).head(top)
    by_popular = among.sort("popular", "property", descending=[True, False]).head(top)
    by_lift = (
        among.filter(pl.col("popular_share") >= MIN_SHARE)
        .sort("lift", "property", descending=[True, False])
        .head(top)
    )
    names = local.names(
        set(by_entities["property"])
        | set(by_popular["property"])
        | set(by_lift["property"])
        | set(popular["id"].head(top))
    )

    print(
        f"\n{entities:,} entities have statements, {overall['statements'].sum():,} in "
        f"all, with {overall.height:,} properties. The most common:"
    )
    show(
        by_entities.select(
            "property",
            named("property", names).alias("name"),
            "entities",
            (100 * pl.col("share")).round(1).alias("% of entities"),
            "statements",
            (pl.col("statements") / pl.col("entities")).round(1).alias("per entity"),
        )
    )

    print(f"\nThe {n:,} items in the most Wikipedias; the first {top}:")
    show(
        popular.head(top).select(
            "id", named("id", names).alias("name"), "wikipedias"
        )
    )
    low = popular["wikipedias"].min()
    print(f"(the {n:,}th is in {low} Wikipedias)")

    print("\nThe properties most of them have:")
    show(
        by_popular.select(
            "property",
            named("property", names).alias("name"),
            (100 * pl.col("popular_share")).round(1).alias("% of popular"),
            (100 * pl.col("share")).round(1).alias("% of all"),
            pl.col("lift").round(1),
        )
    )

    print(
        f"\nThe most over-represented among them (held by at least "
        f"{MIN_SHARE:.0%} of them):"
    )
    show(
        by_lift.select(
            "property",
            named("property", names).alias("name"),
            (100 * pl.col("popular_share")).round(1).alias("% of popular"),
            (100 * pl.col("share")).round(3).alias("% of all"),
            pl.col("lift").round(1),
        )
    )


if __name__ == "__main__":
    main()
