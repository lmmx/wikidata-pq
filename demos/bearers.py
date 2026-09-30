"""The items bearing a property, and what kinds of thing they are, from a local copy of the
wikidata-pq datasets: e.g. the items with a BBC Things ID (P1617).

One pass over the claims, a file at a time (the files hold consecutive id ranges, so
each file's bearers have all their statements in it, but for an id straddling a file
boundary):

- the bearers, and their values of the property;
- what they are ("instance of", P31);
- which properties they have, and which all entities have, to find the properties most
  over-represented among them ("lift": share of bearers over share of all entities).

Then their articles in each Wikipedia's folder of the links (`id` column only), for the
best known of them, and a random sample, to see the long tail. `--pattern` groups the
values by a regex group, where an id's format says something: a Google Knowledge Graph
ID starts `/m/` (from Freebase) or `/g/`, a WordNet synset ID ends in its part of speech.

    python demos/bearers.py P1617
    python demos/bearers.py P2671 --pattern '^/(\\w+)/'
    python demos/bearers.py P8814 --pattern='-(\\w)$' --lang fr
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import polars as pl
from tqdm import tqdm

from classes import INSTANCE, NOT_WIKIPEDIA, Local, named, show

# Over-represented properties are listed from this share of bearers up
MIN_SHARE = 0.05

dv = pl.col("datavalue").struct


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("property", help="Property id, e.g. P1617")
    parser.add_argument("--pattern", help="Regex with one group, to group the values by")
    parser.add_argument("--top", type=int, default=20, help="Rows per table")
    parser.add_argument("--sample", type=int, default=20, help="Random bearers shown")
    parser.add_argument("--seed", type=int, default=0, help="For the random sample")
    parser.add_argument("--lang", default="en", help="Language code (default en)")
    parser.add_argument("--data", type=Path, default=Path("hub"), help="Local copy")
    args = parser.parse_args()
    data: Path = args.data
    prop, top = args.property, args.top
    local = Local(data, args.lang)

    # One pass over the claims, a file at a time
    files = sorted((data / "claims" / "all").glob("*.parquet"))
    held, kinds, among, overall = [], [], [], []
    entities = 0
    for f in tqdm(files, desc="claims", unit="file"):
        lf = pl.scan_parquet(f).filter(pl.col("rank") != "deprecated")
        pairs = lf.select("id", "property")
        bearers = lf.filter(pl.col("property") == prop).select(
            "id", dv.field("datavalue__string").alias("value")
        )
        ids = bearers.select("id").unique()
        results = pl.collect_all(
            [
                bearers,
                lf.filter(pl.col("property") == INSTANCE)
                .select("id", dv.field("id").alias("kind"))
                .join(ids, on="id", how="semi")
                .group_by("kind")
                .agg(pl.col("id").n_unique().alias("items")),
                pairs.join(ids, on="id", how="semi")
                .group_by("property")
                .agg(pl.col("id").n_unique().alias("bearers")),
                pairs.group_by("property").agg(pl.col("id").n_unique().alias("entities")),
                pairs.select(pl.col("id").n_unique().alias("n")),
            ],
            engine="streaming",
        )
        for store, df in zip([held, kinds, among, overall], results):
            store.append(df)
        entities += results[4].item()

    held = pl.concat(held)
    n = held["id"].n_unique()
    if not n:
        raise SystemExit(f"No items have {prop}")
    kinds = (
        pl.concat(kinds)
        .group_by("kind")
        .agg(pl.col("items").sum())
        .sort("items", "kind", descending=[True, False])
        .head(top)
    )
    overall = pl.concat(overall).group_by("property").agg(pl.col("entities").sum())
    among = (
        pl.concat(among)
        .group_by("property")
        .agg(pl.col("bearers").sum())
        .join(overall, on="property")
        .with_columns(
            share=pl.col("bearers") / n,
            overall_share=pl.col("entities") / entities,
        )
        .with_columns(lift=pl.col("share") / pl.col("overall_share"))
        .filter(pl.col("property") != prop, pl.col("share") >= MIN_SHARE)
        .sort("lift", "property", descending=[True, False])
        .head(top)
    )

    # Their articles in each Wikipedia
    sites = [
        d
        for d in sorted((data / "links").iterdir())
        if re.fullmatch(r"[a-z_]+wiki", d.name) and d.name not in NOT_WIKIPEDIA
    ]
    print(f"Counting their articles in {len(sites)} Wikipedias...")
    wikipedias = (
        pl.concat([pl.scan_parquet(d / "*.parquet").select("id") for d in sites])
        .join(held.lazy().select("id").unique(), on="id", how="semi")
        .group_by("id")
        .agg(pl.len().alias("wikipedias"))
        .collect(engine="streaming")
    )
    best_known = wikipedias.sort("wikipedias", "id", descending=[True, False]).head(top)
    sample = (
        held.select("id")
        .unique()
        .sort("id")
        .sample(min(args.sample, n), seed=args.seed)
        .join(wikipedias, on="id", how="left")
        .with_columns(pl.col("wikipedias").fill_null(0))
        .sort("id")
    )

    shown = (
        {prop}
        | set(kinds["kind"])
        | set(among["property"])
        | set(best_known["id"])
        | set(sample["id"])
    )
    names = local.names(shown)
    about = local.descriptions(set(best_known["id"]) | set(sample["id"]))

    def described(col: str) -> pl.Expr:
        if not about:
            return pl.lit("").alias("description")
        return pl.col(col).replace_strict(about, default="").alias("description")

    print(
        f"\n{n:,} items have {names.get(prop, prop)} ({prop}), "
        f"{100 * n / entities:.2f}% of the {entities:,} entities with statements; "
        f"{wikipedias.height:,} of them ({100 * wikipedias.height / n:.1f}%) have a "
        f"Wikipedia article, in {wikipedias['wikipedias'].median() or 0:.0f} "
        "Wikipedias at the median"
    )
    extra = held.height - n
    if extra:
        print(f"({extra:,} more values: some items have several)")

    if args.pattern:
        print(f"\nTheir values, by {args.pattern!r}:")
        show(
            held.select(pl.col("value").str.extract(args.pattern, 1).alias("group"))
            .group_by("group")
            .agg(pl.len().alias("values"))
            .with_columns((100 * pl.col("values") / held.height).round(1).alias("%"))
            .sort("values", "group", descending=[True, False], nulls_last=True)
            .head(top)
        )

    print("\nWhat they are (instance of):")
    show(
        kinds.select(
            named("kind", names),
            "items",
            (100 * pl.col("items") / n).round(1).alias("% of them"),
        )
    )

    print("\nIn the most Wikipedias:")
    show(
        best_known.select(
            "id", named("id", names).alias("name"), "wikipedias", described("id")
        )
    )

    print(f"\nA random {sample.height} of them:")
    show(
        sample.select(
            "id", named("id", names).alias("name"), "wikipedias", described("id")
        )
    )

    print(
        f"\nThe properties most over-represented among them (held by at least "
        f"{MIN_SHARE:.0%}):"
    )
    show(
        among.select(
            "property",
            named("property", names).alias("name"),
            (100 * pl.col("share")).round(1).alias("% of them"),
            (100 * pl.col("overall_share")).round(3).alias("% of all"),
            pl.col("lift").round(1),
        )
    )


if __name__ == "__main__":
    main()
