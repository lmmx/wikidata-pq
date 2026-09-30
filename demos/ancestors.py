"""The ancestors of a Wikidata item, from a local copy of the wikidata-pq datasets.

Walks "father" (P22) and "mother" (P25) statements up from an item, a generation at a
time, then names every ancestor and gives their years of birth and death. Each generation
is one lookup of a batch of ids in wikidata-claims: its rows are sorted by id, so the
lookup reads only the row groups whose id range can hold one of them, not the 17 GB.

    python demos/ancestors.py                     # Elizabeth II (Q9682)
    python demos/ancestors.py Q517 --lang fr      # Napoleon, named in French
    python demos/ancestors.py Q9682 --generations 25 --csv tree.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl
from tqdm import tqdm

PARENTS = {"P22": "father", "P25": "mother"}
LIFE = {"P569": "born", "P570": "died"}
# Ids looked up per query: a batch touches up to one row group per id
BATCH = 50


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "item", nargs="?", default="Q9682", help="Item id (default Q9682)"
    )
    parser.add_argument("--lang", default="en", help="Language code (default en)")
    parser.add_argument("--generations", type=int, default=25, help="Most to walk")
    parser.add_argument("--data", type=Path, default=Path("hub"), help="Local copy")
    parser.add_argument("--csv", type=Path, help="Write every ancestor to this file")
    args = parser.parse_args()
    data: Path = args.data

    langs = [
        k
        for k in dict.fromkeys([args.lang, "mul", "en"])
        if (data / "labels" / k).is_dir()
    ]
    claims = pl.scan_parquet(data / "claims" / "all" / "*.parquet").select(
        "id",
        "property",
        pl.col("datavalue").struct.field("id").alias("parent"),
        pl.col("datavalue")
        .struct.field("time")
        .str.extract(r"^\+?(-?\d+)-")
        .cast(pl.Int32, strict=False)
        .alias("year"),
        "rank",
    )

    def lookup(ids: list[str]) -> pl.DataFrame:
        """The parent and life-date statements of the ids, in batches."""
        frames = []
        for i in range(0, len(ids), BATCH):
            frames.append(
                claims.filter(
                    pl.col("id").is_in(ids[i : i + BATCH]),
                    pl.col("property").is_in([*PARENTS, *LIFE]),
                    pl.col("rank") != "deprecated",
                ).collect(engine="streaming")
            )
        return pl.concat(frames)

    # Breadth-first up the tree: each id is visited once, at its nearest generation
    generation = {args.item: 0}
    frontier = [args.item]
    rows = []
    with tqdm(desc="generations", unit="gen") as bar:
        while frontier:
            found = lookup(sorted(frontier))
            rows.append(found)
            depth = generation[frontier[0]] + 1
            parents = found.filter(pl.col("property").is_in(list(PARENTS)))["parent"]
            frontier = [p for p in parents.drop_nulls().unique() if p not in generation]
            if depth > args.generations:
                frontier = []
            for p in frontier:
                generation[p] = depth
            bar.update()
            bar.set_postfix(ancestors=len(generation) - 1, next=len(frontier))
    facts = pl.concat(rows)

    # Names, from the first of `langs` that has one
    rank = pl.col("language").replace_strict(langs, range(len(langs)), default=None)
    names = (
        pl.concat([pl.scan_parquet(data / "labels" / k / "*.parquet") for k in langs])
        .filter(pl.col("id").is_in(list(generation)))
        .sort(rank, maintain_order=True)
        .unique("id", keep="first")
        .select("id", pl.col("value").alias("name"))
        .collect(engine="streaming")
    )

    def year(prop: str) -> pl.DataFrame:
        """The earliest year given for each id (some have several, from sources that
        disagree)."""
        return (
            facts.filter(pl.col("property") == prop)
            .group_by("id")
            .agg(pl.col("year").min().alias(LIFE[prop]))
        )

    def parent(prop: str) -> pl.DataFrame:
        return (
            facts.filter(pl.col("property") == prop)
            .group_by("id")
            .agg(pl.col("parent").first().alias(PARENTS[prop]))
        )

    people = pl.DataFrame(
        {"generation": list(generation.values()), "id": list(generation)}
    )
    for prop in LIFE:
        people = people.join(year(prop), on="id", how="left")
    for prop in PARENTS:
        people = people.join(parent(prop), on="id", how="left")
    people = (
        people.join(names, on="id", how="left")
        .select("generation", "id", "name", "born", "died", "father", "mother")
        .sort("generation", "born", nulls_last=True)
    )

    root = people.row(0, named=True)
    print(
        f"\n{root['name'] or root['id']}: {people.height - 1} ancestors named on Wikidata"
    )
    per_gen = people.group_by("generation").agg(
        pl.len().alias("people"), pl.col("born").min().alias("earliest born")
    )
    with pl.Config(tbl_rows=-1, tbl_hide_dataframe_shape=True):
        print(per_gen.sort("generation"))
        print("\nThe furthest back:")
        deepest = people["generation"].max()
        print(
            people.filter(pl.col("generation") >= deepest - 1).drop("father", "mother")
        )
    if args.csv:
        people.write_csv(args.csv)
        print(f"\nWrote {people.height} rows to {args.csv}")


if __name__ == "__main__":
    main()
