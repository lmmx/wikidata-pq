"""The items most like a seed item by the sparse autoencoder's features (sae/output/codes.parquet
and features.parquet, from sae/export.py), grouped by the feature that links them most.

Each item's code is weighted by how rare each feature is (log of all coded items over the
feature's items), so that features on millions of items (library authority files,
Freebase) count for little and those on a few thousand (MathWorld + nLab) for a lot.
Similarity is the cosine of the weighted codes. The candidates are the items with any of
the seed's `--features` most heavily weighted features: one scan of the codes.

A seed is a QID, or an English label: of the items with that label, the one with the most
features (the best catalogued).

    uv run --group sae python sae/neighbours.py Q846780
    uv run --group sae python sae/neighbours.py "Kalman filter" --top 50
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import polars as pl

from id_sets import names, show


def seed_id(seed: str, data: Path, codes: pl.LazyFrame) -> str:
    """A QID as given, or the best-catalogued item with that English label."""
    if re.fullmatch(r"Q\d+", seed):
        return seed
    ids = (
        pl.scan_parquet(data / "labels" / "en" / "*.parquet")
        .filter(pl.col("value") == seed)
        .select("id")
        .collect(engine="streaming")["id"]
        .to_list()
    )
    found = (
        codes.filter(pl.col("id").is_in(ids))
        .select("id", pl.col("features").list.len().alias("n"))
        .collect(engine="streaming")
        .sort("n", "id", descending=[True, False])
    )
    if found.is_empty():
        raise SystemExit(f"No coded item is labelled {seed!r}")
    if found.height > 1:
        print(f"{found.height} coded items are labelled {seed!r}; taking the best catalogued")
    return found["id"][0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("seed", help="QID or English label, e.g. Q846780")
    parser.add_argument("--top", type=int, default=30, help="Neighbours shown")
    parser.add_argument(
        "--features", type=int, default=8, help="Seed features to find candidates by"
    )
    parser.add_argument("--out", type=Path, default=Path("sae/output"))
    parser.add_argument("--data", type=Path, default=Path("hub"), help="Local copy")
    args = parser.parse_args()

    codes = pl.scan_parquet(args.out / "codes.parquet")
    features = pl.read_parquet(args.out / "features.parquet")
    total = codes.select(pl.len()).collect().item()
    idf = features.filter(pl.col("items") > 0).select(
        "feature",
        (pl.lit(total) / pl.col("items")).log().alias("idf"),
    )

    def weighted(lf: pl.LazyFrame) -> pl.LazyFrame:
        """Long form: `id`, `feature`, `w` (activation × idf)."""
        return (
            lf.explode(["features", "activations"], empty_as_null=True)
            .rename({"features": "feature", "activations": "activation"})
            .join(idf.lazy(), on="feature")
            .with_columns(w=pl.col("activation") * pl.col("idf"))
            .select("id", "feature", "w")
        )

    seed = seed_id(args.seed, args.data, codes)
    mine = weighted(codes.filter(pl.col("id") == seed)).collect().sort("w", descending=True)
    if mine.is_empty():
        raise SystemExit(f"{seed} has no code (fewer than two kept external IDs?)")
    seed_norm = math.sqrt((mine["w"] ** 2).sum())
    by = mine.head(args.features)["feature"].to_list()

    # Candidates: items with any of the seed's heaviest features; their full codes
    long = weighted(
        codes.filter(
            pl.col("features").list.eval(pl.element().is_in(by)).list.any(),
            pl.col("id") != seed,
        )
    ).collect(engine="streaming")
    norms = long.group_by("id").agg((pl.col("w") ** 2).sum().sqrt().alias("norm"))
    shared = long.join(mine.select("feature", pl.col("w").alias("seed_w")), on="feature")
    scores = (
        shared.with_columns(part=pl.col("w") * pl.col("seed_w"))
        .sort("part", descending=True)
        .group_by("id", maintain_order=True)
        .agg(
            pl.col("part").sum().alias("dot"),
            pl.col("feature").first().alias("via"),
            pl.col("feature").alias("shared"),
        )
        .join(norms, on="id")
        .with_columns(
            (pl.col("dot") / (pl.col("norm") * seed_norm)).alias("similarity")
        )
        .sort("similarity", "id", descending=[True, False])
    )
    nearest = scores.head(args.top)

    label = dict(zip(features["feature"], features["label"]))
    shown = set(nearest["id"]) | {seed}
    name = names(args.data, list(shown))
    print(f"\n{name.get(seed, '')} ({seed}): {mine.height} features, heaviest first")
    show(
        mine.join(features.select("feature", "group", "items"), on="feature")
        .sort("w", descending=True)
        .select(
            "feature",
            "group",
            pl.col("feature").replace_strict(label).alias("label"),
            "items",
            pl.col("w").round(2).alias("weight"),
        )
    )

    print(
        f"\n{scores.height:,} candidates (with any of its {len(by)} heaviest features); "
        f"the {nearest.height} most similar, grouped by the feature linking them most:"
    )
    for via, group in nearest.group_by("via", maintain_order=True):
        print(f"\nVia {via[0]}: {label[via[0]]}")
        show(
            group.select(
                "id",
                pl.col("id").replace_strict(name, default="").alias("name"),
                pl.col("similarity").round(3),
                pl.col("shared")
                .list.eval(pl.element().cast(pl.String))
                .list.join(", ")
                .alias("shared features"),
            )
        )


if __name__ == "__main__":
    main()
