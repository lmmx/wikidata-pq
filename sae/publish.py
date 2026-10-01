"""The sparse autoencoder's results as tables to publish and to query from a browser (the
Space in space/), from sae/output (sae/export.py).

- `features.parquet`: each feature, as in sae/output/features.parquet, with its `idf`
  (log of all coded items over its items) and its number of `children`.
- `items.parquet`: `id`, `label` (English, else multilingual), `features`, `activations`,
  `weights` (activation × idf) and `norm` (of the weights), sorted by `id` in small row
  groups, so that looking up one item reads one row group.
- `postings.parquet`: `feature`, `rank`, `id`, `weight`, `norm`, every (feature, item)
  pair (or each feature's `--postings` heaviest items), sorted by feature and rank, so that
  a feature's items are a run of row groups and its strongest come first. Capping them
  drops the items of broad features that only weigh moderately on them, which the
  neighbours need. The neighbours of an item are the items in the postings of
  its heaviest features, by the weights shared over the item's norms.
- `id_properties.parquet` and `model/` (the weights and trainer config), to encode items
  anew.

Written to `--out` (sae/output/publish), to upload as a dataset.

    python sae/publish.py
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import polars as pl

ROW_GROUP = 20_000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sae", type=Path, default=Path("sae/output"))
    parser.add_argument("--out", type=Path, default=Path("sae/output/publish"))
    parser.add_argument(
        "--postings", type=int, default=0, help="Items kept per feature (default all)"
    )
    parser.add_argument("--data", type=Path, default=Path("hub"), help="Local copy")
    args = parser.parse_args()
    src: Path = args.sae
    args.out.mkdir(parents=True, exist_ok=True)

    codes = pl.scan_parquet(src / "codes.parquet")
    total = codes.select(pl.len()).collect().item()
    features = pl.read_parquet(src / "features.parquet").with_columns(
        pl.when(pl.col("items") > 0)
        .then((pl.lit(total) / pl.col("items")).log())
        .alias("idf")
    )
    children = (
        features.filter(pl.col("items") > 0)
        .group_by(pl.col("parent").alias("feature"))
        .agg(pl.len().cast(pl.UInt32).alias("children"))
        .drop_nulls()
        .with_columns(pl.col("feature").cast(pl.UInt16))
    )
    features = features.join(children, on="feature", how="left").with_columns(
        pl.col("children").fill_null(0)
    )
    features.write_parquet(args.out / "features.parquet")
    print(f"features.parquet: {features.height:,} features")

    # Items: the codes with weights, norms and labels
    idf = features.select("feature", "idf").drop_nulls().lazy()
    weighted = (
        codes.with_row_index("row")
        .explode(["features", "activations"], empty_as_null=True)
        .rename({"features": "feature", "activations": "activation"})
        .join(idf, on="feature")
        .with_columns(weight=(pl.col("activation") * pl.col("idf")).cast(pl.Float32))
        .sort("row", "weight", descending=[False, True])
        .group_by("row", "id", maintain_order=True)
        .agg(
            pl.col("feature").alias("features"),
            pl.col("activation").alias("activations"),
            pl.col("weight").alias("weights"),
            (pl.col("weight") ** 2).sum().sqrt().alias("norm"),
        )
        .drop("row")
    )
    langs = [k for k in ["en", "mul"] if (args.data / "labels" / k).is_dir()]
    labels = (
        pl.concat(
            [pl.scan_parquet(args.data / "labels" / k / "*.parquet") for k in langs]
        )
        .join(codes.select("id"), on="id", how="semi")
        .sort(pl.col("language").replace_strict(langs, range(len(langs)), default=None))
        .unique("id", keep="first")
        .select("id", pl.col("value").alias("label"))
    )
    print("Weighting the codes and labelling the items...")
    items = (
        weighted.join(labels, on="id", how="left")
        .select("id", "label", "features", "activations", "weights", "norm")
        .sort("id")
        .collect(engine="streaming")
    )
    items.write_parquet(args.out / "items.parquet", row_group_size=ROW_GROUP)
    print(f"items.parquet: {items.height:,} items")

    # Postings: each feature's items, heaviest first
    postings = (
        items.lazy()
        .select("id", "norm", "features", "weights")
        .explode(["features", "weights"], empty_as_null=True)
        .rename({"features": "feature", "weights": "weight"})
        .sort("feature", "weight", "id", descending=[False, True, False])
        .with_columns(rank=pl.int_range(pl.len(), dtype=pl.UInt32).over("feature"))
        .filter(pl.col("rank") < (args.postings or 2**32 - 1))
        .select("feature", "rank", "id", "weight", "norm")
        .collect(engine="streaming")
    )
    postings.write_parquet(args.out / "postings.parquet", row_group_size=ROW_GROUP)
    print(f"postings.parquet: {postings.height:,} rows")

    shutil.copy(src / "id_properties.parquet", args.out / "id_properties.parquet")
    model = args.out / "model"
    model.mkdir(exist_ok=True)
    for name in ["ae.pt", "config.json"]:
        shutil.copy(src / "sae" / "trainer_0" / name, model / name)
    for f in sorted(args.out.rglob("*")):
        if f.is_file():
            print(f"{f.relative_to(args.out)}: {f.stat().st_size / 1e6:,.1f} MB")


if __name__ == "__main__":
    main()
