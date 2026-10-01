"""The sparse autoencoder's results as tables to publish and to query from a browser (the
Space in space/), from sae/output (sae/export.py).

- `features.parquet`: each feature, as in sae/output/features.parquet, with its `idf`
  (log of all coded items over its items), its number of `children` and its 40
  `strongest` items (by weight).
- `items.parquet`: `id`, `label` (English, else multilingual, else another language), `kinds` (its "instance of"
  classes, as Q numbers, or for a class with none its "subclass of" parents), `is_class`
  (whether `kinds` are those parents), `features`, `activations`, `weights` (activation × idf) and `norm`
  (of the weights), sorted by `id` in small row groups, so that looking up one item reads
  one row group.
- `postings.parquet`: `feature`, `id` (a Q number), `unit16` (the item's weight for the
  feature over its norm, 0 to 1, times 65,535 as a 16-bit integer) and `kinds`, every (feature, item) pair (or each feature's
  `--postings` heaviest items), sorted by feature and id, so that a feature's items are a
  run of row groups. The ids are delta-encoded and the units byte-stream-split. The cosine of two items is the sum, over their shared features, of
  one's unit times the other's weight, over the other's norm; capping the postings drops the
  items of broad features that only weigh moderately on them, which the neighbours need.
- `classes.parquet`: `class` (Q number), `label`, `parents` ("subclass of", as Q numbers)
  and `items` (coded items with it among their `kinds`), for every class the coded
  items are instances of and every class above those, so that a page can tell whether an
  item is an instance of some class or of anything below it.
- `members.parquet`: `class`, `id` (Q numbers) and `subclass` (whether the item is a
  subclass of the class, else an instance), each class's direct members among the coded
  items, sorted by class, so that a page can list the items under a type.
- `names.parquet`: `key` (the label or alias, lowercased), `label`, `alias` (set on an
  alias's row), `id`, `description` (English), `wikipedias` and `coded` (whether the model
  has a code for it), for every labelled item with an external ID and its English and
  multilingual aliases, sorted by `key` and then by Wikipedias, so that a search for the items whose
  label starts with some text reads only the row groups whose keys can hold it.
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
import pyarrow.parquet as pq

from id_sets import wikipedias

ROW_GROUP = 20_000
POSTINGS_ROW_GROUP = 50_000
STRONGEST = 40
ZSTD_LEVEL = 9  # level 19 saves a further 3-10% and writes about 15 times slower
UNIT_SCALE = 65535  # postings store weight / norm (0 to 1) as a 16-bit integer
# Label languages to fall back on, after English and multilingual
LABEL_FALLBACK = ["de", "fr", "es", "it", "pt", "nl", "sv", "pl", "ru", "ja", "zh"]


def write(
    df: pl.DataFrame,
    path: Path,
    row_group_size: int = ROW_GROUP,
    encodings: dict[str, str] | None = None,
) -> None:
    """Write for a browser to read a few row groups at a time: zstd at `ZSTD_LEVEL`
    (`--zstd-level`), version 2 data pages (which hyparquet needs for DELTA_BYTE_ARRAY),
    dictionaries for every column not given an encoding."""
    encodings = encodings or {}
    pq.write_table(
        df.to_arrow(),
        path,
        row_group_size=row_group_size,
        compression="zstd",
        compression_level=getattr(write, "level", ZSTD_LEVEL),
        data_page_version="2.0",
        use_dictionary=[c for c in df.columns if c not in encodings],
        column_encoding=encodings or None,
    )


def qnumber(expr: pl.Expr) -> pl.Expr:
    """A `Q…` id as its number (null for any other id)."""
    return (
        pl.when(expr.str.starts_with("Q"))
        .then(expr.str.slice(1).cast(pl.UInt32, strict=False))
        .otherwise(None)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sae", type=Path, default=Path("sae/output"))
    parser.add_argument("--out", type=Path, default=Path("sae/output/publish"))
    parser.add_argument(
        "--properties", type=Path, default=Path("sae/output/id_properties.parquet")
    )
    parser.add_argument(
        "--postings", type=int, default=0, help="Items kept per feature (default all)"
    )
    parser.add_argument("--data", type=Path, default=Path("hub"), help="Local copy")
    parser.add_argument(
        "--zstd-level",
        type=int,
        default=ZSTD_LEVEL,
        help="zstd level (default 9; 19 is 3-10%% smaller and about 15 times slower)",
    )
    args = parser.parse_args()
    write.level = args.zstd_level
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
    claims = pl.scan_parquet(args.data / "claims" / "all" / "*.parquet").filter(
        pl.col("rank") != "deprecated"
    )
    # Every item with an external ID, coded or not, and a label for each: English, else
    # multilingual, else the first of some widely used languages, else any (an Italian food
    # with a protected name may have only an Italian label)
    catalogued = (
        claims.filter(pl.col("datatype") == "external-id").select("id").unique()
    )
    print("Labelling the catalogued items...")
    order = [*langs, *[k for k in LABEL_FALLBACK if (args.data / "labels" / k).is_dir()]]
    all_labels = (
        pl.scan_parquet(args.data / "labels" / "*" / "*.parquet")
        .join(catalogued, on="id", how="semi")
        .with_columns(
            rank=pl.col("language").replace_strict(
                order, range(len(order)), default=len(order), return_dtype=pl.UInt16
            )
        )
        .sort("rank", "language")
        .unique("id", keep="first")
        .select("id", pl.col("value").alias("label"))
        .collect(engine="streaming")
    )
    labels = all_labels.lazy()
    # Kinds: each coded item's "instance of" classes, and the classes above them
    value = pl.col("datavalue").struct.field("id")
    print("Reading the items' kinds and the subclass hierarchy...")
    instance_of, subclass_of = pl.collect_all(
        [
            claims.filter(pl.col("property") == "P31")
            .join(codes.select("id"), on="id", how="semi")
            .select("id", qnumber(value).alias("kind"))
            .drop_nulls()
            .unique(),
            claims.filter(pl.col("property") == "P279")
            .select(qnumber(pl.col("id")).alias("class"), qnumber(value).alias("parent"))
            .drop_nulls()
            .unique(),
        ],
        engine="streaming",
    )
    # Members: each class's direct instances and subclasses among the coded items, so that a
    # page can list the items under a type
    coded_q = codes.select(qnumber(pl.col("id")).alias("q")).collect(engine="streaming")
    members = (
        pl.concat(
            [
                instance_of.select(
                    pl.col("kind").alias("class"),
                    qnumber(pl.col("id")).alias("id"),
                    pl.lit(False).alias("subclass"),
                ),
                subclass_of.join(coded_q, left_on="class", right_on="q").select(
                    pl.col("parent").alias("class"),
                    pl.col("class").alias("id"),
                    pl.lit(True).alias("subclass"),
                ),
            ]
        )
        .unique(["class", "id"], keep="first")
        .sort("class", "id")
    )
    write(
        members,
        args.out / "members.parquet",
        POSTINGS_ROW_GROUP,
        {"class": "DELTA_BINARY_PACKED", "id": "DELTA_BINARY_PACKED"},
    )
    print(f"members.parquet: {members.height:,} (class, item) pairs")
    del members, coded_q

    # A class with no "instance of" (attention, say: a subclass of mechanism) takes its
    # "subclass of" parents as its kinds, and is marked `is_class`
    as_class = (
        codes.select("id", qnumber(pl.col("id")).alias("class"))
        .join(instance_of.lazy().select("id").unique(), on="id", how="anti")
        .join(subclass_of.lazy(), on="class")
        .select("id", pl.col("parent").alias("kind"))
        .unique()
        .collect(engine="streaming")
    )
    is_class = as_class.select("id").unique().with_columns(is_class=pl.lit(True))
    instance_of = pl.concat([instance_of, as_class])
    print(f"{is_class.height:,} classes with no instance of take their superclasses as kinds")
    known = instance_of.select(pl.col("kind").alias("class")).unique()
    frontier = known
    while frontier.height:
        frontier = (
            subclass_of.join(frontier, on="class", how="semi")
            .select(pl.col("parent").alias("class"))
            .unique()
            .join(known, on="class", how="anti")
        )
        known = pl.concat([known, frontier])
    class_labels = (
        pl.concat(
            [pl.scan_parquet(args.data / "labels" / k / "*.parquet") for k in langs]
        )
        .filter(pl.col("id").str.starts_with("Q"))
        .with_columns(qnumber(pl.col("id")).alias("class"))
        .join(known.lazy(), on="class", how="semi")
        .sort(pl.col("language").replace_strict(langs, range(len(langs)), default=None))
        .unique("class", keep="first")
        .select("class", pl.col("value").alias("label"))
        .collect(engine="streaming")
    )
    classes = (
        known.join(class_labels, on="class", how="left")
        .join(
            subclass_of.join(known, on="class", how="semi")
            .group_by("class")
            .agg(pl.col("parent").sort().alias("parents")),
            on="class",
            how="left",
        )
        .join(
            instance_of.group_by(pl.col("kind").alias("class")).agg(
                pl.len().cast(pl.UInt32).alias("items")
            ),
            on="class",
            how="left",
        )
        .with_columns(pl.col("items").fill_null(0))
        .sort("class")
    )
    write(classes, args.out / "classes.parquet", 1 << 20)
    print(f"classes.parquet: {classes.height:,} classes")
    kinds = instance_of.group_by("id").agg(pl.col("kind").sort().alias("kinds"))

    print("Weighting the codes and labelling the items...")
    items = (
        weighted.join(labels, on="id", how="left")
        .join(kinds.lazy(), on="id", how="left")
        .join(is_class.lazy(), on="id", how="left")
        .with_columns(pl.col("is_class").fill_null(False))
        .select("id", "label", "kinds", "is_class", "features", "activations", "weights", "norm")
        .sort("id")
        .collect(engine="streaming")
    )
    write(items, args.out / "items.parquet", ROW_GROUP, {"id": "DELTA_BYTE_ARRAY"})
    print(f"items.parquet: {items.height:,} items")

    # Postings: each feature's items, by Q number, with their weight over their norm (so a
    # cosine is a sum of products); the feature's heaviest items go in the features table
    ranked = (
        items.lazy()
        .select("id", "norm", "kinds", "features", "weights")
        .explode(["features", "weights"], empty_as_null=True)
        .rename({"features": "feature", "weights": "weight"})
        .sort("feature", "weight", "id", descending=[False, True, False])
        .with_columns(rank=pl.int_range(pl.len(), dtype=pl.UInt32).over("feature"))
        .filter(pl.col("rank") < (args.postings or 2**32 - 1))
        .collect(engine="streaming")
    )
    strongest = (
        ranked.filter(pl.col("rank") < STRONGEST)
        .group_by("feature", maintain_order=True)
        .agg(pl.col("id").alias("strongest"))
    )
    postings = (
        ranked.select(
            "feature",
            qnumber(pl.col("id")).alias("id"),
            (pl.col("weight") / pl.col("norm") * UNIT_SCALE)
            .round()
            .clip(0, UNIT_SCALE)
            .cast(pl.UInt16)
            .alias("unit16"),
            "kinds",
        )
        .sort("feature", "id")
    )
    del ranked
    write(
        postings,
        args.out / "postings.parquet",
        POSTINGS_ROW_GROUP,
        {
            "feature": "DELTA_BINARY_PACKED",
            "id": "DELTA_BINARY_PACKED",
            "unit16": "BYTE_STREAM_SPLIT",
        },
    )
    print(f"postings.parquet: {postings.height:,} rows")
    features = features.join(strongest, on="feature", how="left")
    write(features, args.out / "features.parquet", 1 << 20)
    print(f"features.parquet: {features.height:,} features")

    # Names: every labelled item with an external ID, coded or not (`coded`), by lowercased
    # label, to search by prefix; an item the model has no code for can then be found, and
    # the page can say why it has no features
    print("Indexing the names...")
    descriptions = (
        pl.scan_parquet(args.data / "descriptions" / "en" / "*.parquet")
        .join(catalogued, on="id", how="semi")
        .unique("id", keep="first")
        .select("id", pl.col("value").alias("description"))
    )
    by_label = (
        labels.join(descriptions, on="id", how="left")
        .join(wikipedias(args.data), on="id", how="left")
        .join(items.lazy().select("id", coded=pl.lit(True)), on="id", how="left")
        .with_columns(
            pl.col("wikipedias").fill_null(0).cast(pl.UInt16),
            pl.col("coded").fill_null(False),
        )
        .collect(engine="streaming")
    )
    # Aliases (English and multilingual) as more rows, keyed by the alias, with `alias` set,
    # so that a search can include them ("red squirrel" for Eurasian red squirrel)
    alias_langs = [k for k in ["en", "mul"] if (args.data / "aliases" / k).is_dir()]
    aliases = (
        pl.concat(
            [pl.scan_parquet(args.data / "aliases" / k / "*.parquet") for k in alias_langs]
        )
        .select("id", pl.col("value").alias("alias"))
        .join(by_label.lazy(), on="id")
        .with_columns(akey=pl.col("alias").str.to_lowercase())
        .filter(pl.col("akey") != pl.col("label").str.to_lowercase())
        .unique(["id", "akey"], keep="first")
        .drop("akey")
        if alias_langs
        else by_label.lazy().with_columns(alias=pl.lit(None, pl.String)).head(0)
    )
    names = (
        pl.concat(
            [
                by_label.lazy().with_columns(alias=pl.lit(None, pl.String)),
                aliases,
            ],
            how="diagonal",
        )
        .with_columns(key=pl.coalesce("alias", "label").str.to_lowercase())
        .sort("key", "wikipedias", "id", descending=[False, True, False])
        .select("key", "label", "alias", "id", "description", "wikipedias", "coded")
        .collect(engine="streaming")
    )
    write(names, args.out / "names.parquet", ROW_GROUP, {"key": "DELTA_BYTE_ARRAY"})
    print(
        f"names.parquet: {by_label.height:,} labelled items with an external ID, "
        f"{by_label['coded'].sum():,} of them coded, and "
        f"{names.height - by_label.height:,} aliases"
    )

    shutil.copy(args.properties, args.out / "id_properties.parquet")
    model = args.out / "model"
    model.mkdir(exist_ok=True)
    for name in ["ae.pt", "config.json"]:
        shutil.copy(src / "sae" / "trainer_0" / name, model / name)
    if (src / "sae" / "run.json").exists():
        shutil.copy(src / "sae" / "run.json", model / "run.json")
    for f in sorted(args.out.rglob("*")):
        if f.is_file():
            print(f"{f.relative_to(args.out)}: {f.stat().st_size / 1e6:,.1f} MB")


if __name__ == "__main__":
    main()
