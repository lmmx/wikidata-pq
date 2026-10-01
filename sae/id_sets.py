"""Each item's set of external identifiers (which external-ID properties it holds), counted by
distinct set, from a local copy of the wikidata-pq datasets: the input to a sparse
autoencoder over item codes (sae/train.py).

Two passes over the claims, a file at a time, over the non-deprecated statements of
datatype `external-id`:

1. how many items hold each property, to keep those held by `--min-items` or more, and
   number them (`index`, most held first);
2. each item's sorted set of kept property indexes, the sets with `--min-ids` or more,
   and how many items hold each distinct set, written a file at a time to `parts/`.

Then the parts are merged, by the streaming engine, into one row per distinct set. Many
items share a set (scholarly articles, taxa, authority records), so the sets are far fewer
than the items. An item whose statements straddle a file boundary counts as two partial
sets; there are at most 33 of those.

Writes, to `--out`:

- `id_properties.parquet`: `property` (P...), `name`, `items` (how many items hold it),
  `index` (its column in the model);
- `id_sets.parquet`: `set` (list of property indexes, sorted), `items` (how many hold it).

    python sae/id_sets.py
    python sae/id_sets.py --min-items 100 --min-ids 3
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import polars as pl
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent / "demos"))
from classes import NOT_WIKIPEDIA  # noqa: E402


def external_ids(path: Path) -> pl.LazyFrame:
    """The file's distinct (item, property) pairs of external-ID statements."""
    return (
        pl.scan_parquet(path)
        .filter(pl.col("rank") != "deprecated", pl.col("datatype") == "external-id")
        .select("id", "property")
        .unique()
    )


def names(data: Path, ids: list[str]) -> dict[str, str]:
    """English labels, else multilingual (`mul`)."""
    keys = [k for k in ["en", "mul"] if (data / "labels" / k).is_dir()]
    if not keys:
        return {}
    return dict(
        pl.concat([pl.scan_parquet(data / "labels" / k / "*.parquet") for k in keys])
        .filter(pl.col("id").is_in(ids))
        .sort(pl.col("language").replace_strict(keys, range(len(keys)), default=None))
        .unique("id", keep="first")
        .select("id", "value")
        .collect(engine="streaming")
        .iter_rows()
    )


def wikipedias(data: Path) -> pl.LazyFrame:
    """How many Wikipedias have an article on each item: `id`, `wikipedias`."""
    sites = [
        d
        for d in sorted((data / "links").iterdir())
        if re.fullmatch(r"[a-z_]+wiki", d.name) and d.name not in NOT_WIKIPEDIA
    ]
    return (
        pl.concat([pl.scan_parquet(d / "*.parquet").select("id") for d in sites])
        .group_by("id")
        .agg(pl.len().alias("wikipedias"))
    )


def show(df: pl.DataFrame) -> None:
    with pl.Config(
        tbl_rows=-1,
        tbl_cols=-1,
        tbl_width_chars=200,
        fmt_str_lengths=100,
        thousands_separator=",",
        tbl_hide_dataframe_shape=True,
        tbl_hide_column_data_types=True,
    ):
        print(df)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--min-items", type=int, default=50, help="Drop rarer properties (default 50)"
    )
    parser.add_argument(
        "--min-ids", type=int, default=2, help="Drop smaller sets (default 2)"
    )
    parser.add_argument("--top", type=int, default=30, help="Rows per table")
    parser.add_argument("--data", type=Path, default=Path("hub"), help="Local copy")
    parser.add_argument(
        "--out", type=Path, default=Path("sae/output"), help="Directory to write to"
    )
    args = parser.parse_args()
    data: Path = args.data
    files = sorted((data / "claims" / "all").glob("*.parquet"))

    # 1. How many items hold each property, and how many items hold any
    counts, totals = [], 0
    for f in tqdm(files, desc="properties", unit="file"):
        pairs = external_ids(f)
        held, n = pl.collect_all(
            [
                pairs.group_by("property").agg(pl.len().alias("items")),
                pairs.select(pl.col("id").n_unique().alias("n")),
            ],
            engine="streaming",
        )
        counts.append(held)
        totals += n.item()
    held = pl.concat(counts).group_by("property").agg(pl.col("items").sum())
    properties = (
        held.filter(pl.col("items") >= args.min_items)
        .sort("items", "property", descending=[True, False])
        .with_columns(index=pl.int_range(pl.len(), dtype=pl.UInt16))
    )
    label = names(data, properties["property"].to_list())
    properties = properties.select(
        "property",
        pl.col("property").replace_strict(label, default=None).alias("name")
        if label
        else pl.lit(None, pl.String).alias("name"),
        "items",
        "index",
    )
    args.out.mkdir(parents=True, exist_ok=True)
    properties.write_parquet(args.out / "id_properties.parquet")
    index = properties.lazy().select("property", "index")

    # 2. Each item's set of kept properties, counted by set, a file at a time
    parts = args.out / "parts"
    shutil.rmtree(parts, ignore_errors=True)
    parts.mkdir()
    for f in tqdm(files, desc="sets", unit="file"):
        (
            external_ids(f)
            .join(index, on="property")
            .group_by("id")
            .agg(pl.col("index").sort().alias("set"))
            .filter(pl.col("set").list.len() >= args.min_ids)
            .group_by("set")
            .agg(pl.len().cast(pl.UInt32).alias("items"))
            .sink_parquet(parts / f.name)
        )

    print("Merging the sets...")
    (
        pl.scan_parquet(parts / "*.parquet")
        .group_by("set")
        .agg(pl.col("items").sum())
        .sink_parquet(args.out / "id_sets.parquet")
    )
    shutil.rmtree(parts)

    sets = pl.scan_parquet(args.out / "id_sets.parquet")
    sizes = (
        sets.group_by(pl.col("set").list.len().alias("properties"))
        .agg(pl.len().alias("sets"), pl.col("items").sum())
        .sort("properties")
        .collect(engine="streaming")
    )
    n_sets, used = sizes["sets"].sum(), sizes["items"].sum()
    counts = (
        sets.select(pl.col("items").sort(descending=True))
        .collect(engine="streaming")["items"]
        .cum_sum()
    )

    print(
        f"{totals:,} items have an external ID, over {held.height:,} properties "
        f"(an item straddling two files counts twice)"
    )
    print(
        f"Kept {properties.height:,} properties (held by {args.min_items:,}+ items) and "
        f"{used:,} items ({100 * used / totals:.1f}%) with {args.min_ids}+ of them, "
        f"in {n_sets:,} distinct sets"
    )
    print(f"Wrote {args.out / 'id_sets.parquet'} and {args.out / 'id_properties.parquet'}")

    print("\nSet sizes (properties per item):")
    show(sizes.with_columns((100 * pl.col("items") / used).round(2).alias("% of items")))

    print("\nHow concentrated: the items in the most common sets")
    tops = [k for k in [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000] if k <= n_sets]
    show(
        pl.DataFrame(
            {"top sets": tops, "items": [counts[k - 1] for k in tops]}
        ).with_columns((100 * pl.col("items") / used).round(1).alias("% of items"))
    )

    print(f"\nThe {args.top} properties held by the most items:")
    show(properties.head(args.top).select("property", "name", "items"))

    print(f"\nThe {args.top} most common sets:")
    name_of = dict(
        zip(
            properties["index"],
            properties["name"].fill_null(properties["property"]),
        )
    )
    show(
        sets.top_k(args.top, by="items")
        .collect(engine="streaming")
        .select(
            "items",
            pl.col("set")
            .map_elements(
                lambda s: ", ".join(name_of[i] for i in s), return_dtype=pl.String
            )
            .alias("set"),
        )
    )


if __name__ == "__main__":
    main()
