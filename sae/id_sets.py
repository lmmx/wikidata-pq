"""Each item's set of external identifiers (which external-ID properties it holds), counted by
distinct set, from a local copy of the wikidata-pq datasets: the input to a sparse
autoencoder over item codes (sae/train.py).

One pass over the claims, a file at a time: the non-deprecated statements of datatype
`external-id`, as each item's sorted set of property numbers (P214 → 214), then how many
items hold each distinct set. Many items share a set (scholarly articles, taxa, authority
records), so the sets are far fewer than the items. An item whose statements straddle a
file boundary counts as two partial sets; there are at most 33 of those.

Then, over all files, the properties held by fewer than `--min-items` items are dropped
from the sets, and the sets left with fewer than `--min-ids` properties are dropped.

Writes, to `--out`:

- `id_sets.parquet`: `set` (list of property numbers, sorted), `items` (how many hold it);
- `id_properties.parquet`: `property` (P...), `number`, `name`, `items` (how many items
  hold it, among those kept), `index` (its column in the model, most held first).

    python sae/id_sets.py
    python sae/id_sets.py --min-items 100 --min-ids 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl
from tqdm import tqdm


def file_sets(path: Path) -> pl.DataFrame:
    """The file's items' sets of external-ID property numbers, counted: `set`, `items`."""
    return (
        pl.scan_parquet(path)
        .filter(pl.col("rank") != "deprecated", pl.col("datatype") == "external-id")
        .select("id", pl.col("property").str.slice(1).cast(pl.UInt32).alias("number"))
        .unique()
        .group_by("id")
        .agg(pl.col("number").sort().alias("set"))
        .group_by("set")
        .agg(pl.len().alias("items"))
        .collect(engine="streaming")
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
    raw = (
        pl.concat([file_sets(f) for f in tqdm(files, desc="claims", unit="file")])
        .group_by("set")
        .agg(pl.col("items").sum())
    )
    items = raw["items"].sum()

    # How many items hold each property, before any is dropped
    held = (
        raw.explode("set", empty_as_null=True)
        .group_by(pl.col("set").alias("number"))
        .agg(pl.col("items").sum())
    )
    kept = held.filter(pl.col("items") >= args.min_items)["number"].implode()

    sets = (
        raw.with_columns(
            pl.col("set").list.eval(pl.element().filter(pl.element().is_in(kept)))
        )
        .filter(pl.col("set").list.len() >= args.min_ids)
        .group_by("set")
        .agg(pl.col("items").sum())
        .sort("items", descending=True)
    )
    used = sets["items"].sum()

    properties = (
        sets.explode("set", empty_as_null=True)
        .group_by(pl.col("set").alias("number"))
        .agg(pl.col("items").sum())
        .sort("items", "number", descending=[True, False])
        .with_columns(
            property=pl.format("P{}", "number"),
            index=pl.int_range(pl.len(), dtype=pl.UInt32),
        )
    )
    label = names(data, properties["property"].to_list())
    properties = properties.select(
        "property",
        "number",
        pl.col("property").replace_strict(label, default=None).alias("name")
        if label
        else pl.lit(None, pl.String).alias("name"),
        "items",
        "index",
    )

    args.out.mkdir(parents=True, exist_ok=True)
    sets.write_parquet(args.out / "id_sets.parquet")
    properties.write_parquet(args.out / "id_properties.parquet")

    print(
        f"{items:,} items have an external ID, in {raw.height:,} distinct sets, "
        f"over {held.height:,} properties"
    )
    print(
        f"Kept {properties.height:,} properties (held by {args.min_items:,}+ items) and "
        f"{used:,} items ({100 * used / items:.1f}%) with {args.min_ids}+ of them, "
        f"in {sets.height:,} distinct sets"
    )
    print(f"Wrote {args.out / 'id_sets.parquet'} and {args.out / 'id_properties.parquet'}")

    print("\nSet sizes (properties per item):")
    show(
        sets.group_by(pl.col("set").list.len().alias("properties"))
        .agg(pl.len().alias("sets"), pl.col("items").sum())
        .sort("properties")
        .with_columns((100 * pl.col("items") / used).round(2).alias("% of items"))
    )

    print("\nHow concentrated: the items in the most common sets")
    cumulative = sets["items"].cum_sum()
    tops = [k for k in [1, 10, 100, 1_000, 10_000, 100_000] if k <= sets.height]
    show(
        pl.DataFrame(
            {"top sets": tops, "items": [cumulative[k - 1] for k in tops]}
        ).with_columns((100 * pl.col("items") / used).round(1).alias("% of items"))
    )

    print(f"\nThe {args.top} properties held by the most items:")
    show(properties.head(args.top).select("property", "name", "items"))

    print(f"\nThe {args.top} most common sets:")
    name_of = dict(zip(properties["number"], properties["name"].fill_null("")))
    show(
        sets.head(args.top).select(
            "items",
            pl.col("set")
            .map_elements(
                lambda s: ", ".join(name_of.get(n) or f"P{n}" for n in s),
                return_dtype=pl.String,
            )
            .alias("set"),
        )
    )


if __name__ == "__main__":
    main()
