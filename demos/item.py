"""One Wikidata item in one language, from a local copy of the six wikidata-pq datasets.

Reads the label, description, aliases and Wikipedia page of an item, and its statements
with every property, item and unit named, from `{data}/{table}/{key}/*.parquet` (the
layout `just download` leaves in hub/, and `snapshot_download` with
`local_dir={data}/{table}` gives).

    python demos/item.py Q42
    python demos/item.py Q64 --lang de --data hub

Names come from the language, then `mul` (Wikidata's names that hold in every language),
then `en`, as Wikidata shows them. Every table is sorted by id, so each lookup reads only
the row groups that can hold the item.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("item", nargs="?", default="Q42", help="Item id (default Q42)")
    parser.add_argument("--lang", default="en", help="Language code (default en)")
    parser.add_argument("--data", type=Path, default=Path("hub"), help="Local copy")
    args = parser.parse_args()

    data: Path = args.data
    item: str = args.item

    def keys(table: str, *wanted: str) -> list[str]:
        """The wanted keys, in order, that the table has a folder for."""
        return [k for k in dict.fromkeys(wanted) if (data / table / k).is_dir()]

    langs = keys("labels", args.lang, "mul", "en")

    def scan(table: str, *wanted: str) -> pl.LazyFrame:
        paths = [data / table / k / "*.parquet" for k in keys(table, *wanted)]
        if not paths:
            raise SystemExit(f"No {table} for {wanted} in {data / table}")
        return pl.concat([pl.scan_parquet(p) for p in paths])

    def first(lf: pl.LazyFrame, *by: str) -> pl.LazyFrame:
        """One row per `by`, from the first language in `langs` that has one."""
        rank = pl.col("language").replace_strict(langs, range(len(langs)), default=None)
        return lf.sort(rank, maintain_order=True).unique(
            by, keep="first", maintain_order=True
        )

    is_item = pl.col("id") == item
    label = first(scan("labels", *langs).filter(is_item), "id").collect()
    description = first(scan("descriptions", *langs).filter(is_item), "id").collect()
    aliases = scan("aliases", *langs).filter(is_item).collect()
    site = f"{args.lang.replace('-', '_')}wiki"
    wikipedia = (
        scan("links", site).filter(is_item).collect()
        if keys("links", site)
        else pl.DataFrame()
    )

    # Statements, with monolingual text kept to the same languages
    dv = pl.col("datavalue").struct
    claims = (
        scan("claims", "all")
        .filter(is_item)
        .filter(
            (pl.col("datatype") != "monolingualtext")
            | dv.field("language").is_in(langs)
        )
        .collect()
    )
    if label.is_empty() and claims.is_empty():
        raise SystemExit(f"No {item} in {data}")

    # Names of the properties, items and units the statements refer to
    refs = (
        pl.concat(
            [
                claims["property"],
                claims["datavalue"].struct.field("id"),
                claims["datavalue"].struct.field("unit"),
            ]
        )
        .drop_nulls()
        .unique()
    )
    names = first(
        scan("claims_labels", *langs).filter(pl.col("ref").is_in(refs.implode())),
        "field",
        "ref",
    ).collect()

    def name(field: str, alias: str) -> pl.DataFrame:
        return names.filter(pl.col("field") == field).select(
            pl.col("ref").alias(alias), pl.col("label").alias(f"{alias}_label")
        )

    statements = (
        claims.with_columns(item=dv.field("id"), unit=dv.field("unit"))
        .join(name("property-labels", "property"), on="property", how="left")
        .join(name("labels", "item"), on="item", how="left")
        .join(name("unit-labels", "unit"), on="unit", how="left")
        .select(
            "property",
            "property_label",
            pl.coalesce(
                "item_label",
                dv.field("text"),
                dv.field("datavalue__string"),
                pl.when(dv.field("amount").is_not_null()).then(
                    pl.concat_str(
                        dv.field("amount"), "unit_label", separator=" ", ignore_nulls=True
                    )
                ),
                dv.field("time"),
                "item",
            ).alias("value"),
            "rank",
        )
    )

    def show(heading: str, df: pl.DataFrame, col: str) -> None:
        values = [f"{r[col]} ({r['language']})" for r in df.iter_rows(named=True)]
        print(f"{heading:<12} {', '.join(values) or '-'}")

    print(f"{item}, in {' > '.join(langs)}\n")
    show("label", label, "value")
    show("description", description, "value")
    show("aliases", aliases, "value")
    pages = [f"{r['site']}: {r['title']}" for r in wikipedia.iter_rows(named=True)]
    print(f"{'wikipedia':<12} {', '.join(pages) or '-'}\n")

    print(
        f"{statements.height} statements on "
        f"{statements['property'].n_unique()} properties"
    )
    with pl.Config(tbl_rows=-1, fmt_str_lengths=60, tbl_hide_dataframe_shape=True):
        print(statements)


if __name__ == "__main__":
    main()
