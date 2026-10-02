#!/usr/bin/env -S uv run
# /// script
# dependencies = ["polars", "tqdm"]
# ///
"""Which classes did philippesaade/wikidata leave out?

Reads `id, claims` from a release's split chunks, takes each entity's "instance of" (P31)
values from its main snaks by regex (the claims JSON is written by dump.py in a fixed key
order), and marks the entities absent from the reduced copy (`--missing`, the anti-join of
the release's ids against philippesaade's). Ids above the largest id present in both are
counted as "new" (created after 2026-05-07), the rest as "old": a class left out by the
filter has nearly all its old entities missing.

Writes `--out` (id, p31, missing, new) and prints, per class, its entities and the share of
its old ones missing, then how many old missing entities none of the mostly-missing
classes covers. Classes are shown with their English labels from the Wikidata API, cached
beside `--out`.
"""

import argparse
import json
import urllib.parse
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import polars as pl
from tqdm import tqdm

P31 = (
    r'"mainsnak":\{"snaktype":"value","property":"P31","datatype":"wikibase-item",'
    r'"datavalue":\{"entity-type":"item","numeric-id":\d+,"id":"(Q\d+)"'
)


API = "https://www.wikidata.org/w/api.php"
USER_AGENT = "wikidata-pq/0.1 (https://github.com/lmmx/wikidata-pq)"


def fetch_labels(qids: list[str]) -> pl.DataFrame:
    rows = []
    for i in tqdm(range(0, len(qids), 50), desc="labels"):
        query = urllib.parse.urlencode({
            "action": "wbgetentities", "ids": "|".join(qids[i : i + 50]),
            "props": "labels", "languages": "en", "format": "json",
        })
        request = urllib.request.Request(f"{API}?{query}", headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request) as response:
            entities = json.load(response)["entities"]
        rows += [(q, e.get("labels", {}).get("en", {}).get("value")) for q, e in entities.items()]
    return pl.DataFrame(rows, schema={"p31": pl.String, "label": pl.String}, orient="row")


def labelled(frame: pl.DataFrame, cache: Path) -> pl.DataFrame:
    """`frame` with the English label of each class in `p31` (a QID or a list of them)."""
    known = pl.read_parquet(cache) if cache.exists() else pl.DataFrame(schema={"p31": pl.String, "label": pl.String})
    listed = frame.schema["p31"] != pl.String
    qids = (frame["p31"].explode() if listed else frame["p31"]).drop_nulls().unique()
    new = sorted(set(qids) - set(known["p31"]))
    if new:
        known = pl.concat([known, fetch_labels(new)]).unique("p31")
        known.write_parquet(cache)
    lookup = dict(known.iter_rows())
    label = (
        pl.col("p31").list.eval(pl.element().replace_strict(lookup, default=None).fill_null(pl.element())).list.join(", ")
        if listed
        else pl.col("p31").replace_strict(lookup, default=None)
    )
    return frame.with_columns(label=label)


def chunk_p31(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path, columns=["id", "claims"]).select(
        "id", p31=pl.col("claims").str.extract_all(P31).list.eval(
            pl.element().str.extract(r'"id":"(Q\d+)"$')
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="releases/{release}/data")
    parser.add_argument("--missing", type=Path, required=True, help="ids missing from the copy")
    parser.add_argument("--out", type=Path, default=Path("p31_survey.parquet"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.99)
    args = parser.parse_args()

    if not args.out.exists():
        paths = sorted(args.data.glob("chunk_*.parquet"))
        with ProcessPoolExecutor(args.workers) as pool:
            frames = list(tqdm(pool.map(chunk_p31, paths, chunksize=4), total=len(paths)))
        missing = pl.read_parquet(args.missing).select("id", missing=pl.lit(True))
        df = (
            pl.concat(frames)
            .join(missing, on="id", how="left")
            .with_columns(pl.col("missing").fill_null(False), num=pl.col("id").str.slice(1).cast(pl.Int64, strict=False))
        )
        last_kept = df.filter(~pl.col("missing"), pl.col("id").str.starts_with("Q"))["num"].max()
        df = df.with_columns(new=pl.col("num") > last_kept).drop("num")
        df.write_parquet(args.out)
        print(f"{len(df):,} entities, last id in the copy Q{last_kept}")

    df = pl.read_parquet(args.out)
    cache = args.out.with_suffix(".labels.parquet")
    old = df.filter(~pl.col("new"))
    print(old.group_by("missing").len(), df.group_by("new", "missing").len().sort("new", "missing"))
    print("old missing entities with no P31:", old.filter(pl.col("missing"), pl.col("p31").list.len() == 0).height)

    by_class = (
        old.filter(pl.col("p31").list.len() > 0)
        .explode("p31", empty_as_null=False)
        .group_by("p31")
        .agg(entities=pl.len(), missing=pl.col("missing").sum())
        .with_columns(share=pl.col("missing") / pl.col("entities"))
    )
    with pl.Config(tbl_rows=60, fmt_str_lengths=80, tbl_width_chars=160):
        print("Classes by old entities missing:")
        print(labelled(by_class.sort("missing", descending=True).head(60), cache))
    dropped = by_class.filter(pl.col("share") >= args.threshold, pl.col("missing") >= 100)
    with pl.Config(tbl_rows=200, fmt_str_lengths=80, tbl_width_chars=160):
        print(f"Classes with >= {args.threshold:.0%} of their old entities missing (>= 100):")
        print(labelled(dropped.sort("missing", descending=True), cache))
    covered = old.filter(pl.col("missing")).with_columns(
        hit=pl.col("p31").list.set_intersection(dropped["p31"].implode()).list.len() > 0
    )
    print("old missing entities covered by those classes:", covered["hit"].sum(), "of", covered.height)
    with pl.Config(tbl_rows=30, fmt_str_lengths=80, tbl_width_chars=160):
        print("P31 sets of the uncovered ones:")
        uncovered = covered.filter(~pl.col("hit")).group_by("p31").len()
        print(labelled(uncovered.sort("len", descending=True).head(30), cache))


if __name__ == "__main__":
    main()
