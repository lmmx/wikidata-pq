"""The trained Matryoshka sparse autoencoder (sae/train.py) as tables: what each feature is,
and which features each item has.

1. Every distinct set of external-ID properties (sae/output/id_sets.parquet) is encoded on
   the GPU, with the threshold the trainer settled on. Along the way, how many items each
   feature is active on, and on how many items each pair of features is active together.
2. A feature's parent is the feature of a broader group most often active with it (the
   share of its items on which that feature is active too), which makes the groups a tree;
   a feature never active has none.
3. One pass over the claims, a file at a time: each item's set, joined to its set's code
   (skipped with `--no-items`, which reuses codes.parquet from a previous run). For
   examples, each feature's items in the most Wikipedias, of those with the feature among
   their strongest three.

Writes, to `--out`:

- `features.parquet`: `feature`, `group`, `label` (its top 3 properties), `properties` and
  `weights` (its top 10 by decoder weight), `items`, `sets`, `parent`, `parent_label`,
  `parent_share`, `examples` (named items);
- `codes.parquet`: `id`, `features`, `activations` (strongest first).

`--find` prints the features that raise some properties most, e.g. MathWorld (P2812) and
nLab (P4215).

    uv run --group sae python sae/export.py --find P2812 P4215
    uv run --group sae python sae/export.py --no-items    # reuse codes.parquet
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch
from dictionary_learning.trainers.matryoshka_batch_top_k import MatryoshkaBatchTopKSAE
from tqdm import tqdm

from id_sets import external_ids, names, show
from train import batch

sys.path.append(str(Path(__file__).resolve().parent.parent / "demos"))
from classes import NOT_WIKIPEDIA  # noqa: E402

EXAMPLES = 5
KEY = pl.col("set").cast(pl.List(pl.String)).list.join(",").alias("key")


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--model", type=Path, default=Path("sae/output/sae/trainer_0/ae.pt")
    )
    parser.add_argument("--sets", type=Path, default=Path("sae/output/id_sets.parquet"))
    parser.add_argument(
        "--properties", type=Path, default=Path("sae/output/id_properties.parquet")
    )
    parser.add_argument("--out", type=Path, default=Path("sae/output"))
    parser.add_argument("--find", nargs="*", default=[], help="Property ids, e.g. P2812")
    parser.add_argument("--no-items", action="store_true", help="Reuse codes.parquet")
    parser.add_argument("--top", type=int, default=20, help="Rows per table")
    parser.add_argument("--batch", type=int, default=16384)
    parser.add_argument("--data", type=Path, default=Path("hub"), help="Local copy")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)

    ae = MatryoshkaBatchTopKSAE.from_pretrained(args.model, device=args.device)
    groups = ae.group_sizes.tolist()
    m = sum(groups)
    starts_of = np.concatenate([[0], np.cumsum(groups)[:-1]])
    group_of = np.repeat(np.arange(len(groups)), groups)

    properties = pl.read_parquet(args.properties).sort("index")
    prop_names = properties["name"].fill_null(properties["property"]).to_list()
    n_inputs = properties.height
    sets = pl.read_parquet(args.sets).with_row_index("row")
    lengths = sets["set"].list.len().to_numpy().astype(np.int64)
    to = lambda a: torch.from_numpy(a).to(device)  # noqa: E731
    lengths_t = to(lengths)
    starts_t = to(np.concatenate([[0], np.cumsum(lengths)[:-1]]))
    indices_t = to(sets["set"].explode(empty_as_null=True).to_numpy().astype(np.int64))
    items_t = to(sets["items"].to_numpy().astype(np.float32))

    # 1. Encode every set; count items per feature and per pair of features
    together = torch.zeros(m, m, device=device)
    n_sets = torch.zeros(m, device=device)
    rows_out, features_out, acts_out = [], [], []
    with torch.no_grad():
        for b in tqdm(range(0, sets.height, args.batch), desc="encode", unit="batch"):
            rows = torch.arange(b, min(b + args.batch, sets.height), device=device)
            f = ae.encode(batch(rows, starts_t, lengths_t, indices_t, n_inputs))
            on = (f > 0).float()
            together += on.T @ (on * items_t[rows, None])
            n_sets += on.sum(0)
            r, c = on.nonzero(as_tuple=True)
            rows_out.append((rows[r]).cpu().numpy())
            features_out.append(c.cpu().numpy())
            acts_out.append(f[r, c].cpu().numpy())
    n_items = together.diagonal().clone()
    active = pl.DataFrame(
        {
            "row": np.concatenate(rows_out).astype(np.uint32),
            "feature": np.concatenate(features_out).astype(np.uint16),
            "activation": np.concatenate(acts_out),
        }
    )
    set_codes = (
        active.sort("activation", descending=True)
        .group_by("row", maintain_order=True)
        .agg(pl.col("feature").alias("features"), pl.col("activation").alias("activations"))
        .join(sets.select("row", "items", KEY), on="row")
    )
    print(
        f"{sets.height:,} sets encoded: {active.height / sets.height:.1f} features "
        f"active per set, {sets.height - set_codes.height:,} sets with none"
    )

    # 2. Each feature's parent: of the broader groups' features, the one most often
    # active with it
    share = together / n_items.clamp(min=1)[:, None]
    parent = np.full(m, -1)
    parent_share = np.zeros(m, dtype=np.float32)
    for g in range(1, len(groups)):
        lo, hi = starts_of[g], starts_of[g] + groups[g]
        best = share[lo:hi, :lo].max(1)
        parent[lo:hi] = best.indices.cpu().numpy()
        parent_share[lo:hi] = best.values.cpu().numpy()
    parent[(n_items == 0).cpu().numpy()] = -1  # never active: no parent

    top = ae.W_dec.detach().topk(10, dim=1)
    weights, top_idx = top.values.cpu().numpy(), top.indices.cpu().numpy()
    labels = [", ".join(prop_names[j] for j in top_idx[i, :3]) for i in range(m)]
    features = pl.DataFrame(
        {
            "feature": np.arange(m, dtype=np.uint16),
            "group": group_of.astype(np.uint8),
            "label": labels,
            "properties": [[prop_names[j] for j in top_idx[i]] for i in range(m)],
            "weights": [weights[i].tolist() for i in range(m)],
            "items": n_items.cpu().numpy().astype(np.int64),
            "sets": n_sets.cpu().numpy().astype(np.int64),
            "parent": [int(p) if p >= 0 else None for p in parent],
            "parent_label": [labels[p] if p >= 0 else None for p in parent],
            "parent_share": [float(s) if p >= 0 else None for p, s in zip(parent, parent_share)],
        }
    )

    # 3. One pass over the claims: each item's code (or the codes from a previous run)
    codes_path = args.out / "codes.parquet"
    if not args.no_items:
        index = properties.lazy().select("property", "index")
        codes_lf = set_codes.lazy().select("key", "features", "activations")
        parts = args.out / "codes_parts"
        shutil.rmtree(parts, ignore_errors=True)
        parts.mkdir(parents=True)
        files = sorted((args.data / "claims" / "all").glob("*.parquet"))
        for f in tqdm(files, desc="items", unit="file"):
            (
                external_ids(f)
                .join(index, on="property")
                .group_by("id")
                .agg(pl.col("index").sort().alias("set"))
                .select("id", KEY)
                .join(codes_lf, on="key")
                .select("id", "features", "activations")
                .sink_parquet(parts / f.name)
            )
        pl.scan_parquet(parts / "*.parquet").sink_parquet(codes_path)
        shutil.rmtree(parts)
        n_coded = pl.scan_parquet(codes_path).select(pl.len()).collect().item()
        print(f"Wrote {codes_path}: {n_coded:,} items")

    # Examples: the items in the most Wikipedias among those with the feature in their
    # strongest three
    examples = pl.DataFrame(schema={"feature": pl.UInt16, "examples": pl.List(pl.String)})
    if codes_path.exists():
        print("Counting the coded items' Wikipedias for examples...")
        best = (
            pl.scan_parquet(codes_path)
            .join(wikipedias(args.data), on="id")
            .select("id", "wikipedias", pl.col("features").list.head(3).alias("feature"))
            .explode("feature", empty_as_null=True)
            .sort("wikipedias", descending=True)
            .group_by("feature", maintain_order=True)
            .head(EXAMPLES)
            .collect(engine="streaming")
        )
        label_of = names(args.data, best["id"].unique().to_list())
        examples = best.group_by("feature", maintain_order=True).agg(
            pl.format(
                "{} ({})", pl.col("id").replace_strict(label_of, default=""), "id"
            ).alias("examples")
        )

    features = features.join(examples, on="feature", how="left")
    features.write_parquet(args.out / "features.parquet")
    print(f"Wrote {args.out / 'features.parquet'}")

    print("\nThe groups:")
    show(
        features.group_by("group")
        .agg(
            pl.len().alias("features"),
            (pl.col("items") > 0).sum().alias("live"),
            pl.col("items").filter(pl.col("items") > 0).median().alias("median items"),
            pl.col("parent_share").median().alias("median parent share"),
        )
        .sort("group")
    )

    shown = ["feature", "group", "label", "items", "parent_label", "examples"]
    for g in range(len(groups)):
        print(f"\nGroup {g}: the {args.top} features on the most items")
        show(
            features.filter(pl.col("group") == g)
            .sort("items", descending=True)
            .head(args.top)
            .select(shown)
        )

    if args.find:
        idx = properties.filter(pl.col("property").is_in(args.find))["index"].to_list()
        found_names = ", ".join(prop_names[i] for i in idx)
        score = ae.W_dec.detach()[:, idx].sum(1).cpu().numpy()
        print(f"\nThe {args.top} features that raise {found_names} most:")
        show(
            features.with_columns(score=score)
            .filter(pl.col("items") > 0)
            .sort("score", descending=True)
            .head(args.top)
            .select("feature", "group", pl.col("score").round(2), *shown[2:])
        )


if __name__ == "__main__":
    main()
