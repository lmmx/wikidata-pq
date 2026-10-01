"""Train a Matryoshka sparse autoencoder over items' sets of external-ID properties
(sae/output/id_sets.parquet, from sae/id_sets.py), with the Matryoshka BatchTopK trainer of
saprmarks/dictionary_learning (Bussmann et al., "Learning Multi-Level Features with
Matryoshka Sparse Autoencoders", ICML 2025).

Each input is one distinct set, as a 0/1 vector over the kept properties, drawn with
probability proportional to `items ** alpha`: 1 draws by item (a fifth of the items are
stars or places), 0 draws each set alike (mostly people with many authority IDs). The first
run (v0) drew by `items ** 0.5` and 100M sets, and its broadest features went to films and
libraries; the defaults, 0.75 and 200M, lean towards the domains with the most items. 1% of
the sets are held out, to measure the trained model on.

The trainer writes `--out`/trainer_0/ae.pt and config.json, and this script the settings
to `--out`/run.json. Then it prints, on the held-out sets, how many of each set's
properties are among its top reconstructed values, and, for the first features, the
properties each one's decoder row raises most.

    uv run --group sae python sae/train.py
    uv run --group sae python sae/train.py --samples 300e6 --k 12 --alpha 1
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import polars as pl
import torch
from dictionary_learning.trainers.matryoshka_batch_top_k import (
    MatryoshkaBatchTopKSAE,
    MatryoshkaBatchTopKTrainer,
)
from dictionary_learning.training import trainSAE


def batch(
    rows: torch.Tensor,
    starts: torch.Tensor,
    lengths: torch.Tensor,
    indices: torch.Tensor,
    n_inputs: int,
) -> torch.Tensor:
    """The sets at `rows` as a dense 0/1 matrix, from the CSR arrays."""
    lens = lengths[rows]
    which = torch.repeat_interleave(torch.arange(len(rows), device=rows.device), lens)
    offset = torch.arange(len(which), device=rows.device) - torch.repeat_interleave(
        lens.cumsum(0) - lens, lens
    )
    x = torch.zeros(len(rows), n_inputs, device=rows.device)
    x[which, indices[starts[rows][which] + offset]] = 1.0
    return x


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sets", type=Path, default=Path("sae/output/id_sets.parquet"))
    parser.add_argument(
        "--properties", type=Path, default=Path("sae/output/id_properties.parquet")
    )
    parser.add_argument("--out", type=Path, default=Path("sae/output/sae"))
    parser.add_argument(
        "--groups",
        type=int,
        nargs="+",
        default=[64, 192, 768, 3072],
        help="Matryoshka group sizes, broadest first (default 64 192 768 3072)",
    )
    parser.add_argument("--k", type=int, default=8, help="Active features per set, on average")
    parser.add_argument("--alpha", type=float, default=0.75, help="Draw sets by items ** alpha")
    parser.add_argument("--samples", type=float, default=200e6, help="Sets drawn in all")
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show", type=int, default=64, help="Features to print")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)

    sets = pl.read_parquet(args.sets)
    properties = pl.read_parquet(args.properties).sort("index")
    n_inputs = properties.height
    lengths = sets["set"].list.len().to_numpy().astype(np.int64)
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    indices = sets["set"].explode(empty_as_null=True).to_numpy().astype(np.int64)
    weights = sets["items"].to_numpy().astype(np.float64) ** args.alpha
    print(
        f"{sets.height:,} sets over {n_inputs:,} properties, {len(indices):,} entries; "
        f"training on {args.device}"
    )

    rng = np.random.default_rng(args.seed)
    held = rng.random(sets.height) < 0.01
    to = lambda a: torch.from_numpy(a).to(device)  # noqa: E731
    lengths_t, starts_t, indices_t = to(lengths), to(starts), to(indices)
    train_w = to(np.where(held, 0.0, weights))
    held_rows, held_w = to(np.flatnonzero(held)), to(weights[held])

    def draws(w: torch.Tensor, rows: torch.Tensor | None = None):
        while True:
            drawn = torch.multinomial(w, args.batch, replacement=True)
            yield batch(
                drawn if rows is None else rows[drawn],
                starts_t, lengths_t, indices_t, n_inputs,
            )

    steps = math.ceil(args.samples / args.batch)
    dict_size = sum(args.groups)
    trainSAE(
        data=draws(train_w),
        trainer_configs=[
            {
                "trainer": MatryoshkaBatchTopKTrainer,
                "steps": steps,
                "activation_dim": n_inputs,
                "dict_size": dict_size,
                "k": args.k,
                "layer": 0,  # required, for language models
                "lm_name": "wikidata-id-sets",
                "group_fractions": [g / dict_size for g in args.groups],
                "decay_start": int(0.8 * steps),
                "seed": args.seed,
                "device": args.device,
                "wandb_name": "wikidata-id-sets",
            }
        ],
        steps=steps,
        save_dir=str(args.out),
        log_steps=2000,
        verbose=True,
        device=args.device,
    )
    settings = {
        "alpha": args.alpha, "samples": int(args.samples), "k": args.k,
        "groups": args.groups, "batch": args.batch, "seed": args.seed,
    }
    (args.out / "run.json").write_text(json.dumps(settings, indent=2) + "\n")
    path = args.out / "trainer_0" / "ae.pt"
    ae = MatryoshkaBatchTopKSAE.from_pretrained(path, device=args.device)
    print(f"Wrote {path}")

    # On held-out sets: recall at each set's size, features active, features never active
    held_out = draws(held_w, held_rows)
    hits = total = active = 0
    fired = torch.zeros(dict_size, dtype=torch.bool, device=device)
    with torch.no_grad():
        for _ in range(25):
            x = next(held_out)
            f = ae.encode(x)
            x_hat = ae.decode(f)
            size = x.sum(1, keepdim=True)
            rank = x_hat.argsort(1, descending=True).argsort(1)
            hits += ((rank < size) * x).sum().item()
            total += size.sum().item()
            active += (f > 0).sum().item()
            fired |= (f > 0).any(0)
    n = 25 * args.batch
    print(
        f"Held out: recall at set size {hits / total:.3f}, "
        f"{active / n:.1f} features active per set, "
        f"{(~fired).sum().item():,} of {dict_size:,} features never active"
    )

    names = properties["name"].fill_null(properties["property"]).to_list()
    ends = np.cumsum(args.groups)
    dec = ae.W_dec.detach().cpu()
    print(f"\nThe first {args.show} features, by the properties each raises most:")
    for i in range(min(args.show, dict_size)):
        group = int(np.searchsorted(ends, i, side="right"))
        top = dec[i].topk(6)
        print(
            f"{i:>4} (group {group}): "
            + ", ".join(f"{names[j]} {w:.2f}" for w, j in zip(top.values, top.indices))
        )


if __name__ == "__main__":
    main()
