# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "polars", "torch", "tqdm"]
# ///
"""Train a Matryoshka sparse autoencoder over items' sets of external-ID properties
(sae/output/id_sets.parquet, from sae/id_sets.py), after Bussmann et al., "Learning
Multi-Level Features with Matryoshka Sparse Autoencoders" (ICML 2025).

Each input is one distinct set, as a 0/1 vector over the kept properties. The encoder
maps it to a dictionary of features, of which only `k` per input on average are kept
(BatchTopK: the `k × batch` largest activations in the batch). The decoder maps the
features back to a logit per property. The Matryoshka part: the dictionary is ordered,
and the loss is the mean, over nested prefixes (by default the first 64, 256, 1024 and
4096 features), of how well that prefix alone rebuilds the set, so early features learn
what is broad and later ones what is specific. The inputs are 0/1, so the loss is binary
cross-entropy on the logits, not the paper's squared error.

Sets are drawn with probability proportional to `items ** alpha`: 1 draws by item (a
fifth of the items are stars or places), 0 draws each set alike (mostly people with many
authority IDs). After training, the activation threshold that BatchTopK settled on is
kept, so an item's code does not depend on its batch (sae/export.py).

Writes `--out` (sae/output/sae.pt), and prints, for the first features, the properties
each rebuilds most strongly.

    uv run sae/train.py
    uv run sae/train.py --samples 300_000_000 --k 12 --alpha 0.3
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class MatryoshkaSAE(nn.Module):
    def __init__(self, n_inputs: int, prefixes: list[int], k: int) -> None:
        super().__init__()
        m = prefixes[-1]
        self.prefixes, self.k = prefixes, k
        self.enc = nn.Linear(n_inputs, m)
        self.dec = nn.Parameter(self.enc.weight.detach().clone())  # (m, n_inputs)
        self.bias = nn.Parameter(torch.zeros(n_inputs))
        # Running mean of the smallest activation BatchTopK keeps, used after training
        self.register_buffer("threshold", torch.tensor(-1.0))

    def encode(self, x: torch.Tensor, batch_topk: bool = True) -> torch.Tensor:
        acts = F.relu(self.enc(x))
        if not batch_topk:
            return acts * (acts > self.threshold)
        top = acts.flatten().topk(self.k * len(x))
        kept = torch.zeros_like(acts).flatten().scatter_(0, top.indices, top.values)
        return kept.view_as(acts)

    def losses(self, x: torch.Tensor, acts: torch.Tensor) -> list[torch.Tensor]:
        """Each prefix's binary cross-entropy, summed over properties, mean over inputs."""
        logits, start, out = self.bias.expand_as(x), 0, []
        for end in self.prefixes:
            logits = logits + acts[:, start:end] @ self.dec[start:end]
            out.append(
                F.binary_cross_entropy_with_logits(logits, x, reduction="sum") / len(x)
            )
            start = end
        return out


def batch(
    rows: torch.Tensor, starts: torch.Tensor, lengths: torch.Tensor,
    indices: torch.Tensor, n_inputs: int,
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
    parser.add_argument("--out", type=Path, default=Path("sae/output/sae.pt"))
    parser.add_argument(
        "--prefixes", type=int, nargs="+", default=[64, 256, 1024, 4096],
        help="Nested dictionary sizes (default 64 256 1024 4096)",
    )
    parser.add_argument("--k", type=int, default=8, help="Active features per set, on average")
    parser.add_argument("--alpha", type=float, default=0.5, help="Draw sets by items ** alpha")
    parser.add_argument("--samples", type=float, default=100e6, help="Sets drawn in all")
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show", type=int, default=64, help="Features to print")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
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

    # Hold out 1% of the sets to measure on
    rng = np.random.default_rng(args.seed)
    held = rng.random(sets.height) < 0.01
    to = lambda a: torch.from_numpy(a).to(device)  # noqa: E731
    lengths_t, starts_t, indices_t = to(lengths), to(starts), to(indices)
    train_w = to(np.where(held, 0.0, weights))
    held_rows = to(np.flatnonzero(held))
    held_w = to(weights[held])

    model = MatryoshkaSAE(n_inputs, args.prefixes, args.k).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    steps = math.ceil(args.samples / args.batch)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimiser, lambda s: min(1.0, s / 1000) * min(1.0, 5 * (steps - s) / steps)
    )
    m = args.prefixes[-1]
    last_fired = torch.zeros(m, dtype=torch.long, device=device)

    @torch.no_grad()
    def measure() -> dict[str, float]:
        """On held-out sets drawn like the training ones: each prefix's loss, and how
        many of each set's properties are among its top logits (recall at its size)."""
        rows = held_rows[torch.multinomial(held_w, 4096, replacement=True)]
        x = batch(rows, starts_t, lengths_t, indices_t, n_inputs)
        acts = model.encode(x, batch_topk=False)
        logits = acts @ model.dec + model.bias
        size = x.sum(1, keepdim=True)
        rank = logits.argsort(1, descending=True).argsort(1)
        recall = ((rank < size) * x).sum() / size.sum()
        return {
            "recall": recall.item(),
            "active": (acts > 0).sum(1).float().mean().item(),
            **{f"loss@{p}": l.item() for p, l in zip(args.prefixes, model.losses(x, acts))},
        }

    bar = tqdm(range(steps), desc="train", unit="step")
    for step in bar:
        rows = torch.multinomial(train_w, args.batch, replacement=True)
        x = batch(rows, starts_t, lengths_t, indices_t, n_inputs)
        acts = model.encode(x)
        losses = model.losses(x, acts)
        loss = torch.stack(losses).mean()
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        schedule.step()
        with torch.no_grad():
            fired = (acts > 0).any(0)
            last_fired[fired] = step
            positive = acts[acts > 0]
            if len(positive):
                t, smallest = model.threshold, positive.min()
                model.threshold = smallest if t < 0 else 0.99 * t + 0.01 * smallest
        if step % 200 == 0:
            dead = (step - last_fired > 2000).float().mean().item() if step > 2000 else 0.0
            bar.set_postfix(loss=f"{loss.item():.3f}", dead=f"{dead:.1%}")
        if step % 5000 == 0 or step == steps - 1:
            stats = measure()
            tqdm.write(
                f"step {step:,}: "
                + ", ".join(f"{k} {v:.3f}" for k, v in stats.items())
                + f", dead {(step - last_fired > 2000).float().mean().item():.1%}"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state": model.state_dict(),
            "n_inputs": n_inputs,
            "prefixes": args.prefixes,
            "k": args.k,
            "properties": properties["property"].to_list(),
            "args": {k: str(v) for k, v in vars(args).items()},
        },
        args.out,
    )
    print(f"Wrote {args.out}; threshold {model.threshold.item():.4f}")

    # A first look: what the broadest features rebuild
    names = properties["name"].fill_null(properties["property"]).to_list()
    dec = model.dec.detach().cpu()
    print(f"\nThe first {args.show} features, by the properties each raises most:")
    level = 0
    for f in range(min(args.show, m)):
        while f >= args.prefixes[level]:
            level += 1
        top = dec[f].topk(6)
        print(
            f"{f:>4} (level {args.prefixes[level]}): "
            + ", ".join(f"{names[i]} {w:.1f}" for w, i in zip(top.values, top.indices))
        )


if __name__ == "__main__":
    main()
