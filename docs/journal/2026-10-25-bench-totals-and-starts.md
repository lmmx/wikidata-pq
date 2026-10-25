# 2026-10-25: A measured end-to-end total, the pipeline's worker starts, and a dump-shaped slice (bench 11.0.0)

Continues docs/journal/2026-10-25-claims-weights.md (bench 10.0.0, lmmx/wikidata-pq#128),
which listed what its ETA still overstated. Bench 11.0.0 is lmmx/wikidata-pq#130. The
full figures and sources are in bench/CHANGELOG.md 11.0.0.

## Current State

- **The measured total.** Each run now gives the sample end to end: the split, the
  pipeline and finalise's first run, CPU time and wall time summed. Nothing in it is
  modelled. The comment shows it under the score and as the table's first two rows.
- **The ETA by stage.** The comment also gives each stage's measured CPU time, how it is
  scaled to a release, and its rate: the ETA's seconds per CPU-second on the sample.
  Moving work from one stage to another leaves the measured total where it was. The ETA
  moves by the difference of the two rates.
- **The pipeline's worker starts** are a fixed cost, counted for a release from its
  chunks and the pool's settings.
- **The dump slice** is written as the dumps write their lines.
- **Left alone, for want of facts on record:** the other tables' weights, and
  `merge_group` by table.

## Facts and their sources

- **A pipeline worker's start, measured.** It costs 0.62 ± 0.08 CPU-s (run 40208739011,
  12 runs; 0.68 on the PR's first run, 0.72 on #128's). This is the worker's CPU time at its exit, as its atexit
  handlers run (`phases.track_process`), less its chunks' CPU time (`bench_work`). Of
  that, 0.61 comes before its first chunk: the interpreter, the spawn and
  `wikidata.main`'s imports. The first chunk on a worker costs another 0.02-0.03 s
  (full tier), which stays in the per-chunk part.
- **A release's chunks, measured:** 12,182 in the main set and 10,853 in the scholarly
  set (`data/manifest.jsonl`, 20260928).
- **A release's starts, counted:** 2,884. `pool.process_chunks`, replayed with chunks of
  equal length at `CHUNKS_PER_WORKER` = 8 and 4 workers, starts 1,524 + 1,360. The
  bounds are 2,884-2,890.
  - The replay matches the bench: it counts 4 for the quick tier, as measured.
  - On the full tier it counts 12 against 10-12 measured.
- **The escaping.** The share of a line's bytes in strings that carry escapes, in the
  dumps' form: 13.1% (c0), 9.8% (r50), 15.2% (s30), 17.4% (s90). The coordinator's figure
  for real lines was 13.3%. In orjson's form the bench had none of those escapes, and
  floats like `2.77778e-5` were written `0.0000277778`.
- **The other tables' costs.** They go with their partition files, not their rows. On
  the full tier, a partition file costs 0.32-0.72 ms and a row 0.8-1.9 µs, by table (fit
  over the 11 chunk files).
  - A release's partition files a chunk: **not on record**.
  - Its rows by table at 20260928: **not on record**. Only the 20260507 main set's are.
  - Its scholarly set's rows: **not on record**.

## The model, old against new, on the same runs

PR #130's run 40208739011 (12 runs: EPYC 9V45 ×6, EPYC 7763 ×4, EPYC 9V74 ×2), as the
mean ± its 95% interval. Both models were computed by the harness in each run:

| | bench 10.0.0's model | bench 11.0.0 | change |
|---|---|---|---|
| ETA: pipeline | 4h15m ± 33m | 3h37m ± 29m | −14.9% ± 0.4% (−38m ± 4m) |
| ETA: split | 2h04m ± 16m (orjson's slice) | 2h08m ± 15m (the dumps' form) | +3.6% ± 2.8% |
| ETA, a release | 8h45m ± 1h10m | 8h11m ± 1h06m | −6.4% ± 0.6% (−34m ± 6m) |
| the pipeline's starts at a release, CPU | 2h31m (14,668 starts) | 30m (2,884) | |

- **The same pipeline model elsewhere:** −14.6% ± 0.4% on the PR's first run, and
  −14.3% ± 0.3% applied offline to #128's run. Bench 10.0.0 had estimated "perhaps
  −20%".
- **The split on the dumps' form:** 6.2% ± 2.8% more CPU than on orjson's form, and
  7.7% ± 3.0% more wall. The orjson slice's split runs second, after the timed run, so
  some of that may be what running second does.
- **The measured total:** CPU 125.2 ± 12.8 s (the split 23.4, the pipeline 19.3,
  finalise 82.5) and wall 46.0 ± 5.0 s. Neither moved beyond noise between the arms.
- **Parity:** passed, including the split's on both slices.

## The other tables: left alone

With a release's main-set chunks taken as the bench's routed ones (195-416 files a table)
and its scholarly ones as the `s` files:
- these tables come to 8,800-10,700 CPU-s at a release, against the 16,200 that entity
  scaling gives them: −10% to −14% of the pipeline ETA;
- with `c0`'s files a chunk instead, +4%.

That assumption is not on record, and it sets the sign, so the weights stay. Scaling
these tables by rows alone would understate a release, because their cost is mostly per
file. The data to settle it is on the release machine: its audit sidecars
(`audit/<table>/chunk_*.parquet`) give rows and files by table and chunk. Committing
their totals per set would make this a measured weight. Each bench run now records the
sample's side (`pipeline_tables`).

## `merge_group`: not timed apart

`merge_group` glues every table but `claims_labels` at once, a thread each, and pq_glue
does the work on its own threads. A per-table clock would count the other tables' threads
too, or miss pq_glue's. Running the tables one at a time would change the wall time being
measured. It remains scaled by entities: 1.1 CPU-s, about 7% of the pipeline's.

## Reading moves between stages

The rates on run 40208739011, in ETA seconds per CPU-second on the sample:
- the split: 327;
- the pipeline's claims stages: 365 (main) and 610 (scholarly);
- the rest of the pipeline: 1,142;
- finalise per byte: 330 (claims) and 917 (the rest).

So the claims' decode moved from the pipeline into the split (the work queue's next
reshape step) should leave the measured total about where it was. The ETA should change
by roughly (327 − 365) to (327 − 610) ETA-seconds per CPU-second moved, a small fall,
plus whatever work the move removes.
