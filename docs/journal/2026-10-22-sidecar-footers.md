# 2026-10-22: Audit sidecar stats from the footers

Scripts and logs are in `2026-10-22-sidecar-footers/`.

**The change:** `write_sidecar` (partitioning/core.py) takes each partition file's row
count and its ids' min and max from the file's Parquet footer, which the sink has just
written. Before, it ran a Polars query over each file's data.

**The effect, on 66 chunks with 6 workers:**
- **Wall time:** 7.5% less (161.7 s to 149.6 s), with workers taking 8 chunks each and
  the claims explode change (docs/journal/2026-10-22-claims-explode-views.md).
- **Output:** sidecars are byte-identical.

## Why

Once the claims explodes stopped copying, the sidecars were the next cost in
partitioning that did no work of its own.
- **On chunk 0:** writing the sidecars took 3.9 s of a 12–14 s partition (2,321 files,
  one `scan_parquet(...).collect()` each). On routed chunk 50 it was 1.5 s of 2.9 s.
- **What a sidecar needs:** each file's row count and the min and max of its `id` (or
  `ref`) column. The sink writes all three into the file's footer.

## Change

`file_stats(f)` reads the footer with pyarrow (`pq.read_metadata`):
- **Row count:** `num_rows`.
- **Bounds:** the min of the row groups' min statistics on the id column, and the max of
  their maxes.
- **When it reads the file instead:** when any row group has no min/max, or a bound is 60
  bytes or longer.
  - Polars truncates a string's statistics to 64 bytes: strings of 65, 200 and 5,000
    bytes were all stored as 64 (`logs/sidecar.txt`).
  - pyarrow 25 does not expose Parquet's `is_min_value_exact` / `is_max_value_exact`
    flags, so length is the test. 60 leaves room for a truncation that stops at a UTF-8
    character boundary.
  - Wikidata ids are far shorter, so in practice no file is read.

A query over all of a table's files at once (`pl.concat` of the scans, not kept) took
1.0–1.4 s on chunk 0, against 0.16–0.21 s from the footers. In the pool it saved 4%, not
7.5% (`logs/pool_66.txt`).

## Correctness

- **Real outputs** (`sidecar_vs_master.py`, `sidecar_stats.py`):
  - The sidecars are byte-identical to master's for every table of chunk 0 (2,321
    partition files) and routed chunk 50 (849).
  - Every file's footer bounds equal a scan's.
- **Tests:** `scripts/test_sidecar.py` (new) checks `file_stats` against a scan on:
  - ids long enough to be truncated, ids just under that, and multi-byte ids;
  - no rows, some null ids, only null ids;
  - a `ref` column, and three row groups;
  - and a `partition_parquet` sidecar against scans of its files.

  With the length check removed, so that truncated bounds would be trusted, the first
  case fails.
- **Whole pipeline** (`logs/outputs_identical.txt`): every file a 22-chunk run writes,
  14,330 under `results/` and `audit/`, is byte-identical between perf/persistent-workers
  and it with the explode change and this one. Both ran with 8 chunks per worker, in the
  same run directory.
- **Lint:** flake8 and `ty check` are clean on the changed files.

## Measured

**Sidecars alone** (`logs/sidecar.txt`):

| outputs | master | footers | peak RSS, fresh process |
|---|---|---|---|
| chunk 0, 2,321 files | 3.9 s | 0.3 s | 114 → 110 MB |
| routed chunk 50, 849 files | 1.6 s | 0.13 s | |

**The pool** (`logs/pool_66.txt`):
- **Setup:** 66 chunks, 6 workers, 8 chunks per worker, 4-core VM. Both trees have
  perf/persistent-workers and perf/claims-explode-views. Order was shuffled each round,
  each run after `primer.py`.

| | wall | summed RSS peak | summed RSS mean | partitioning, summed |
|---|---|---|---|---|
| per-file scans | 160.6, 160.8, 163.5 s | 8.1–9.5 GB | 5.9–6.1 GB | 657–672 s |
| footers | 147.6, 146.7, 154.5 s | 8.9–9.7 GB | 6.4–6.5 GB | 542–563 s |

**Memory:**
- **Per worker:** peak RSS is unchanged, at most 2.8–2.9 GB in both.
- **Summed:** RSS is 0.45 GB higher on average. The sidecars themselves use almost no
  memory. The workers now spend less of each chunk in the light sidecar phase, so more
  of the run's time is spent with several of them in memory-heavy phases at once.
- **RSS over time:** summed RSS integrated over the run (mean × wall) is unchanged:
  966 GB·s against 972.
- **The one-query variant** showed the same rise, 0.45 GB on average.

**CPU:** the chunks' summed `process()` time rises from 236 s to 278 s. With
partitioning shorter, more workers run `process()` at the same time on the 4 cores.
Their summed user CPU falls, from 467–474 s to 432–445 s.

**All three together** (`logs/pool_66_together.txt`), against a process per chunk:

| chunks per worker | wall | summed RSS peak | summed RSS mean |
|---|---|---|---|
| 1 | 163.9, 157.8 s | 7.9–8.1 GB | 5.6–5.7 GB |
| 8 | 153.3, 149.5 s | 9.3–9.5 GB | 6.5 GB |
| never replaced | 146.5, 149.7 s | 9.0–9.4 GB | 6.7 GB |

## Limits

- **Statistics:** this relies on the partition files having statistics, which Polars'
  `sink_parquet` writes by default (`statistics=True`).
- **Length test:** it assumes that Polars truncates statistics to 64 bytes. If Polars
  truncated shorter, a bound between that length and 60 bytes would be trusted.
  `scripts/test_sidecar.py` would catch a change that affects such ids.
