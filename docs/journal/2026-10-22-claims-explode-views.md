# 2026-10-22: The claims explodes copy nothing

Continues docs/journal/2026-10-21-claims-arrow-explode.md. Scripts and logs are in
`2026-10-22-claims-explode-views/`.

**The change:** `explode` (partitioning/claims.py) keeps Polars' own string views on the
way to Arrow and back, and gathers the other columns in Polars. A column with a null or
empty list still goes through Arrow's large types, because pyarrow's `take` has no kernel
for string views.

**The effect, on 66 chunks with 6 workers:**
- **Wall time:** 17% less with 8 chunks per worker (192 s to 159 s), 20% less with a
  process per chunk (221 s to 176 s).
- **Memory:** summed worker RSS is about a third lower. The peak falls from 12.5 GB to
  8.3 GB, and a worker's own peak from 4.9 GB to 2.8 GB.
- **Output:** every output file is byte-identical.

## Why

A worker that takes several chunks (perf/persistent-workers) keeps the memory of its last
chunk until the next one reuses it: 2.5–2.9 GB after chunk 0. Worker RSS after a chunk
was close to that chunk's peak. That peak was in partitioning, and within partitioning
in `claims_base`.

**What a worker holds** (`logs/retained.txt`, `retain.py`): after chunk 0, no DataFrame
or Arrow object of the chunk is alive, and `gc.collect()` frees nothing. It is all
memory the allocators keep for reuse:
- **pyarrow's pool:** about 0.7 GB, which `release_unused()` returns;
- **jemalloc (Polars) and mimalloc (genson):** the rest, which `malloc_trim` does not
  touch.

Rather than release that memory after the fact, the change avoids allocating it.

**What each explode copied** (`logs/claims_base.txt`): `explode` ran once over the
entities' properties and once over the properties' statements. Each run made three full
copies of the column:
- **`df.to_arrow()`:** converts Polars' string views to Arrow large strings, a copy
  (0.5 s, +300–550 MB on chunk 0).
- **`combine_chunks()`:** copies the column even when it is a single chunk, as it always
  was here. In a fresh process the copy cost 4–6 s on chunk 0. `perf` puts 56% of it in
  `clear_page_erms`, clearing the huge pages faulted in for the copy. Run again in the
  same process, it took 0.24 s.
- **`pl.DataFrame(table)`:** converts the large strings back to views, another copy
  (0.4 s, +465 MB).

## Change

- **Lengths:** `explode` takes the list lengths from Polars (`list.len()`).
- **The list column:**
  - Without a null or empty list, it is exported with `CompatLevel.newest()`, so the
    string views pass to Arrow without a copy. Its flattened values come back to Polars
    as a Series, again without a copy.
  - With one, it is exported as before, in Arrow's large types, and pyarrow's `take`
    places the null rows. pyarrow 25 has no `take` for string views. A Polars `gather`
    of these nested values instead ran out of memory (15 GB), the slow path the 10-21
    change avoided.
- **The other columns** (only `id` here) are gathered in Polars, with the same repeat
  indices as before.

None of the sampled chunks has an empty claims or statements list (after
`drop_nulls`), so the fallback is there for correctness, not speed.

## Correctness

- **Tests:** `scripts/test_claims_explode.py` passes. It checks `explode` against Polars
  on null and empty lists, sliced, filtered and multi-chunk frames, and no rows. It
  checks `claims_base` against the Polars expressions it replaced, including
  byte-identical partition files, on synthetic claims with empty lists.
- **Real claims files** (`views_vs_master.py`): the partition files and sidecar are
  byte-identical to master's on:
  - chunk 0 and scholar chunk 90;
  - chunk 0 with every 100th claims list made empty, which exercises the fallback.
  The test script's own check against the Polars reference runs out of memory on a real
  file, on master too, which is the bug the 10-21 change fixed.
- **Whole pipeline** (`logs/outputs_identical.txt`): every file a 22-chunk run writes,
  14,330 under `results/` and `audit/`, is byte-identical with and without the change.
  This was 8 chunks per worker, both runs in the same run directory.
- **Lint:** flake8 and `ty check` are clean on claims.py.

## Measured

**`claims_base` alone** (`logs/claims_base.txt`): first run in a fresh process, then two
more in the same process.

| claims of | before: fresh, warm | after: fresh, warm | peak RSS, before → after |
|---|---|---|---|
| chunk 0 | 13.8 s, 5.5–6.9 s | 3.7 s, 3.3–3.5 s | 3.2 GB → 0.9 GB |
| scholar chunk 90 | 9.6 s, 3.6–3.7 s | 2.3 s, 2.1 s | 2.2 GB → 0.7 GB |

**The pool** (`logs/pool_66.txt`, `pool_runs.sh`, `pool_bench.py`):
- **Setup:**
  - `pool.process_chunks` with the real `process_and_partition`, 6 workers, on a 4-core
    VM;
  - chunk i is the (i mod 11)-th of the 11 sample chunks, and partition files are
    deleted once timed;
  - the 5 variants ran in an order shuffled each round, each after `primer.py`, so that
    every run starts from the same memory history.
- **Baseline:** perf/persistent-workers (f871343). At 1 chunk per worker, that is
  master's pool.

| chunks per worker | without the change: wall | with it: wall | summed RSS peak | summed RSS mean |
|---|---|---|---|---|
| 1 | 223.2, 219.2 s | 175.4, 176.8 s | 11.2–12.1 → 8.0–8.2 GB | 7.4 → 5.3–5.4 GB |
| 8 | 192.3, 192.1 s | 160.8, 158.2 s | 12.5–12.6 → 8.2–8.4 GB | 8.8 → 6.0–6.1 GB |
| never replaced | | 163.8, 158.4 s | 8.0–8.2 GB | 6.0 GB |

**Where the time and memory went** (`logs/pool_66_workers.txt`):
- **Partitioning:** the 66 chunks' partitioning took 647–665 s, against 831–887 s
  before. Their `process()` time is unchanged.
- **Per worker:** peak RSS is at most 2.8–2.9 GB, against 4.9 GB before. Mean RSS after
  a chunk is 0.86–0.91 GB, against 1.06–1.12 GB before.
- **After chunk 0:** a worker holds 2.0–2.3 GB, against 2.7–2.8 GB before. pyarrow's pool
  now holds nothing after a chunk (`logs/retained.txt`).
- **Fresh against kept workers:**
  - With the change, 8 chunks per worker beats a process per chunk by 9% (159.5 against
    176.1 s). It holds 0.7 GB more on average, at the same peak.
  - It also uses less memory than a process per chunk did without the change (mean 6.0
    against 7.4 GB, peak 8.3 against 11.6 GB).
  - Never replacing workers is no different from replacing them every 8 chunks.

## Limits

- **One machine:** a 4-core VM where faulting in fresh memory is expensive. Part of the
  fresh-process saving is the copies' page faults, which a machine with cheap faults pays
  less for. The memory saving does not depend on that.
- **What a worker still keeps:** the allocators' memory from its last chunk, up to that
  chunk's peak (2.0–2.3 GB after chunk 0). It is reused by the next chunk, so it raises
  the mean of the summed RSS, not its peak.
