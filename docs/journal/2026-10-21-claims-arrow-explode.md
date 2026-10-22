# 2026-10-21: The claims explodes run in Arrow

Continues docs/journal/2026-10-21-chunk-read-columns.md. Scripts and logs are in
`2026-10-21-claims-arrow-explode/`.

## Why

Partitioning a chunk costs more than processing it, and nearly all of that cost is the
claims table's two `explode`s in `claims_base` (partitioning/claims.py).

- **What makes it slow:** Polars only explodes a list column without copying when a flag
  on the column says it holds no empty lists, and a column read from Parquet never has
  that flag. Without it, Polars gathers the list's values with indices `0..n`. For a
  nested list it does that one element at a time, cloning each element's arrays and data
  types, then concatenating them.
- **On claims:** a list of properties, each a list of statements, each with lists of
  qualifier and reference snaks. On chunk 0 the partition was killed at the session's
  memory limit (about 14.0 GB) every time. On the largest sampled chunk it took 20.5 s and
  13.4 GB.
- **In Arrow** the same explode is a flatten of the lists and a take of the other
  columns. pyarrow and numpy are already dependencies.

## Change

- **`explode(df, col)`** in partitioning/claims.py does
  `df.explode(col, empty_as_null=True)` through pyarrow:
  - The list array's `flatten()` gives the values, and `take` repeats the other columns'
    rows once per value. `value_lengths()` gives each list's length.
  - The result is built with `pl.DataFrame(table)`, which goes through the same
    `arrow_to_pydf` with the same defaults as `pl.from_arrow(table)`, and is typed as a
    DataFrame.
  - A null or empty list gives one null row in its place, as Polars' `empty_as_null=True`
    does (null lists too, as with Polars' default `keep_nulls=True`).
  - If no list is null or empty, the flattened values are used as they are. Otherwise
    they are taken with a null index for each such list.
- **`claims_base`** collects the claims, explodes `claims` and then `value` with it, and
  leaves the unnests and renames to Polars as before. It returns a LazyFrame as before.
  `prepare_for_partition` stays lazy for the other tables, and its docstring now says
  claims are collected.

## Measured

**Chunks:**
- **c0:** chunk 0 of release 20261012 (10,000 entities, Q31 to Q83751), as in the previous
  entry.
- **r10–r90 and s10–s90:** the routed chunks of 10,000-entity windows at 10%, 30%, 50%,
  70% and 90% through the release's dump, from polars-genson's `sample_chunks.py` (it
  seeks to a bzip2 block). Each window is routed as a release routes it: `r` is the
  general chunk and `s` the scholarly articles' chunk.

**Setup:**
- **Processing:** each chunk was processed once, by `process()` at #17's head
  (`process_only.py`).
- **Partitioning:** both codes partitioned the same processed tables (`partition_one.py`).
  It runs `partition_chunk`'s per-table loop, `prepare_for_partition` then
  `partition_parquet` with `PARTITION_COLS`, for all seven tables, in a fresh process per
  run, as `pool.py` runs a chunk.
- **Rounds:** 3, with base and new alternating order by round (`bench.sh`).
- **Machine and versions:** 4 cores, Polars 2.0.0, pyarrow 25.0.1.
- **Code check:** each run checks that it imported the code it was given. An earlier
  attempt passed a relative path that didn't survive the run's `chdir`, and silently ran
  the installed checkout; those runs were discarded.

**Results** (`logs/summary.md`; median and range of 3 runs; "all tables" is the whole
partition step):

| chunk | claims wall, base → new (s) | claims CPU, base → new (s) | all tables wall, base → new (s) | peak RSS, base → new (MB) |
|---|---|---|---|---|
| c0 | killed, 3 of 3 runs → 13.01 | → 21.0 | → 19.64 | ~14,000 (killed) → 3,472–3,497 |
| r10 | 6.57 → 2.27 | 21.0 → 3.3 | 8.18 → 4.02 | 7,040–7,284 → 805–991 |
| s10 | 5.57 → 3.74 | 15.0 → 4.2 | 5.77 → 3.99 | 4,447–4,698 → 985–1,078 |
| r30 | 1.81 → 1.15 | 5.7 → 1.3 | 3.67 → 2.98 | 2,488–2,553 → 394–404 |
| s30 | 15.06 → 6.38 | 48.7 → 9.7 | 15.36 → 6.83 | 10,502–10,892 → 2,276–2,342 |
| r50 | 3.33 → 1.25 | 10.6 → 1.6 | 4.93 → 2.75 | 4,433–4,544 → 522–623 |
| s50 | 3.60 → 1.38 | 9.8 → 1.7 | 3.87 → 1.54 | 3,598–3,670 → 550–577 |
| r70 | 2.10 → 0.71 | 7.2 → 0.9 | 3.65 → 2.32 | 3,456–3,501 → 388–404 |
| s70 | 9.37 → 5.64 | 26.5 → 6.5 | 9.62 → 5.87 | 7,676–7,932 → 1,492–1,666 |
| r90 | 0.47 → 0.36 | 1.4 → 0.4 | 2.05 → 1.81 | 732–747 → 225–230 |
| s90 | 20.50 → 7.07 | 71.0 → 12.1 | 21.26 → 7.47 | 13,379–13,628 → 2,694–2,793 |

- **Peak memory:** 3.2–9.0 times lower. Chunk 0 now partitions in 3.5 GB; before, it
  was killed at about 14.0 GB in all 3 runs (`logs/oom_kills.txt`).
- **Claims CPU:** 3.5–8.0 times lower. The old code kept 2.7–3.5 cores busy on it.
- **Wall time:** 1.3–3.0 times lower for the claims table and 1.1–2.8 times for the whole
  partition step.
- **What remains in the new claims time:** on chunk 0, the two Arrow explodes take 4.1 and
  3.9 s, and the write about 3.7 s (`steps_c0.py`).

## Equivalence

**Byte comparison** (`compare.py`; `logs/compare.txt`). Every file the partition writes
(partition files and audit sidecars, for all seven tables) was compared by relative path
and sha256: each new run, and base's rounds 2 and 3, against base's round 1.
- **9 of the 10 chunks base can partition:** every file is byte-identical, in every round.
  That is 75–1,086 files per chunk.
- **s90** has the same data in a different page layout:
  - **What matches:** its claims file holds the same frame (`DataFrame.equals` with
    `null_equal=True`, same schema, same row order) in the same row groups (122,880,
    122,880 and 52,912 rows), with equal column statistics.
  - **What differs:** in the last row group only, 37 nested columns under references and
    qualifiers (strings, integers and doubles) are 53–4,255 bytes larger uncompressed and
    2–30% larger compressed. The file is 16,430,550 bytes against 16,311,144.
  - **The sidecar** differs only in that file size.
- **c0 can't be compared whole,** because base can't partition it. Instead (`c0_slices.py`;
  `logs/c0_slices.txt`):
  - Its processed claims were cut into 8 slices of 1,250 entities. Each slice was
    partitioned by base (4.1–8.6 s, peak 4.8–7.2 GB) and by new (1.7–3.4 s, peak
    0.5–0.9 GB). Every slice's partition file and sidecar are byte-identical.
  - The new code's whole-chunk claims partition equals the 8 slices' outputs concatenated
    (619,789 rows). Explode works row by row, so the slices cover the whole chunk.

**Why s90's layout differs** (`claims_bytes.py`, `morsels.py`; `logs/threads_s90.txt`):
- **How Polars cuts pages:** it cuts a column chunk into pages by an estimate of the leaf
  array's bytes, and the estimate depends on how the arrays reaching the writer are cut.
  - With the old code, the streaming engine explodes the scan's morsels, one per
    pipeline, and Polars runs one pipeline per thread. s90 reached the sink in 4 morsels
    (72,423, 77,558, 76,541 and 72,150 rows).
  - The new code passes an in-memory frame, which Polars' in-memory source cuts into
    evenly sized morsels of its own (polars-stream's `in_memory_source.rs`). Re-cutting the
    frame into chunks at the old morsel sizes didn't reproduce the old bytes, as that
    source slices the frame again.
- **Base's own bytes depend on the thread count.** With `POLARS_MAX_THREADS` at 2, 3 and 4,
  base writes three different s90 claims files: 16,370,562, 16,324,697 and 16,311,144
  bytes. The new code writes one file at 3 and 4 threads, and another at 2.
- **Where it shows:** in these chunks, only once a chunk's claims fill more than one row
  group. s30's two row groups came out identical; s90's third didn't.

So on chunks with more than one claims row group, the files' bytes are a function of the
machine's thread count under both codes, and byte identity holds only at a fixed thread
count. Within that, every file is identical except s90's claims page layout.

## Tests

`scripts/test_claims_explode.py` (`logs/test_synth.txt`, `logs/test_real_s30.txt`):
- **`explode` against `DataFrame.explode(col, empty_as_null=True)`:** null and empty lists,
  null elements, nested empty lists, and frames read from Parquet, sliced, filtered, in two
  chunks, with no null or empty lists, with only null and empty lists, and with no rows.
- **`claims_base` against the Polars expression it replaced,** on 3,000 synthetic entities
  read from Parquet (14,784 claims):
  - **Data:** the same frame (values, nulls, order, schema). The input includes empty
    claims lists and empty statement lists.
  - **Files:** byte-identical from `partition_parquet`.
- **With a processed claims file as its argument,** it also checks that the partition
  files are byte-identical. Given s30's processed claims, it passes.
- **It catches mistakes** (`logs/mutations.txt`): an `explode` that drops null and empty
  lists fails the first check, and one that reverses rows fails the `claims_base` check.

## Six workers

`pool6.py` runs a set's chunks through `pool.process_chunks` and
`main.process_and_partition`, as `main.run` does, with `WIKIDATA_WORKERS=6` (the
default). It skips the pull (the chunks are already local) and the upload (`close` does
nothing). Each chunk is processed and partitioned in its own spawned process, as in
production. Every 0.1 s it sums the workers' RSS from /proc.

**The sets** (`pool_bench.sh`):
- **Main:** chunk 0 and r10–r90, six chunks.
- **Scholarly:** s10–s90, plus a second copy of s90 as chunk 91, so that six heavy chunks
  run at once.

**Machine:** 4 cores, so six workers, each with Polars' 4 threads, oversubscribe it.

**Results, 3 rounds each** (`logs/pool_bench.jsonl`):

| set | wall (s) | CPU (s) | peak of the workers' summed RSS (MB) | worker peaks (MB) |
|---|---|---|---|---|
| main | 36.4, 29.3, 30.2 | 78.5, 63.8, 66.2 | 5,428–5,447 | 315–5,456 |
| scholarly | 26.1, 22.5, 23.9 | 97.9, 83.7, 88.5 | 9,910–10,186 | 742–3,823 |

- **Chunk 0 bounds the main set:** it takes 28–35 s and peaks at 5.4 GB. That is above
  the 3.5 GB of the partition alone, because `process()` runs in the same process first
  and the memory it frees stays in the allocator.
- **The other main chunks** finish in 4–11 s.
- **Outputs** (`compare_pool.py`; `logs/compare_pool.txt`): every partition file is
  byte-identical to the single-process run's, and every sidecar equal as a frame, with its
  paths taken from the table directory on. That is 6,629 files and sidecars in each main
  round and 710 in each scholarly round, all 3 rounds.
- **The old code at six workers** (`logs/pool_base.txt`) fails both sets:
  - **Main:** chunk 0 is killed by the kernel's memory limit.
  - **Scholarly:** chunks 30, 70, 90 and 91 are killed.

## Checks

- **flake8** (the Justfile's settings) reports only E303 in claims_labels.py and E305 in
  dump.py, files the change doesn't touch. The new test is clean.
- **ty** (0.0.90) reports 121 diagnostics, as at #17's head, and none in the changed
  files or the test, with no suppressions.
  - **The first version** added 3: `pc.list_value_length` and `pc.list_flatten`, which
    pyarrow.compute's stubs lack, and `pl.from_arrow`, typed as `DataFrame | Series`.
  - **The fix:** the list array's own `value_lengths()` and `flatten()` call those same
    kernels (`_pc().list_value_length(self)`, `_pc().list_flatten(self, ...)` in
    pyarrow's array.pxi). `pl.DataFrame(table)` replaces `pl.from_arrow`.
  - **Checked:** all 11 chunks' partition outputs are byte-identical to the first
    version's (`logs/typed_check.txt`), and the test passes.

## Missing

- **Other thread counts:** the byte comparison is at 4 threads, the setting the benchmark
  used.
- **Source-repo mode:** only release mode was run.
