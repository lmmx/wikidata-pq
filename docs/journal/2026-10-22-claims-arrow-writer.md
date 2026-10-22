# 2026-10-22: The claims partition is written with pyarrow

Continues docs/journal/2026-10-22-sidecar-footers.md. Scripts and logs are in
`2026-10-22-claims-arrow-writer/`.

**The change:** `partition_parquet(..., arrow=True)` writes the one unsplit file with
pyarrow's `ParquetWriter` (`write_arrow` in partitioning/core.py). Polars' sink no longer
writes it. `partition_chunk` passes `arrow=True` for claims only; every other table still
goes through Polars' sink.

**The effect, on 66 chunks with 6 workers and 8 chunks per worker:**
- **Time:** 25% less wall (141.6 s to 106.1 s) and 28% less chunk CPU (537 s to 386 s).
- **Memory:**
  - Mean summed RSS is unchanged (6.6 GB), and so is each worker's peak (2.8–2.9 GB).
  - The summed peak is 0.55 GB higher on average (9.3 GB to 9.8 GB).
- **Files:**
  - The claims files hold the same data, the same Arrow schema and the same row groups,
    in different bytes, and are 1.3% smaller.
  - Every other output file is byte-identical.

## Why

**Where the time went** (`logs/profile.txt`): a perf profile of `process_and_partition`
over the 11 sample chunks in one process, on the #22 stack.
- **Partitioning:** 61% of the CPU, of which:
  - 42.4% is Polars' Parquet writer, 30.3% of it in `BufferedDremelIter::next`, which
    computes the repetition and definition levels of nested columns;
  - 8.8% is its nested reader.
- **genson's `process()`:** 37%.
- **The pool is CPU-bound:** 569 of the 600 core-seconds of a 66-chunk run.

**What sets the writer's cost:**
- The claims are the one table with deep nesting. Their references and qualifiers hold
  lists of structs with lists of snaks, about 94 leaves in all.
- On scholarly chunks the claims write runs at 1.3 cores: 3 s for 110,270 rows on s70.
- The frame's layout makes no difference: rechunked, it writes the same bytes in the same
  time (`sink_layout.py`).
- pyarrow writes the same frame with a third to a sixth of the CPU (`writer_cmp.py`).

**Ruled out:**
- Polars' sink with other settings writes other bytes, without removing the level cost.
- Reading the processed claims with pyarrow takes more CPU than Polars does (s70: 2.1 s
  against 1.2 s).

## Change

`write_arrow(lf, dst)` collects the frame, then writes one Parquet file with pyarrow.

**How it writes:**
- **Schema:** the Arrow schema of `df.to_arrow()`, in Arrow's large types. That is the
  schema Polars' sink stores in the file (`ARROW:schema`), so pyarrow readers see the same
  types.
- **Row groups:** 122,880 rows each, as Polars' sink makes them.
- **Compression:** zstd at Arrow's default level. Level 1 matches the sink's sizes; level
  3 made bigger files.
- **Export:** a row group at a time. Each slice is rechunked, then exported with
  `to_arrow()`, then written.
  - Exported unrechunked, a slice of the frame's chunks took 5 times as long with more
    than twice the memory (`logs/claims_write.txt`).
  - Before the rechunk, the 66-chunk run had a summed peak of 12.1–12.7 GB
    (`logs/pool_66.txt`).
  - pyarrow cannot write string views ("Slicing not implemented for StringView"), so
    the export to large types is needed.
- **Atomic:** the file is written to `<stem>.tmp`, which replaces the partition file once
  complete.
- **No rows:** a frame with no rows writes no file, as the sink does.

**Not kept:**
- Batches written with `write_batch` make a row group each, so smaller batches did not
  lower the peak and doubled the CPU.

## Correctness

The contract is logical equivalence, not byte identity.

**Real claims** (`arrow_equiv.py`): s70, s90, chunk 0, and chunk 0 with every 100th claims
list empty. Each file read back:
- equals the sink's in Polars, nested nulls and empty lists included;
- equals the sink's as a pyarrow table;
- has an Arrow schema equal to the sink's, metadata included;
- has the same row groups;
- has the same sidecar rows, but for the file sizes.

**Every output file of a 22-chunk run** (`logs/outputs_parity.txt`, `parity.py`), against
the #22 stack:
- 14,178 of 14,330 files are byte-identical.
- The other 152 hold the same data:
  - the 22 claims partition files, checked as above;
  - the audit sidecars, equal but for the run directory in `path` and the claims files'
    sizes.
- None differs.

**Downstream** (`logs/merge_parity.txt`):
- **Merging:** `push.merge_group` re-reads a group's partition files with Polars and
  sinks them. On the 22 claims files that gives the same rows and row groups whichever
  writer made the inputs, including a mix of the two. Checked row group by row group in
  pyarrow and Polars, with schemas equal including metadata.
- **Compaction:** its schema check passes on a mix of the two writers' files.
- **Not byte-identical:** the merged files themselves differ in bytes. The one merged
  from pyarrow-written files is 0.65% larger: a few plain-encoded string columns
  compress slightly differently.

**Tests:** `scripts/test_claims_explode.py` checks `partition_parquet(arrow=True)` against
the sink:
- on synthetic claims with null and empty lists;
- with no rows (no file either way);
- on a processed claims file, when one is given.

With the writer dropping a row, the check fails.

**Lint:** flake8 is clean on the changed files. `ty check` reports only the two
diagnostics `main.py` already has.

## Measured

**The pool** (`logs/pool_66.txt`, `pool_runs.sh`):
- **Setup:** 66 chunks, 6 workers, 8 chunks per worker, 4-core VM. The #22 stack against
  it plus this change, 3 rounds in an order shuffled each round, each run after
  `primer.py`.

| | wall | chunk CPU | partition CPU | summed RSS peak | summed RSS mean | worker peak |
|---|---|---|---|---|---|---|
| Polars' sink | 143.2, 141.8, 139.8 s | 531–540 s | 335–344 s | 9.0–9.6 GB | 6.6 GB | 2.8–2.9 GB |
| pyarrow | 104.7, 105.9, 107.6 s | 379–392 s | 185–198 s | 9.5–10.1 GB | 6.6 GB | 2.8–2.9 GB |

- **Partitioning:** its CPU falls by 44%. `process()` wall falls too (258 s to 221 s
  summed), because there is less contention for the 4 cores.
- **Memory:**
  - A worker's mean RSS after a chunk is 1.02 GB against 0.92 GB. Freed export copies
    stay with Polars' allocator until the next chunk reuses them.
  - Summed peaks are 0.55 GB higher on average.
- **Sizes** (22 chunks):
  - The claims partition files total 185.3 MB against 187.7 MB.
  - A group's merged claims file is 0.65% larger (see Downstream).

## Limits

- **One file per chunk:** Polars' sink would split a partition whose frame passes 4 GiB in
  memory into several files. `write_arrow` writes one; chunk 0's claims frame is about
  290 MB.
- **The merge still uses Polars:** it re-writes claims with Polars' sink once per group, so
  the level cost remains there. It is outside these runs, where no group closes.
- **The pyarrow requirement:** the writer relies on pyarrow, already a dependency (>=25).
