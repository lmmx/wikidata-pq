# 2026-10-24: Claims written with genson's Parquet writer

Continues docs/journal/2026-10-23-claims-stream.md.

## Current State

- `partitioning.core.write_table`, which writes each chunk's claims laid out as their
  partition (one row per statement), writes with `polars_genson.write_parquet` in place
  of pyarrow's `ParquetWriter`: genson's own writer, which encodes each row group's leaf
  columns in parallel, given the Arrow table through the Arrow PyCapsule interface.
- The published schema and data are unchanged: references and qualifiers stay inline.

## Why

**Where `process()` went** (the 11 benchmark chunks in one process, its step functions
timed, the warm second pass; 15.9 s wall, 29.4 s CPU):

| step | wall | share | cores |
|---|---|---|---|
| claims: pyarrow `write_table` | 8.58 s | 54% | 0.99 |
| claims: genson (decode, infer, normalise) | 4.72 s | 30% | 3.36 |
| labels, descriptions, aliases (genson) | 0.84 s | 5% | 1.9 |
| sitelinks | 0.66 s | 4% | 3.2 |
| the rest | ~1.1 s | 7% | |

**Inside the write** (py-spy, native): nested levels (building and RLE encoding) about a
third, string dictionaries a sixth, zstd a sixth, string statistics 7%. By top-level
column, written alone: `references` 56% (31 leaves), `qualifiers` 22.5% (29 leaves),
`datavalue` 10%, every other column under 5%.

**Ruled out:**
- pyarrow's settings: statistics on `id` alone (the sidecars read no others) 9% faster;
  no dictionaries no faster and 5% larger; zstd level 1 the same.
- Writing references once each, by their content hash (1,077,924 occurrences, 169,287
  distinct): it halves the write, but deduplicating in pyarrow costs most of that back
  (2.85 s), and a table of hashes in place of inline references makes every reader
  join to see a statement's provenance. The published schema keeps them inline.

## Measured

The 11 chunks' statements tables (1,562,029 rows), captured as `process()` hands them to
`write_table`, written by each writer:

| writer | wall | CPU | size |
|---|---|---|---|
| pyarrow `ParquetWriter`, zstd | 8.52 s | ~8.5 s | 97.2 MB |
| genson, 1 thread | 6.35 s | ~6.4 s | 92.8 MB |
| genson, 4 threads | 2.25 s | ~5.9 s | 92.8 MB |

## Correctness

For each of the 11 files, genson's against pyarrow's: the same Arrow schema (metadata
aside), the same rows, the same Polars schema, the same row group counts, and the same
`id` min, max and null count in every row group (`file_stats` reads these for the
sidecars).

## Missing

- The 66-chunk pool: the CI benchmark's full tier.
