# Compaction

`compact.py` replaces each table's group files on the Hub with files of about
`COMPACT_FILE_BYTES` (500 MiB), the size the Hub recommends for Parquet. The upload left
one file per key per group, most of them small, since most languages have few rows in any
one group.

## Stages

Recorded in `state/compact.jsonl`, one table at a time:

| Stage | Does |
|---|---|
| `downloaded` | the table's group files into `compact/src/{table}` |
| `written` | each key rewritten into `compact/out/{table}/{key}/`, checked, and listed in the table's manifest |
| `committed` | each key's new files added and its group files deleted on the Hub, in batched commits |
| `verified` | the Hub has exactly the new files of every key, by size and hash |
| `done` | the table's files, bytes and rows per key written to the card metadata JSON; the new files moved to `hub/{table}` for the sort, and the group files removed |

## Writing a key

A key's group files, in chunk order, are cut into **runs** of consecutive files, each run
up to `COMPACT_FILE_BYTES` (`_runs`). Each run becomes one output file named
`chunks-{first}-{last}.parquet` by the chunks it covers, so files split only between
groups. The numbers are padded to the digits of the run's last chunk index (at least 4),
as group names are. Group files are read at any width and ordered by chunk number:
release 20260928's groups were named with 4 digits up to chunk 9999 and 5 after. Files are written with pyarrow (`_write_file`):

- zstd level 3, a page index, and content-defined chunking, so the Hub can deduplicate
  unchanged pages between versions;
- row groups of about `COMPACT_ROW_GROUP_BYTES` (128 MiB) **in memory**, estimated from
  the Arrow size per row of the key's largest file. A reader holds a whole row group in
  memory, and claims take several times more memory than their Parquet size.

Record batches are copied as they are, without a cast: casting nested claims structs with
pyarrow corrupted them. Instead, every group file of a key must already have the same
schema.

## Checks

Each output file is read back with Polars, which did not write it, and compared with its
run of group files by `_fingerprint`: the row count and two sums of row hashes, each row
hashed with its position under two seeds. A changed, missing, extra or reordered row
changes a sum. The fingerprint is computed by Polars' streaming engine, so a key is checked
without holding its rows in memory.

claims_labels is deduplicated across the whole key (`DEDUPLICATE`), since a group only
removed duplicates within itself. Its files are checked together against the distinct rows
of its group files, in order of first occurrence.

## Resuming

- `manifest.jsonl` lists each finished key; a key is redone only if its group files
  changed.
- `files.jsonl` lists each output file checked so far, so a restart in the middle of a large
  key (claims has a single key, `all`) reuses the files already written if their size and
  sha256 match. A deduplicated key is checked, and so resumed, as a whole.
- Commits skip keys whose files on the Hub are already their new files. A key's additions
  and deletions are in the same commit, so the Hub never has a key in both layouts.

??? info "Documented against"
    Commit `6243a2c` (2026-10-04). See [About these docs](../about.md) to check for changes.

    | File | SHA-256 |
    |---|---|
    | `src/wikidata/compact.py` | `1eeae26d830e05d1f28aa3d032b95ab5e22ef991cd648b9910f7561aef60f4a8` |
