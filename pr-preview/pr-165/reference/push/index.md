# Push

`push/` uploads partitioned chunks in **groups**: ranges of consecutive chunks whose rows are made into one file per key before upload. `groups.py` keeps the ledger and the group size; `core.py` closes a group.

## Why groups

Uploading each chunk's partitions as they are would put one file per language per chunk on the Hub: about 2,800 files per chunk, millions in all, far past the Hub's guidance of under 100,000 files per repo. A group of hundreds of chunks gives one file per key per group, and compaction later rewrites those into files of about 500 MB.

## Group size

`group_threshold_bytes` sets the partition bytes at which a group closes, so that a set comes to about `GROUP_TARGET_COUNT` (30) groups:

```text
projected = (partition bytes so far / source bytes so far) × total source bytes
threshold = clamp(projected / GROUP_TARGET_COUNT, GROUP_MIN_GB, GROUP_MAX_GB)
```

Source bytes are the measure because chunk sizes vary widely, so a group is a share of the data, not a number of chunks. The ratio is re-estimated from `partition_sizes.jsonl` as chunks finish, so the threshold moves during the run. `GROUP_MAX_GB` (25 GB) bounds local disk; if it applies, there are more groups than the target.

## The ledger

`state/groups.jsonl` has one line per stage a group reaches: `closed`, `merged`, `pushed`, `verified`, `done`. `unfinished_group` returns the one group not yet `done` with its last stage; there is at most one. `open_chunks` lists the chunks at `PARTITION` that are not in that group.

## Closing a group

`close_group` runs the stages after the last recorded one. Each is safe to repeat.

1. **Merge** (`merge_group`) makes one file per table per key, `staging/{table}/{key}/chunks-{first}-{last}.parquet` (`chunk_range_name`: both numbers padded to the digits of the run's last chunk, at least 4), its row count checked against the audit sidecars' sum before any input is deleted.
   - **A table split by key** (`GROUP_PARTITIONED`: labels, descriptions, aliases, links, claims_labels) is partitioned here, once for the group: a chunk's partition step writes only its sidecar (`write_stats_sidecar`) and keeps its processed file. The group's processed files are read in chunk order and split by key by one Polars sink, in batches of up to `GROUP_SINK_ROWS` rows (10M) to bound its memory; a key's batch files, when there are several, are glued. The rows, and their order, are those the chunks' partition files concatenated would have; the row groups are the sink's. For claims_labels, rows repeated across the group's chunks are dropped (`DEDUPLICATE`), so its count must be positive and no more than the sum. Once a table's files are staged, its marker (`staging/.merged/{group}/{table}`) is written and its processed files deleted; a table with a marker is not partitioned again, and one without is redone from its processed files.
   - **An unsplit table** (claims, entities) has one partition file a chunk, whose encoded row groups are copied into the group file (`pq_glue.concat_many`), then the partition files are deleted. A key whose staged file exists and whose partitions are gone was merged before an interruption and is skipped. Emptied key directories are left for the run to remove at its end.
1. **Upload** (`push_group`). Per table: create the repo if needed, create the build branch if needed ([Hub](https://lmmx.github.io/wikidata-pq/pr-preview/pr-165/reference/hub/index.md)), add the table's dataset card, rendered without figures, if the repo has none, and upload the table's staged files for this group to the branch. The chunks move to `PUSH`.
1. **Verify** (`verify_group`). Every staged file must be on the Hub with the same size and sha256 (or git blob hash, for a file not stored in LFS). The chunks move to `POST_CHECK`.
1. **Clean up.** The verified files, recorded with their sha256 in `state/verified.jsonl`, are moved to `compact/src/{table}` for finalise to read rather than download again where kept (`keep_group_files`: `WIKIDATA_KEEP_GROUP_FILES=1` always, `auto` while the disk has `KEEP_GROUP_FILES_MIN_FREE_GB`, 150 GB, free; unset or `0` never, the default; never a release's claims_labels), or else deleted. The staging directory (with the group's markers) is deleted and the chunks move to `COMPLETE`.

The upload is checked against the staged bytes, not by reading the data back, and the staged rows were checked against the sidecars at merge, so each row on the Hub is accounted for.

## On the Hub

After the run, each repo's build branch has `{key}/chunks-{first}-{last}.parquet`, one file per key per group, plus `README.md`. Compaction and the sort replace these files.

Documented against

Commit `6243a2c` (2026-10-04). See [About these docs](https://lmmx.github.io/wikidata-pq/pr-preview/pr-165/about/index.md) to check for changes.

| File                            | SHA-256                                                            |
| ------------------------------- | ------------------------------------------------------------------ |
| `src/wikidata/push/__init__.py` | `c802428454a2174df25399ecb9620ebeb4ef91de81de2ea510998ad747383a1d` |
| `src/wikidata/push/core.py`     | `864a96ebd1f877ca4f37d717f25a47412270aba0457a686d9a8eefa5c4841b0e` |
| `src/wikidata/push/groups.py`   | `29cc2b01bd164cf8dbd0182564bad31877feaf16dc9be825f0bae765c5712caa` |
