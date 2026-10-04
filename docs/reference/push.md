# Push

`push/` uploads partitioned chunks in **groups**: ranges of consecutive chunks whose
partition files are merged into one file per key before upload. `groups.py` keeps the
ledger and the group size; `core.py` closes a group.

## Why groups

Uploading each chunk's partitions as they are would put one file per language per chunk on
the Hub: about 2,800 files per chunk, millions in all, far past the Hub's guidance of under
100,000 files per repo. A group of hundreds of chunks gives one file per key per group, and
compaction later rewrites those into files of about 500 MB.

## Group size

`group_threshold_bytes` sets the partition bytes at which a group closes, so that a set
comes to about `GROUP_TARGET_COUNT` (30) groups:

```text
projected = (partition bytes so far / source bytes so far) × total source bytes
threshold = clamp(projected / GROUP_TARGET_COUNT, GROUP_MIN_GB, GROUP_MAX_GB)
```

Source bytes are the measure because chunk sizes vary widely, so a group is a share of the
data, not a number of chunks. The ratio is re-estimated from `partition_sizes.jsonl` as
chunks finish, so the threshold moves during the run. `GROUP_MAX_GB` (25 GB) bounds local
disk; if it applies, there are more groups than the target.

## The ledger

`state/groups.jsonl` has one line per stage a group reaches: `closed`, `merged`, `pushed`,
`verified`, `done`. `unfinished_group` returns the one group not yet `done` with its last
stage; there is at most one. `open_chunks` lists the chunks at `PARTITION` that are not in
that group.

## Closing a group

`close_group` runs the stages after the last recorded one. Each is safe to repeat.

1. **Merge** (`merge_group`). For each table and key, the group's partition files (listed
   in the audit sidecars) are concatenated, streaming, into
   `staging/{table}/{key}/chunks-{first:04d}-{last:04d}.parquet`. The merged row count must
   equal the sidecars' sum. For claims_labels, rows repeated across the group's chunks are
   dropped (`DEDUPLICATE`), so its count must be positive and no more than the sum. Then
   the partition files are deleted. A key whose staged file exists and whose partitions
   are gone was merged before an interruption and is skipped. Emptied key directories are
   left for the run to remove at its end, since another chunk's process may be writing
   into them.
2. **Upload** (`push_group`). Per table: create the repo if needed, create the build
   branch if needed ([Hub](hub.md)), add the table's dataset card, rendered without figures, if the repo has none,
   and upload the table's staged files for this group to the branch. The chunks move to
   `PUSH`.
3. **Verify** (`verify_group`). Every staged file must be on the Hub with the same size
   and sha256 (or git blob hash, for a file not stored in LFS). The chunks move to
   `POST_CHECK`. A file missing or different raises (`... is not on the Hub`,
   `... differs on the Hub`) and stops the run. The group stays at `pushed` with its
   staged files kept, so a rerun verifies it again but does not upload it again.
4. **Clean up.** The staging directory is deleted and the chunks move to `COMPLETE`.

The upload is checked against the staged bytes, not by reading the data back, and the
staged rows were checked against the sidecars at merge, so each row on the Hub is
accounted for.

## On the Hub

After the run, each repo's build branch has `{key}/chunks-{first:04d}-{last:04d}.parquet`,
one file per key per group, plus `README.md`. Compaction and the sort replace these files.

??? info "Documented against"
    Commit `b8ac85a` (2026-10-04). See [About these docs](../about.md) to check for changes.

    | File | SHA-256 |
    |---|---|
    | `src/wikidata/push/__init__.py` | `c802428454a2174df25399ecb9620ebeb4ef91de81de2ea510998ad747383a1d` |
    | `src/wikidata/push/core.py` | `864a96ebd1f877ca4f37d717f25a47412270aba0457a686d9a8eefa5c4841b0e` |
    | `src/wikidata/push/groups.py` | `29cc2b01bd164cf8dbd0182564bad31877feaf16dc9be825f0bae765c5712caa` |
