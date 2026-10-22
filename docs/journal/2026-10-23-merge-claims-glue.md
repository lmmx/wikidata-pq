# 2026-10-23: Group merges copy the claims' row groups (pq-glue)

Continues docs/journal/2026-10-22-claims-arrow-writer.md. Scripts and logs are in
`2026-10-23-merge-claims-glue/`.

## Current State

**The change:**
- `push.merge_group` merges claims with `pq_glue.concat`, which copies the partition
  files' encoded row groups into the group file. Polars' sink no longer decodes and
  re-writes them.
- Every other table still merges with Polars' sink, and claims_labels still drops repeated
  rows with it.
- pq-glue ([lmmx/pq-glue](https://github.com/lmmx/pq-glue)) is a new dependency.

### Why the merge

The earlier benchmarks (`pool_bench.py`) deleted each chunk's partition files once
partitioned, so groups never merged. `pipeline_bench.py` runs the pool as `run` does:
- **Groups:** groups of 22 chunks close in the background through `merge_group`, as
  `close_group` does it.
- **Skipped:** the upload and its verification.

With the merge in, master took 180–185 s for the 66 chunks, against 107 s without it.
- **Merge CPU:** the merges used 270 s of CPU, against 385 s for processing and
  partitioning every chunk.
- **Contention:** they slowed the chunks, which finished at 120 s.
- **Tail:** the last group's merge took 60 s after the last chunk.

**Where the merge's time went** (`merge_bench.py`, the 66 chunks' partition files merged
as one group), for each table:

| table | wall | CPU | files in |
|---|---|---|---|
| claims | 60–62 s | 236–241 s | 66 |
| labels | 4.0–4.3 s | 7.0–7.3 s | 13,062 |
| descriptions | 3.9–4.4 s | 7.1–7.4 s | 11,604 |
| links | 4.2–4.5 s | 6.5 s | 7,272 |
| aliases | 2.9 s | 4.9 s | 10,458 |
| entities | 0.2 s | 0.4 s | 66 |

The claims merge is Polars' sink on 93 nested leaf columns, the writer #23 replaced in
partitioning.

**What else was tried** (`claims_merge_proto.py`):
- **pyarrow:** writing the merge with pyarrow (row groups re-encoded, read ahead in a
  thread) took 56 s and 74 s of CPU. In the pipeline that gave 171–179 s: the CPU went
  down, but the last group's merge was still single-threaded.
- **Copying encoded row groups:** this took 0.6 s.

### Equivalence

The group file has the rows of a Polars merge of the same files, in the same order:
- **Rows:** row count and the two sums of position-seeded row hashes (`compact._fingerprint`).
- **Schema:** the same Arrow schema, metadata included.
- **Row groups:** the partition files' row groups, each with its source's statistics.

**Real claims:** on the 66 chunks' claims files, all 10,602 column chunks' statistics
equal their sources' as pyarrow reads them, including 378 DOUBLE min/max. That relies on
pq-glue restoring the inputs' column orders (pq-glue's journal, 2026-10-23).

**Tests:** `scripts/test_merge.py` checks these on synthetic chunks. It also checks
that labels still merge with Polars, and that a claims file with another schema fails the
merge, leaving no group file and keeping the partition files.

**Row groups and compaction:**
- **Sizes:** the group file's row groups are the partition files', so a group of small
  chunks has many small row groups (Polars' sink re-cut them to 122,880 rows).
- **Compaction:** it reads group files a row group at a time and writes its own row
  groups, so the Hub's files are unchanged.

## Measured

`pipeline_bench.py` on the 66 chunks, groups of 22, three runs interleaved with master
(`ab.sh`, `logs/pipeline_66.txt`):

| | master | this branch |
|---|---|---|
| wall | 182.9–185.6 s | 131.7–134.8 s |
| last chunk done | 106.6–120.4 s | 103.7–105.3 s |
| merge CPU | 270–275 s | 46–48 s |
| chunk CPU (process + partition) | 363–371 s | 369–371 s |
| RSS peak | 10.0–10.3 GB | 9.3–10.5 GB |
| worker RSS peak | 2.8–2.9 GB | 2.8–2.9 GB |

- **Wall:** 28% less. The last group's merge now ends 11 s after the last chunk, against
  60 s.
- **Merge CPU:** 83% less. What is left is the other tables' Polars merges.
- **Memory:** the peak is unchanged within run-to-run spread. The mean RSS is higher
  (5.3 GB against 4.4–4.6 GB) because the run is shorter, so more of it is spent with
  all workers busy.

**Outputs** (`keep_parity.py`, `logs/outputs_parity.txt`): one run of each kept its
staged group files. Their groups closed at different chunks (master's second group took
chunks 22–65, this branch's 22–54 and 55–65), so they are compared per table and
partition key: the key's group files read in order.
- **Keys:** the same 2,328 table/key pairs in both, and none differ in rows, row order,
  Arrow schema or metadata.
- **Quarantine:** no files in either.
- **Audit sidecars:** all 462 are equal but for the run directory in their paths.
