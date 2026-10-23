# 2026-10-23: Claims written once, one row per statement

Continues docs/journal/2026-10-23-merge-glue.md. Scripts and logs are in
`2026-10-23-claims-stream/`.

## Current State

**The change:**
- `process.normalise_claims_direct` takes the claims from genson as Arrow record batches
  (`normalise_from_parquet(output_format="arrow")`, an Arrow C stream, new in
  polars-genson: its 2026-10-23 arrow-output journal entry), one row per entity, and
  writes them once, laid out as the claims partition: one row per statement.
- `partitioning.claims.statements` lays them out: it flattens both lists in Arrow, so the
  statements' values are not copied, only the entity ids gathered. A table with a null
  id, an empty list, or a null property group or statement list goes through
  `claims_base` itself, which gives those a row of nulls (none of the 11 chunks has one).
- `partition_chunk` places that file as the claims partition with a hard link
  (`partitioning.core.place_unsplit`) and writes its sidecar: the claims are no longer
  read back and rewritten.
- The claims' ids are checked against the chunk's (`check_ids`) on the batches, before
  the file is written: the file keeps no entity without claims.

### Why

Where partitioning's CPU went (`part_profile.py`, the 11 chunks, best of 2 rounds, one
process, `logs/partition_profile.txt`):

| table | read | explode, unnest | write, sidecar | all |
|---|---|---|---|---|
| claims | 17.4 s | 0.5 s | 23.5 s | 41.4 s |
| descriptions | 0.2 s | 0.1 s | 4.7 s | 5.1 s |
| labels | 0.1 s | 0.0 s | 4.8 s | 5.0 s |
| aliases | 0.1 s | 0.1 s | 3.8 s | 4.0 s |
| links | 0.2 s | 0.3 s | 2.9 s | 3.4 s |

- **Explodes:** 1 s in all. The empty lists Polars' explode gathers for
  (`empty_as_null=True`) never reach it: the processed tables hold an empty map as null,
  and `drop_nulls` drops those rows first.
- **Claims:** 70% of partitioning was reading back the nested claims file genson had just
  written and writing it again in another layout. Split by step (`claims_split.py`,
  `logs/claims_split.txt`): Polars' read 19.8 s, the Polars-to-Arrow export 13.2 s,
  pyarrow's write 12.9 s, against 6.6 s for genson's own write of the nested file.
  pyarrow reads it no faster (30.3 s).

### Not kept: Arrow IPC

The first version had genson write the batches to an Arrow IPC file, read back through a
memory map. It took as long as the stream (below), but uncompressed IPC is 15 times the
size of the nested Parquet (1.7 GB for the 11 chunks' claims, about 10 GB written for 66,
`logs/ipc_size.txt`), and its run's disk use peaked 0.8 GB higher.

## Measured

**Machine:** these runs are on a slower VM than the 2026-10-23 merge-glue entry's
(PR #25's head took 208–224 s here, 109–113 s there), with polars-genson built from its
1.2.2 with `output_format="arrow"` added, for all arms. PyPI's 1.2.2 wheel gave the same
time for PR #25's head (222.3 s).

`ab3.sh`, the 66 chunks in groups of 22, 3 rounds, arms in a shuffled order each round
(`logs/ab3_66.txt`; `logs/ab_ipc_66.txt` is an earlier 3-round A/B of IPC alone):

| | PR #25's head | IPC (not kept) | Arrow stream |
|---|---|---|---|
| wall | 208.0–223.9 s | 152.5–159.7 s | 143.7–159.3 s |
| partition CPU | 294–314 s | 86–92 s | 81–91 s |
| process CPU | 385–420 s | 363–379 s | 335–362 s |
| RSS peak (all processes) | 11.0–11.6 GB | 6.8–7.1 GB | 6.3–6.6 GB |
| worker RSS peak | 3.3–3.4 GB | 2.1 GB | 1.7 GB |
| disk use, peak over the start | 830–972 MB | 1,748–1,836 MB | 946–969 MB |
| bytes the disk device wrote | 1,305–1,363 MB | 1,724–1,929 MB | 1,081–1,156 MB |
| claims intermediate files | nested Parquet, ~0.7 GB per 66 chunks | IPC, ~10 GB per 66 chunks | none |
| temporary directories left | 0 | 0 | 0 |

- **Wall:** 30% less (150 s against 214 s, the means).
- **Memory:** the peak is 4.7 GB lower, and a worker's 1.7 GB against 3.3 GB: Polars'
  read of the nested claims was each worker's peak.
- **Disk:** the peak is the run's own partition and staging files, the same as before;
  the device wrote 0.2 GB less, the nested claims files no longer written.

**Outputs** (a kept run of PR #25's head and of this branch, `logs/outputs_parity.txt`):
- **Keys:** all 2,328 table/key pairs have the same rows in the same order, with the same
  Arrow schema and metadata (`keep_parity.py`); no quarantine files in either.
- **Sidecars:** every column of all 462 sidecars is equal but the file size of 42 of the
  66 claims files (`sidecar_cmp.py`): the same rows, in Parquet a few hundred bytes
  apart, as pyarrow encodes the batches genson hands over, not those of Polars' export.

**Tests:** `scripts/test_claims_explode.py` checks `statements` against `claims_base` on
synthetic claims, with null and empty lists (through `claims_base`) and without (its
Arrow flatten, on one batch, several, and a slice), and the placed claims file against
`partition_parquet(arrow=True)`'s: byte-identical, with the same sidecar rows.

**Dependency:** this needs a polars-genson release with `output_format="arrow"`
(lmmx/polars-genson, branch `feat/arrow-output`).
