# 2026-10-06: The entities table kept only properties

Release 20260928 finished finalising its scholarly set, and then the Hub showed
`wikidata-scholar-entities` with 1 row. The card metadata had both sets' entities near empty:

| Set | Entities routed to it (`data/route.jsonl`) | Rows in the entities table |
|---|---|---|
| main | 75,432,640 | 13,929 |
| scholar | 46,383,002 | 1 |

13,929 is about the number of Wikidata properties.

## Cause

`partitioning.transforms.prepare_for_partition` began with
`pl.scan_parquet(table_file).drop_nulls()` for every table. For the other tables, the file
read there is an id and one map or list column (labels, claims, ...), so it drops only
entities with none, which give no rows anyway. An entity row has a field per column
(`ENTITY_SCHEMA`), and `datatype` is null for every item, so each item was dropped and only
properties were kept. The scholarly set has almost no properties, so it kept 1 row.

## Why no check caught it

- Processing's `check_ids` checks each chunk's entities file before partitioning, and that
  file was whole.
- The audit sidecars count the rows written by partitioning, and every later check compares
  against them: the group merge (`push/core.py`), compaction, the sort. They agree with
  each other, so the loss went through every check.
- Nothing compares the partitioned rows with the rows going in. The cards' row count was the
  first place it showed.

## Fix

`prepare_for_partition` returns the entities table before `drop_nulls`. The other tables
are unchanged.

## Repair

- Partitioning now checks that the entities rows it writes are the rows it was given
  (`main.partition_chunk`).
- The other tables were checked against 20260507 from the recorded row counts: the main
  set's grew 0.6 to 2.0%; entities is the only table out of line. `scripts/hub_audit.py`
  checks the same on the Hub, from the Parquet footers.
- The chunks and the dump had been deleted, so `scripts/rebuild_entities.py` reads the
  dump again and makes the entities rows with the pipeline's own functions: `entity_row`,
  `is_scholarly`, the process step's decoding, `prepare_for_partition` and
  `partition_parquet`. Each chunk is checked against the route log (rows, first and last
  id per set), and the rows are merged into the same group files as before. Compaction
  and the sort then run on entities alone, from the reset ledgers.
- `finalise.done` had been written for both sets, so promotion may have run. Card pushes
  went to the release's branch, which promotion deletes; they now go to `main` once the
  branch is gone.
