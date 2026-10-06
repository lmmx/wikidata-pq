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

## Not yet done

- The entities table of both sets is still wrong on the Hub. The chunks and the dump have
  been deleted, so the rows have to be made again from the dump.
- A check that partitioning keeps every row of a table that is not exploded (entities: one
  row per entity).
