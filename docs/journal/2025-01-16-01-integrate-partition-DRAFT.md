# 2025-01-16: Integrate Partition

## Current State

- `partition_parquet(by, source_pq, dst_dir, log_dir)` accepts partition column name, source path, destination directory, and audit log directory (`partitioning.py:44-53`)
- `pl.PartitionByKey(dst_dir, by=[by], file_path=fp, finish_callback=cb)` creates partitioner with custom path callback and sidecar callback (`partitioning.py:52`)
- `custom_file_path(ctx, source, ext)` receives `pl.KeyedPartitionContext`, extracts language from `ctx.keys[0].str_value`, returns `{lang}/{source.stem}.parquet` (`partitioning.py:21-27`)
- `sink_sidecar(report, source, log_dir)` selects first 5 columns from partition report, unnests structs, renames `lower_bound`→`min_id` and `upper_bound`→`max_id`, writes to `log_dir/source.name` (`partitioning.py:33-42`)
- `Table` enum defines LABEL, DESC, ALIAS, LINKS, CLAIMS with string values "labels", "descriptions", "aliases", "links", "claims" (`config.py:27-33`)
- `OUTPUT_DIR` set to `Path("results")` (`config.py:22`)
- `process()` writes tables to `results/{table}/{chunk_file}.parquet` via `tbl_pq(tbl)` helper returning `output_dir / tbl / pq_path.name` (`process.py:192-194`)
- Labels, descriptions, aliases normalised with `language` column from map key (`process.py:25-27`, `process.py:38-44`)
- Links normalised with `site` column from sitelinks map key (`process.py:67-71`)
- Claims coalesced with `language` column from property label language matching (`README.md:70-82`)

## Missing

- `AUDIT_DIR` constant not defined in `config.py`
- `main.py` partition step is comment placeholder at line 63
- No loop over `Table` enum calling `partition_parquet()` for each table type
- No mapping of table type to partition column: `{"labels": "language", "descriptions": "language", "aliases": "language", "links": "site", "claims": "language"}`
- Partitioned output structure `results/{table}/language={lang}/{chunk}.parquet` does not exist
- Sidecar audit structure `audit/{table}/{chunk}.parquet` does not exist

## Implementation Elements

- Add `AUDIT_DIR = Path("audit")` to `config.py`
- Add `PARTITION_COLS = {Table.LABEL: "language", Table.DESC: "language", Table.ALIAS: "language", Table.LINKS: "site", Table.CLAIMS: "language"}` to `config.py`
- After process loop in `main.py`, for each `tbl in Table`:
  - `src = OUTPUT_DIR / tbl / pq_path.name`
  - `dst = OUTPUT_DIR / tbl` (partitioner creates `language=X/` subdirs)
  - `log = AUDIT_DIR / tbl`
  - `partition_parquet(by=PARTITION_COLS[tbl], source_pq=src, dst_dir=dst, log_dir=log)`
- Sidecar output columns: `language`, `rows`, `min_id`, `max_id` per `sink_sidecar` transform (`partitioning.py:36-40`)