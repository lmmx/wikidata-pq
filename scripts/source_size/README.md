# Source size listing

`chunk_totals.csv` is the per-chunk size listing for the current source layout: one file per
chunk, `data/chunk_N.parquet` for N in 0..7448 (7,449 files total, 959.2 GB), fetched from the
HF tree API (`https://huggingface.co/api/datasets/philippesaade/wikidata/tree/main/data`) on
2026-09-27.

- Total: 959.2 GB (893.3 GiB) across 7,449 chunks
- Mean chunk size: 0.129 GB, median: 0.104 GB
- Min: 0.020 GB, max: 1.138 GB (56.4x range)

The `size_gb` column is in GiB (`size_bytes / 1024**3`).

## `old/`

Stats for a previous version of the source dataset, which used a different layout: 113 chunks,
each split into many small part-files (`chunk_N-XXXXX-of-XXXXX.parquet`, 9,687 files total,
94.2 GiB largest chunk). The source dataset has since been restructured to one file per chunk;
these are kept for reference only and no longer reflect the current source.
