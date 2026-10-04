# Partitioning

`partitioning/` splits each processed table into one file per language (or site) and
records what it wrote in an audit sidecar.

## Preparing each table

`prepare_for_partition` (`transforms.py`) reads a processed table lazily, drops rows with
nulls, and flattens it to scalar columns plus the partition column:

| Table | Transform | Partition column |
|---|---|---|
| labels, descriptions | explode the `{key, value}` list: `id, language, value` | `language` |
| aliases | as labels, then explode the list of aliases: one row per alias | `language` |
| links | explode, unnest the value, drop the key: `id, site, title` (and `badges`) | `site` |
| claims | one row per statement, the main snak's fields unnested (`claims_base`) | `partition` = `all` |
| claims_labels | already one row per label | `language` |
| entities | unchanged | `partition` = `all` |

`claims_base` (`claims.py`) explodes the claims to one row per statement. A release's
statements have their own `id` and `type`, which are renamed `statement_id` and
`statement_type` so they do not collide with the entity's `id`. Claims are not split by
language, because a statement has no language of its own: its labels are in claims_labels,
which is.

## Writing partitions

`partition_parquet` (`core.py`) sinks the frame with `pl.PartitionBy`, into
`results/{table}/{key}/chunk_{N}.parquet`. The file name keeps the source chunk's name, so
each chunk's partitions are distinct files and several chunks can be partitioned at once.
For the unsplit tables the partition column is not written.

`write_sidecar` then writes `audit/{table}/chunk_{N}.parquet`: one row per partition file,
with its path, row count, size, key, and minimum and maximum id (`ref` for
claims_labels). The sidecars drive the rest of the run:

- the group size comes from their file sizes (`chunk_partition_bytes`);
- merging checks each merged file's rows against their row counts;
- `scripts/release_eta.py` reads their modification times to count processed chunks.

They are kept after the run.

??? info "Documented against"
    Commit `6243a2c` (2026-10-04). See [About these docs](../about.md) to check for changes.

    | File | SHA-256 |
    |---|---|
    | `src/wikidata/partitioning/__init__.py` | `8e1636c11c860b23822bdf8c492db4457d8c3726bbe4025c8440c4ec4ed5ddc1` |
    | `src/wikidata/partitioning/core.py` | `7e2ead148139e947a7762c16e991dfb907343504a96dff283f68eb3e253461df` |
    | `src/wikidata/partitioning/transforms.py` | `495b93f00926d0f6ae22baa3c37bf541c8f9c9525fa5c0f7d26f4682cd54a9f8` |
    | `src/wikidata/partitioning/claims.py` | `990a02e0738e82dd9cb328a6ae7d1ee3439539518153b385360d56b2ba9e37e2` |
