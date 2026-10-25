"""The scale results are translated to: the last release whose size is on record
(docs/guide/performance.md), its entities in both sets."""

RELEASE, RELEASE_ENTITIES = "20260928", 121_815_642
# The dump's size: wikidata-20261020-all.json.bz2 (the newest on record), the split's ETA
# scaled to it
RELEASE_DUMP_BYTES = 103_488_697_311
# The same release's gz dump, wikidata-20261020-all.json.gz: the split's ETA from the gz
# dump (pipeline.split_gz) scaled to it
RELEASE_DUMP_GZ_BYTES = 156_603_056_120
# The release's main-set claims key as compacted (20260928; docs/journal/2026-10-05-sort-speed.md):
# its bytes in its compacted files, one a group (each group's claims file larger than
# COMPACT_FILE_BYTES), and the release's COMPACT_FILE_BYTES, the size of its part files.
# Measured, and pinned by its 667 buckets (one a 64 MiB of source, rounded up: between
# 44.69 and 44.76 GB). It equals the main set's chunk files' 44.7 GB (below) by the
# pipeline's own ratio, not by a copying slip: each of the bench's 11 chunk files'
# claims partition is 0.91-1.11 times its chunk file (CHANGELOG 10.0.0)
RELEASE_CLAIMS_BYTES, RELEASE_CLAIMS_FILES = 44_700_000_000, 30
RELEASE_COMPACT_FILE_BYTES = 500 * 1024**2

# The release's two sets, by entities (docs/guide/performance.md)
RELEASE_MAIN_ENTITIES, RELEASE_SCHOLAR_ENTITIES = 75_432_640, 46_383_002
# Their chunk files (route-release's, Parquet at zstd 9), in bytes: measured (20260928,
# data/manifest.jsonl; docs/journal/2026-10-03-release-20260928-run.md), before any claims
# were compacted. The scholarly set's claims, which nothing on record measures, are
# modelled from them (pipeline.claims_model)
RELEASE_SOURCE_BYTES = {"main": 44_700_000_000, "scholar": 49_500_000_000}
# And their count (measured, likewise): each set's pipeline (process-wikidata) takes them
# CHUNKS_PER_WORKER chunks a worker process (src/wikidata/pool.py), so its workers' starts
# go with these, not with entities (pipeline.starts_model)
RELEASE_CHUNKS = {"main": 12_182, "scholar": 10_853}
# The main set's claims key's bucket path (20260928; docs/journal/2026-10-05-sort-speed.md),
# measured: 667 buckets of 64 MiB of source (SORT_BUCKET_BYTES), 86 part files; its rows
# are modelled (CHANGELOG 7.4.0: about 780 million; the philippesaade build of the
# 2026-05-07 dump's main set, measured, has 774,255,243, docs/dataset_cards_metadata.json)
RELEASE_CLAIMS_BUCKETS, RELEASE_CLAIMS_PARTS = 667, 86
RELEASE_CLAIMS_ROWS = 780_000_000
# The claims keys' workers in a release's finalise, one a job (parallel.in_parallel,
# tasks_per_child=1): no compaction job (each group file is a compacted file, linked:
# CHANGELOG 8.0.0), a bucketing job a compacted file, a bucket sort a bucket, a packing job
# a part file. The main set's are measured (30 + 667 + 86); the scholarly set's are
# modelled: its group files about as many (about 30 upload groups a set,
# docs/journal/2026-10-03-release-20260928-run.md), its buckets and part files the main
# set's in proportion to its entities
RELEASE_CLAIMS_WORKERS = {
    "main": RELEASE_CLAIMS_FILES + RELEASE_CLAIMS_BUCKETS + RELEASE_CLAIMS_PARTS,
    "scholar": RELEASE_CLAIMS_FILES + round((RELEASE_CLAIMS_BUCKETS + RELEASE_CLAIMS_PARTS)
                                            * RELEASE_SCHOLAR_ENTITIES / RELEASE_MAIN_ENTITIES),
}
# A release's finalise workers: FINALISE_WORKERS' default (src/wikidata/config.py)
RELEASE_FINALISE_WORKERS = 6
# A release's upload groups a set (GROUP_TARGET_COUNT, src/wikidata/config.py: the group
# size adapts so a set comes to about 30): the main set's 30 measured (its claims key's 30
# group files, above), the scholarly set's taken as the same target. Every key of a table
# but claims has a group file in each group its rows reach (push_group), so a release's
# group files of the other tables are at most their keys times these (pipeline.files_model).
# Measured for one main set (docs/journal/2026-09-29-hub-layout-and-compaction.md: 31
# groups): labels, descriptions, aliases and links 71,718 group files for 2,757 keys, 26 a
# key
RELEASE_GROUPS = {"main": RELEASE_CLAIMS_FILES, "scholar": 30}
# The other tables' rows in a release's main set (pipeline.tables_model): the 20260507
# build's main set's (73,769,737 entities; docs/dataset_cards_metadata.json, from the Hub's
# footers: measured), grown by RELEASE_MAIN_TABLES_GROWTH. 20260928's main-set tables were
# 0.6-2.0% larger than 20260507's, by the recorded row counts
# (docs/journal/2026-10-06-entities-dropped.md); the upper end is taken. The scholarly
# set's are not on record, and are modelled from the sample's rows a scholarly entity.
# The entities table has a row an entity (RELEASE_ENTITIES, measured)
RELEASE_MAIN_TABLE_ROWS_20260507 = {"labels": 642_354_157, "descriptions": 1_550_918_676,
                                    "aliases": 171_954_210, "links": 98_773_816}
RELEASE_MAIN_TABLES_GROWTH = 1.02
RELEASE_MAIN_TABLE_ROWS = {t: round(n * RELEASE_MAIN_TABLES_GROWTH)
                           for t, n in RELEASE_MAIN_TABLE_ROWS_20260507.items()}
