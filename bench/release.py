"""The scale results are translated to: the last release whose size is on record
(docs/guide/performance.md), its entities in both sets."""

RELEASE, RELEASE_ENTITIES = "20260928", 121_815_642
# The dump's size: wikidata-20261020-all.json.bz2 (the newest on record), the split's ETA
# scaled to it
RELEASE_DUMP_BYTES = 103_488_697_311
# The same release's gz dump, wikidata-20261020-all.json.gz: the split's ETA from the gz
# dump (pipeline.split_gz) scaled to it
RELEASE_DUMP_GZ_BYTES = 156_603_056_120
