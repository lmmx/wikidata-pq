"""Configuration constants for the wikidata processing pipeline."""

import os
from enum import StrEnum
from pathlib import Path

# HuggingFace repository identifier
REPO_ID = "philippesaade/wikidata"
REMOTE_REPO_PATH = "data"

# A release: the date of an official Wikidata JSON dump (dumps.wikimedia.org/wikidatawiki/
# entities/{release}/), e.g. 20260928, chosen by WIKIDATA_RELEASE. A release's working
# directories are under releases/{release}/ and its card figures under docs/releases/
# {release}/. Unset, the source is the philippesaade/wikidata Parquet copy (the dump of
# 2026-05-07 without scholarly articles) and the directories are those of that build.
RELEASE = os.environ.get("WIKIDATA_RELEASE") or None
RELEASES_DIR = Path("releases")
# A release is two sets of tables: its scholarly works (see scholarly.py), chosen by
# WIKIDATA_SCHOLAR=1, published as `wikidata-scholar-{tbl}` from releases/{release}-scholar/,
# and everything else, as `wikidata-{tbl}` from releases/{release}/. `route-release` moves
# the scholarly works out of the release's chunks into the scholarly set's.
SCHOLAR = os.environ.get("WIKIDATA_SCHOLAR") == "1"
if SCHOLAR and not RELEASE:
    raise SystemExit("WIKIDATA_SCHOLAR=1 needs WIKIDATA_RELEASE: the scholarly set is a release's")


def set_name(scholar: bool) -> str:
    """A set's directory name under releases/ and docs/releases/."""
    return f"{RELEASE}-scholar" if scholar else str(RELEASE)


WORK_DIR = RELEASES_DIR / set_name(SCHOLAR) if RELEASE else Path(".")
# The release's other set (its labels name the refs of this set's claims_labels)
OTHER_WORK_DIR = RELEASES_DIR / set_name(not SCHOLAR) if RELEASE else None

# Directory names
STATE_DIR = WORK_DIR / "state"
ROOT_DATA_DIR = WORK_DIR / "data"
OUTPUT_DIR = WORK_DIR / "results"
# A release's dump file, as downloaded (see dump.py)
DUMP_DIR = WORK_DIR / "dump"
# The dump's compression, chosen by WIKIDATA_DUMP_FORMAT: bz2 (the default) or gz, which is
# about 1.5 times the download and decompresses in a small part of bz2's CPU time (see
# dump.DECODERS). The chunks are the same from either.
DUMP_FORMAT = os.environ.get("WIKIDATA_DUMP_FORMAT") or "bz2"
if DUMP_FORMAT not in ("bz2", "gz"):
    raise SystemExit(f"WIKIDATA_DUMP_FORMAT is bz2 or gz, not {DUMP_FORMAT!r}")

# Sidecar audit files are written during partitioning storing row counts and min/max
# entity IDs for each language subset of each source file. Post-check reads these to
# verify uploaded files match what was partitioned locally.
AUDIT_DIR = WORK_DIR / "audit"

# Claims snaks on deleted properties (see process.QUARANTINE_FIELDS) are pruned from the
# claims and kept here instead, one file per source file, never uploaded or deleted.
QUARANTINE_DIR = WORK_DIR / "quarantine"

# Delete local files once they are no longer needed: sources once processed, processed
# tables once partitioned, partitions once merged for upload, staging once verified.
# Everything deleted can be regenerated from the source repo.
CLEAN_UP_LOCAL = True

# Source files are one per chunk, named chunk_{N}.parquet (state files chunk_{N}.jsonl)
CHUNK_RE = r"chunk_(\d+)\."


def chunk_glob(chunk_idx: int | None = None) -> str:
    """Glob for the source file of one chunk, or of every chunk if `chunk_idx` is None."""
    return f"chunk_{'*' if chunk_idx is None else chunk_idx}.parquet"


# Table types
class Table(StrEnum):
    LABEL = "labels"
    DESC = "descriptions"
    ALIAS = "aliases"
    LINKS = "links"
    CLAIMS = "claims"
    # Label maps extracted from claims: one row per (field, ref, language, label), field
    # being labels/property-labels/unit-labels, ref the id/property/unit they belong to
    CLAIMS_LABELS = "claims_labels"
    # A release's entities' own fields (type, ns, title, pageid, lastrevid, modified), one
    # row per entity; the philippesaade copy had none of them
    if RELEASE:
        ENTITIES = "entities"


# Claims are not split: a claim has no language of its own, and the labels it refers to
# are in claims_labels, which is split by language. Claims rows get UNSPLIT_COL set to
# UNSPLIT_KEY so they take the same partition, merge and upload path as the other tables
# (one file per group); the column is not written to the files.
UNSPLIT_COL = "partition"
UNSPLIT_KEY = "all"

# Maps each table type to its partition column. Labels, descriptions, aliases and
# claims_labels are partitioned by language because their rows have a language code
# from the multilingual map normalisation. Links is partitioned by site because
# sitelinks use site codes like enwiki, frwiki rather than bare language codes.
PARTITION_COLS = {
    Table.LABEL: "language",
    Table.DESC: "language",
    Table.ALIAS: "language",
    Table.LINKS: "site",
    Table.CLAIMS: UNSPLIT_COL,
    Table.CLAIMS_LABELS: "language",
}
if RELEASE:
    PARTITION_COLS[Table.ENTITIES] = UNSPLIT_COL

HF_USER = "permutans"
# Whether target repos are created private (free accounts get 100GB private storage)
HF_REPO_PRIVATE = False

# Grouped upload (see DESIGN.md, 4. Push). Partitioned chunks are merged into one file per
# language per group; the group size adapts so the dataset comes to about
# GROUP_TARGET_COUNT groups, keeping each repo well under the Hub's 100k file guidance.
STAGING_DIR = WORK_DIR / "staging"
GROUP_TARGET_COUNT = 30
# Bounds on a group's partition bytes: MAX bounds local disk (staging needs about as
# much again), MIN avoids tiny groups early on
GROUP_MIN_GB = 1.0
GROUP_MAX_GB = 25.0
# A table split by key (labels, descriptions, aliases, links, claims_labels) is
# partitioned once a group, at its close, from its chunks' processed files (see
# push.core.merge_group), by one sink a batch of the group's chunks holding up to this
# many of the table's rows: the sink holds about its batch's rows in memory (~50 MB a
# million label rows). A group of more batches has its batches' files glued per key.
# Chosen by WIKIDATA_GROUP_SINK_ROWS
GROUP_SINK_ROWS = int(os.environ.get("WIKIDATA_GROUP_SINK_ROWS") or 10_000_000)

# Chunks processed at once, each in its own process (see pool.py), chosen by
# WIKIDATA_WORKERS; a group uploads in the background while they run
CHUNK_WORKERS = int(os.environ.get("WIKIDATA_WORKERS") or 6)
# Chunks a worker process takes before it is replaced, chosen by WIKIDATA_CHUNKS_PER_WORKER
# (1: a fresh process per chunk). A worker keeps its last chunk's memory until the next
# one reuses it, so more per worker trades memory for the imports and page faults of
# fresh processes (docs/journal/2026-10-22-persistent-workers.md)
CHUNKS_PER_WORKER = int(os.environ.get("WIKIDATA_CHUNKS_PER_WORKER") or 32)
# Jobs of finalise run at once, each in its own process (see parallel.py), chosen by
# WIKIDATA_FINALISE_WORKERS: claims files read for claims_labels' refs, files compacted,
# keys sorted (larger keys FINALISE_LARGE_WORKERS at once), source files bucketed, buckets
# sorted, part files packed, claims_labels languages written
FINALISE_WORKERS = int(os.environ.get("WIKIDATA_FINALISE_WORKERS") or 6)
FINALISE_LARGE_WORKERS = 2
# The memory finalise's bucket sorts may take at once, in GB, chosen by
# WIKIDATA_FINALISE_MEMORY_GB; unset, the memory available when each step starts
# (see parallel.memory_budget)
FINALISE_MEMORY_BYTES = int(float(os.environ.get("WIKIDATA_FINALISE_MEMORY_GB") or 0) * 1e9)

REPO_PREFIX = "wikidata-scholar-" if SCHOLAR else "wikidata-"
REPO_TARGET = "{hf_user}/" + REPO_PREFIX + "{tbl}"
# A release is uploaded, compacted and sorted on a branch of each repo (see hub.py), so the
# repos' main branch keeps the previous release until `promote-release`; None is main
HUB_REVISION = f"build-{RELEASE}" if RELEASE else None

# Dataset card (README.md) templates for each table's Hub repo (see cards.py), rendered
# to RENDERED_CARDS_DIR and pushed by `finalise` where they differ from the repo's card
DATASET_CARDS_DIR = Path(__file__).resolve().parents[2] / "docs" / "dataset_cards"
# A release's cards (official dumps: every field, the entities table, the release's date)
# (and the scholarly set's, about its scholarly works)
CARD_TEMPLATES_DIR = (
    DATASET_CARDS_DIR / ("scholar" if SCHOLAR else "dump") if RELEASE else DATASET_CARDS_DIR
)
RENDERED_CARDS_DIR = (
    DATASET_CARDS_DIR.parent / "releases" / set_name(SCHOLAR) / "rendered"
    if RELEASE
    else DATASET_CARDS_DIR / "rendered"
)
# Files, bytes and rows per partition key of each table on the Hub, written by compaction
# and rewritten by the sort
_CARD_FIGURES_DIR = (
    DATASET_CARDS_DIR.parent / "releases" / set_name(SCHOLAR)
    if RELEASE
    else DATASET_CARDS_DIR.parent
)
DATASET_CARDS_METADATA = _CARD_FIGURES_DIR / "dataset_cards_metadata.json"
# Coverage figures of the language-split tables, from the local copy (see card_stats.py)
DATASET_CARDS_STATS = _CARD_FIGURES_DIR / "dataset_cards_stats.json"

# Compaction (see compact.py), once every group is uploaded: each key's group files are
# rewritten into files of about COMPACT_FILE_BYTES, split only between groups, with row
# groups of about COMPACT_ROW_GROUP_BYTES of uncompressed Arrow data (the Hub's Parquet
# guidance is ~500 MB files and 100-300 MB row groups). The group files are downloaded
# to COMPACT_DIR/src, the rewritten files written to COMPACT_DIR/out, one table at a time.
COMPACT_DIR = WORK_DIR / "compact"
COMPACT_FILE_BYTES = 500 * 1024**2
# (WIKIDATA_COMPACT_FILE_BYTES: set only by the benchmark, for its finalise run on three
# times the rows, so its claims key has a release's files for its bytes; bench/CHANGELOG.md 8.0.0)
COMPACT_FILE_BYTES = int(os.environ.get("WIKIDATA_COMPACT_FILE_BYTES") or COMPACT_FILE_BYTES)
COMPACT_ROW_GROUP_BYTES = 128 * 1024**2
# Keys are committed in batches, each key's new files and deletions in the same commit
COMPACT_COMMIT_MAX_ADDS = 50
COMPACT_COMMIT_MAX_OPS = 2000
COMPACT_DOWNLOAD_WORKERS = 32

# Local copy of the finalised Hub repos, one directory per table (download-wikidata)
HUB_COPY_DIR = WORK_DIR / "hub"

# Sorting (see sort_by_id.py), after compaction: each key's rows are sorted by its sort
# column (string order, stable) across all its files, from the local copy of the Hub.
# How a key is sorted and laid out is planned from its data alone, never from how its
# files happen to be encoded (sort_by_id.plan_key; docs/journal/2026-10-25-decoupled-
# thresholds.md): from its rows and its decoded bytes, estimated from its files' footers
# as each leaf column's values (the same whichever writer or encoding wrote them) at a
# nominal width each, SORT_STRING_BYTES a string (a 4-byte offset and 12 bytes of value).
# A key over SORT_IN_MEMORY_BYTES so estimated is sorted through id-range buckets of about
# SORT_BUCKET_BYTES so estimated each, working files under SORT_DIR. Before 2026-10-25,
# both were bytes of the source files (2 GiB, 64 MiB): the bucket size is as many buckets
# for the claims as 64 MiB of their files made (27 decoded bytes, so estimated, a byte of
# the claims' group files, on the benchmark's sample); the threshold keeps a release's
# keys where they were, its claims through buckets and its other keys in memory (its
# largest, each set's entities, about 7.3 and 4.5 GiB so estimated).
SORT_DIR = COMPACT_DIR / "sort"
SORT_STRING_BYTES = 16
SORT_IN_MEMORY_BYTES = 8 * 1024**3
SORT_BUCKET_BYTES = 1792 * 1024**2
# A bucketed key's buckets are cut between cells, not ids: an id's cell is its first
# SORT_CELL_CHARS characters (sort_by_id.cell), so a thousand numbers of a 9-digit id
# (about 35k claims rows at a release, a bucket about 1.2M), and a shorter id its own
# cell: a partition of the ids fixed before any is written
SORT_CELL_CHARS = 7
# A key's part files: as many as its estimated Hub bytes fill COMPACT_FILE_BYTES, its
# decoded bytes (estimated, as above) times its table's Hub bytes a decoded byte. These
# are measured (2026-10-25), on the benchmark's sample (its quick tier's group files,
# chunks c0, r50, s30 and s90 of release 20261012), as the sort wrote each table: its
# part files' bytes over its keys' decoded bytes so estimated. The claims, by set: the
# main set's rows (c0, r50: 665,335 rows, 47.5 MB written for 992 MB) and the scholarly
# set's (s30, s90: 509,777 rows, 26.9 MB for 986 MB), sorted and written apart, as their
# files compress so differently (the whole sample's, 0.038, would make the main set's
# part files 1.25 and the scholarly set's 0.71 times COMPACT_FILE_BYTES). Every other
# table's from both sets' rows together; claims_labels', not in the sample, taken as
# labels'. A part count goes with its factor: 10% more Hub bytes a decoded byte is 10%
# more part files (at a release, modelled: the main set's claims 88 against today's 86,
# about 79 to 97 within ±10%), and only a key of more than one part file has its count
# changed by it (at a release, the claims, entities and labels/en).
SORT_HUB_BYTES_PER_DECODED = {
    "labels": 0.31,  # 9.28 MB written for 29.7 MB
    "descriptions": 0.13,  # 8.47 MB for 64.8 MB
    "aliases": 0.32,  # 2.71 MB for 8.4 MB
    "links": 0.24,  # 6.46 MB for 27.0 MB
    "claims": 0.027 if SCHOLAR else 0.048,
    "claims_labels": 0.31,
    "entities": 0.17,  # 0.58 MB for 3.5 MB
}
# Bucketing buffers each bucket's rows to about this much Arrow memory before writing
# them as a row group (one bucketing worker holds about this much per bucket)
SORT_BUCKET_WRITE_BYTES = 4 * 1024**2

# Prefetch (background download) settings
PREFETCH_ENABLED = True
# “fill up to” this much source data locally
PREFETCH_BUDGET_GB = 60.0
# Never go more than N chunks ahead
PREFETCH_MAX_AHEAD = 60
# Skip prefetch if disk tighter than this
PREFETCH_MIN_FREE_GB = 100.0
# Concurrent chunk downloads within the prefetch worker.
PREFETCH_CONCURRENCY = 1
