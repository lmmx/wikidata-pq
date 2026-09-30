"""Configuration constants for the wikidata processing pipeline."""

from enum import StrEnum
from pathlib import Path

# HuggingFace repository identifier
REPO_ID = "philippesaade/wikidata"
REMOTE_REPO_PATH = "data"

# Directory names
STATE_DIR = Path("state")
ROOT_DATA_DIR = Path("data")
OUTPUT_DIR = Path("results")

# Sidecar audit files are written during partitioning storing row counts and min/max
# entity IDs for each language subset of each source file. Post-check reads these to
# verify uploaded files match what was partitioned locally.
AUDIT_DIR = Path("audit")

# Claims snaks on deleted properties (see process.QUARANTINE_FIELDS) are pruned from the
# claims and kept here instead, one file per source file, never uploaded or deleted.
QUARANTINE_DIR = Path("quarantine")

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

HF_USER = "permutans"
# Whether target repos are created private (free accounts get 100GB private storage)
HF_REPO_PRIVATE = False

# Grouped upload (see DESIGN.md, 4. Push). Partitioned chunks are merged into one file per
# language per group; the group size adapts so the dataset comes to about
# GROUP_TARGET_COUNT groups, keeping each repo well under the Hub's 100k file guidance.
STAGING_DIR = Path("staging")
GROUP_TARGET_COUNT = 30
# Bounds on a group's partition bytes: MAX bounds local disk (staging needs about as
# much again), MIN avoids tiny groups early on
GROUP_MIN_GB = 1.0
GROUP_MAX_GB = 25.0

REPO_TARGET = "{hf_user}/wikidata-{tbl}"

# Dataset card (README.md) templates for each table's Hub repo (see cards.py), rendered
# to RENDERED_CARDS_DIR and pushed by `finalise` where they differ from the repo's card
DATASET_CARDS_DIR = Path(__file__).resolve().parents[2] / "docs" / "dataset_cards"
RENDERED_CARDS_DIR = DATASET_CARDS_DIR / "rendered"
# Files, bytes and rows per partition key of each table on the Hub, written by compaction
# and rewritten by the sort
DATASET_CARDS_METADATA = DATASET_CARDS_DIR.parent / "dataset_cards_metadata.json"
# Coverage figures of the language-split tables, from the local copy (see card_stats.py)
DATASET_CARDS_STATS = DATASET_CARDS_DIR.parent / "dataset_cards_stats.json"

# Compaction (see compact.py), once every group is uploaded: each key's group files are
# rewritten into files of about COMPACT_FILE_BYTES, split only between groups, with row
# groups of about COMPACT_ROW_GROUP_BYTES of uncompressed Arrow data (the Hub's Parquet
# guidance is ~500 MB files and 100-300 MB row groups). The group files are downloaded
# to COMPACT_DIR/src, the rewritten files written to COMPACT_DIR/out, one table at a time.
COMPACT_DIR = Path("compact")
COMPACT_FILE_BYTES = 500 * 1024**2
COMPACT_ROW_GROUP_BYTES = 128 * 1024**2
# Keys are committed in batches, each key's new files and deletions in the same commit
COMPACT_COMMIT_MAX_ADDS = 50
COMPACT_COMMIT_MAX_OPS = 2000
COMPACT_DOWNLOAD_WORKERS = 32

# Local copy of the finalised Hub repos, one directory per table (download-wikidata)
HUB_COPY_DIR = Path("hub")

# Sorting (see sort_by_id.py), after compaction: each key's rows are sorted by its sort
# column (string order, stable) across all its files, from the local copy of the Hub.
# A key over SORT_IN_MEMORY_BYTES of Parquet is sorted through id-range buckets of about
# SORT_BUCKET_BYTES of source Parquet each, working files under SORT_DIR.
SORT_DIR = COMPACT_DIR / "sort"
SORT_IN_MEMORY_BYTES = 2 * 1024**3
SORT_BUCKET_BYTES = 64 * 1024**2

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
