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


# Maps each table type to its partition column. Four tables (labels, descriptions,
# aliases, claims) are partitioned by language because their rows have a language code
# from the multilingual map normalisation. Links is partitioned by site because
# sitelinks use site codes like enwiki, frwiki rather than bare language codes.
PARTITION_COLS = {
    Table.LABEL: "language",
    Table.DESC: "language",
    Table.ALIAS: "language",
    Table.LINKS: "site",
    Table.CLAIMS: "language",
    Table.CLAIMS_LABELS: "language",
}

HF_USER = "permutans"

REPO_TARGET = "{hf_user}/wikidata-{tbl}"

# Prefetch (background download) settings
PREFETCH_ENABLED = True
# “fill up to” this much source data locally
PREFETCH_BUDGET_GB = 300.0
# Never go more than N chunks ahead
PREFETCH_MAX_AHEAD = 50
# Skip prefetch if disk tighter than this
PREFETCH_MIN_FREE_GB = 50.0
