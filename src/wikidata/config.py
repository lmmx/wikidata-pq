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

# Filename chunk/part regex patterns
CHUNK_RE = r"chunk_(\d+)-"
PART_RE = r"chunk_\d+-(\d+)"

# Use to replace the capture group of a regex
CAPTURE_GROUP_RE = r"\(([^)]*)\)"


# Table types
class Table(StrEnum):
    LABEL = "labels"
    DESC = "descriptions"
    ALIAS = "aliases"
    LINKS = "links"
    CLAIMS = "claims"


HF_USER = "permutans"

REPO_TARGET = "{hf_user}/wikidata-{tbl}"

# Prefetch (background download) settings
PREFETCH_ENABLED = True
# “fill up to” this much source data locally
PREFETCH_BUDGET_GB = 300.0
# Never go more than N chunks ahead
PREFETCH_MAX_AHEAD = 3
# Skip prefetch if disk tighter than this
PREFETCH_MIN_FREE_GB = 50.0
