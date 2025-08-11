"""Configuration constants for the wikidata processing pipeline."""

from enum import StrEnum
from pathlib import Path

# HuggingFace repository identifier
REPO_ID = "philippesaade/wikidata"
REMOTE_REPO_PATH = "data"

# Directory names
STATE_DIR = Path("state")
LOCAL_DATA_DIR = Path("data")
OUTPUT_DIR = Path("results")

# Filename chunk/part regex patterns
CHUNK_RE = r"chunk_(\d+)-"
PART_RE = r"chunk_\d+-(\d+)"


# Table types
class Table(StrEnum):
    LABEL = "labels"
    DESC = "descriptions"
    ALIAS = "aliases"
    LINKS = "links"
    CLAIMS = "claims"


HF_USER = "permutans"

REPO_TARGET = "{hf_user}/wikidata-{tbl}"
