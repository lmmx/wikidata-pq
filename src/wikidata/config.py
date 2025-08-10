"""Configuration constants for the wikidata processing pipeline."""

from enum import StrEnum
from pathlib import Path

# HuggingFace repository identifier
REPO_ID = "philippesaade/wikidata"

# Directory names
STATE_DIR = Path("state")
LOCAL_DATA_DIR = Path("data")
OUTPUT_DIR = Path("results")


# Table types
class Table(StrEnum):
    LABEL = "labels"
    DESC = "descriptions"
    ALIAS = "aliases"
    LINKS = "links"
    CLAIMS = "claims"


TABLE_TYPES = list(Table)
