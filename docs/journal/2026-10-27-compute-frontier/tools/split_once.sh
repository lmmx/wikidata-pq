#!/bin/sh
# One split-dump of a dump slice, as the bench's split_stage runs it, timed by rt.py:
# wall, user and sys CPU of the whole tree, peak RSS, page faults.
# Usage: split_once.sh PYTHON SRC_DIR SLICE.json.bz2 WORK_DIR
py=$1 src=$2 slice=$3 d=$4
here=$(cd "$(dirname "$0")" && pwd)
rm -rf "$d"; mkdir -p "$d/releases/20261012/dump"
ln "$slice" "$d/releases/20261012/dump/wikidata-20261012-all.json.bz2"
cd "$d" && WIKIDATA_RELEASE=20261012 HF_HUB_OFFLINE=1 python3 "$here/rt.py" "$py" -c \
  "import sys; sys.path.insert(0, '$src'); from wikidata import dump; dump.split()"
