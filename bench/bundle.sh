#!/usr/bin/env bash
# One file from a workstation run: DIR packed as DIR.tar.gz beside it, symbolic links (a
# shared corpus) and files over 50 MB left out, then its path and size printed. Every
# workstation recipe ends with it (`just bench-i9`), and `just submit` archives the file
# it makes (bench/submit.sh).
#   bash bench/bundle.sh target/bench-i9/20261107T100854Z
set -euo pipefail
dir=$(realpath "${1:?usage: bench/bundle.sh DIR}")
out="$dir.tar.gz"
( cd "$(dirname "$dir")" && find "$(basename "$dir")" -type f -size -50M -print0 \
  | tar --null -czf "$out.tmp" -T - )
mv "$out.tmp" "$out"
echo
echo "The bundle: $(realpath "$out") ($(du -h "$out" | cut -f1))"
