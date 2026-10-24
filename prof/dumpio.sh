#!/usr/bin/env bash
# The dump's download and decompression, bz2 against gz, on a CI runner: each file's
# download speed from Wikimedia and a mirror (a 1 GB range, one connection and two), and
# each decoder's wall and CPU time over the first GB of each file, its output counted.
set -u
# The newest release with both files
for R in $(curl -sS https://dumps.wikimedia.org/wikidatawiki/entities/ | grep -o 'href="[0-9]\{8\}/"' | tr -dc '0-9\n' | sort -r); do
  ok=1
  for ext in bz2 gz; do
    n=$(curl -sSI "https://dumps.wikimedia.org/wikidatawiki/entities/$R/wikidata-$R-all.json.$ext" | tr -d '\r' | awk 'tolower($1)=="content-length:"{print $2}' | tail -1)
    [ "${n:-0}" -gt 10000000000 ] || ok=0
  done
  [ $ok = 1 ] && break
done
echo "release $R"; grep "model name" /proc/cpuinfo | head -1; nproc
BASES="https://dumps.wikimedia.org/wikidatawiki/entities https://ftp.acc.umu.se/mirror/wikimedia.org/other/wikibase/wikidatawiki"
GB=1000000000
for base in $BASES; do
  for ext in bz2 gz; do
    url="$base/$R/wikidata-$R-all.json.$ext"
    size=$(curl -sSI "$url" | tr -d '\r' | awk 'tolower($1)=="content-length:"{print $2}' | tail -1)
    one=$(curl -sS -r $GB-$((2*GB-1)) -o /dev/null -w '%{speed_download}' --max-time 300 "$url")
    a=$(mktemp); b=$(mktemp)
    t=$(date +%s.%N)
    curl -sS -r $GB-$((GB+GB/2-1)) -o /dev/null --max-time 300 "$url" & curl -sS -r $((GB+GB/2))-$((2*GB-1)) -o /dev/null --max-time 300 "$url"; wait
    two=$(echo "$GB / ($(date +%s.%N) - $t) / 1e6" | bc -l)
    printf "download %s %s: size %s, one connection %.1f MB/s, two %.1f MB/s\n" "$base" "$ext" "$size" "$(echo "$one/1e6" | bc -l)" "$two"
  done
done
mkdir -p io
for ext in bz2 gz; do curl -sS -r 0-$((GB-1)) -o io/head.json.$ext "https://dumps.wikimedia.org/wikidatawiki/entities/$R/wikidata-$R-all.json.$ext"; done
cat io/head.json.* > /dev/null
run() {  # name, file, command...: the decoder's wall, user and sys time, and its output
  local name=$1 f=$2; shift 2
  out=$( { /usr/bin/time -f "%e %U %S" -o io/t "$@" < "$f" 2>/dev/null | wc -c; } )
  read -r e u s < <(tail -1 io/t)
  printf "decode %-12s wall %6.2fs user %6.2fs sys %5.2fs  %5.0f MB out  %6.0f MB/s of wall  %6.0f MB/s of CPU\n" \
    "$name" "$e" "$u" "$s" "$(echo "$out/1e6" | bc -l)" "$(echo "$out/1e6/$e" | bc -l)" "$(echo "$out/1e6/($u+$s)" | bc -l)"
}
PY=.venv/bin/python
for i in 1 2; do
  run lbzip2 io/head.json.bz2 lbzip2 -dc
  run lbzip2-n1 io/head.json.bz2 lbzip2 -n1 -dc
  run bzip2 io/head.json.bz2 bzip2 -dc
  run gzip io/head.json.gz gzip -dc
  run pigz io/head.json.gz pigz -dc
  run igzip io/head.json.gz igzip -dc
  run py-zlib io/head.json.gz $PY -c "
import sys, zlib
d = zlib.decompressobj(31); r = sys.stdin.buffer; w = sys.stdout.buffer
while b := r.read(1 << 20): w.write(d.decompress(b))"
  run py-isal io/head.json.gz $PY -c "
import sys
from isal import isal_zlib
d = isal_zlib.decompressobj(31); r = sys.stdin.buffer; w = sys.stdout.buffer
while b := r.read(1 << 20): w.write(d.decompress(b))"
  run py-zlib-ng io/head.json.gz $PY -c "
import sys
from zlib_ng import zlib_ng
d = zlib_ng.decompressobj(31); r = sys.stdin.buffer; w = sys.stdout.buffer
while b := r.read(1 << 20): w.write(d.decompress(b))"
done
