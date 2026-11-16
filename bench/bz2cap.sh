#!/usr/bin/env bash
# The split's decoder cap after mibz2 0.5.1 (`just bench-bz2-cap [RELEASE [THREADS [ROUNDS]]]`;
# bench/README.md, "The native decoder's threads"): bench/bz2threads.py's release-shaped
# comparison (bench-full's run, overlapped, from the cached dump) once an arm a round, each
# arm's WIKIDATA_BZ2_THREADS paired with auto (the default cap, 4), its published files'
# digest checked. The workload is bench-bz2-teams's (docs/journal/2026-10-30-bz2-threads-teams.md):
# --release 20261019 --windows 24 --window-mb 25, 8 rounds. The arms by default: auto, 6, 8,
# 9, 10 and 20. mibz2 0.5.1 plans one block a thread (Workers, Cells) from 6 threads to the
# physical cores, with lazy letters from 8 workers, and past the cores on a machine of 8 to
# 10 of them one worker a core; so on the i9 (10 cores, 20 MiB L3) every arm but auto (4,
# teams) runs workers, and 20 runs 10 of them.
#   bash bench/bz2cap.sh                       # 20261019, auto,6,8,9,10,20, 8 rounds
#   bash bench/bz2cap.sh 20261019 auto,8 4
#
# Refuses, before anything is made, when tracked files have changes, or when the mibz2
# reshape/Cargo.lock locks is not BZ2_CAP_MIBZ2 (default 0.5.1; `any` skips the check).
#
# Everything goes into one new directory, target/bench-bz2-cap/<UTC stamp> (meta.txt,
# plans.txt, bz2threads.py's machine.json, runs.jsonl and summary.md, each run's
# full-r<round>-<arm>/, console.log, outcome.txt: COMPLETE, PARITY FAILED or INCOMPLETE:
# <step>), packed by bench/bundle.sh on exit, whatever the outcome. `just submit
# bench-bz2-cap` then archives the bundle and posts it on the queue issue (meta.txt names no
# pull request). The dump stays in bench-full's cache (bench/full/run.py: WIKIDATA_BENCH_CACHE,
# else ~/.cache/wikidata-pq/bench-full), fetched first if it isn't there; each run's work
# directory is removed when it ends.
set -euo pipefail

release=${1:-20261019}
threads=${2:-auto,6,8,9,10,20}
rounds=${3:-8}
scale="--windows 24 --window-mb 25"
want=${BZ2_CAP_MIBZ2:-0.5.1}
die() { echo "bench-bz2-cap: $*" >&2; exit 1; }
[[ $release =~ ^[0-9]{8}$ ]] || die "RELEASE is a dump's date (20261019), not '$release'"
[[ $threads =~ ^(auto|[1-9][0-9]*)(,(auto|[1-9][0-9]*))*$ ]] || die "THREADS is counts and auto, comma-separated, not '$threads'"
[[ $rounds =~ ^[1-9][0-9]*$ ]] || die "ROUNDS is a positive count, not '$rounds'"

root=$(git rev-parse --show-toplevel)
cd "$root"
top="$root/target/bench-bz2-cap"

# --- refusals, before anything is made: nothing measured, nothing to submit
git diff --quiet HEAD -- || die "tracked files have uncommitted changes: commit or stash them (the run is HEAD as committed)"
locked=$(awk '$0 == "name = \"mibz2\"" {getline; gsub(/version = |"/, ""); print; exit}' reshape/Cargo.lock)
if [ "$want" != any ] && [ "$locked" != "$want" ]; then
  die "reshape/Cargo.lock locks mibz2 ${locked:-(none)}, not $want: run the commit that bumps it (BZ2_CAP_MIBZ2=any, or the version wanted, to override)"
fi

# --- the attempt's directory, new every time, and its console log
mkdir -p "$top"
DIR="$top/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir "$DIR"
exec 3>&1 4>&2
exec > >(tee -a "$DIR/console.log") 2>&1
tee_pid=$!

step="setup"
outcome=""
finish() {
  code=$?
  set +e
  [ -n "$outcome" ] || outcome="INCOMPLETE: $step"
  echo "$outcome" > "$DIR/outcome.txt"
  echo "bench-bz2-cap: $outcome"
  # (the console log closed before it is packed, so tar reads it whole)
  exec 1>&3 2>&4
  wait "$tee_pid" 2>/dev/null
  bash "$root/bench/bundle.sh" "$DIR"
  [ "$outcome" = COMPLETE ] && exit "$code"
  exit $(( code ? code : 1 ))
}
trap finish EXIT

head=$(git rev-parse HEAD)
echo "bench-bz2-cap: $head, mibz2 ${locked:-?}, release $release ($scale), arms $threads, $rounds rounds, in $DIR"

# --- The environment: the checkout's lockfile, its extension rebuilt so the decoder is the
#     mibz2 Cargo.lock locks (uv's cache key for it is Cargo.toml, not Cargo.lock)
step="environment"
uv sync --frozen --reinstall-package wikidata-reshape

# --- The plan each arm gets: derived, not logged (the split logs the decoder's threads only).
#     mibz2 0.5.1's Plan::for_threads (plan_with) on this machine's physical cores and L3, as
#     mibz2 reads them (0.5.1's service core stands down whenever the plan is workers)
step="plans"
uv run --frozen python - "$threads" "${locked:-?}" > "$DIR/plans.txt" <<'PY'
import os, sys
from pathlib import Path
os.environ.pop("WIKIDATA_BZ2_THREADS", None)
from wikidata.dump import bz2_threads
arms, version = sys.argv[1].split(","), sys.argv[2]
cores = set()
for c in sorted(os.sched_getaffinity(0)):
    t = Path(f"/sys/devices/system/cpu/cpu{c}/topology")
    try:
        cores.add(((t / "physical_package_id").read_text().strip(), (t / "core_id").read_text().strip()))
    except OSError:
        pass
phys = len(cores) or None
l3 = None
size = Path("/sys/devices/system/cpu/cpu0/cache/index3/size")
if size.exists():
    s = size.read_text().strip()
    l3 = int(s[:-1]) * {"K": 1 << 10, "M": 1 << 20}[s[-1]] if s[-1] in "KM" else int(s)
def plan(threads):  # mibz2 0.5.1, src/parallel.rs, plan_with (and Plan::lazy)
    def workers(n):
        return f"Workers({n}){' (lazy letters)' if n >= 8 else ''}"
    if phys is None:
        return workers(threads)
    use = min(threads, phys)
    if use < 6 or (threads < 11 and threads <= phys):
        return workers(threads)
    if threads > phys and 8 <= phys < 11:
        return workers(phys) + ", one a physical core"
    if phys == 10 and l3 == 20 << 20 and threads >= 10:
        return "Teams { teams: 3, k: 3 } (pipelined)"
    cap = max(1, round(0.7 * l3 / 3.6e6)) if l3 else 4
    return f"Teams {{ teams: {max(1, min(use // 2, cap))}, k: 2 }}"
print(f"# mibz2 {version}'s Plan::for_threads, derived for {phys} physical cores and a "
      f"{l3 and l3 >> 20} MiB L3 (not logged by the split)")
for a in arms:
    n = bz2_threads() if a == "auto" else int(a)
    print(f"{a} {n} {plan(n)}")
PY
cat "$DIR/plans.txt"

{
  echo "harness $head"
  echo "mibz2_locked ${locked:-?}"
  plans=$(awk '!/^#/ {a = $1; n = $2; $1 = $2 = ""; sub(/^ +/, ""); printf "%s%s=%s %s", sep, a, n, $0; sep = "; "}' "$DIR/plans.txt")
  echo "note mibz2 ${locked:-?}; arms $threads, $rounds rounds; plans (derived): $plans"
  echo "release $release $scale"
  echo "nproc $(nproc)"
  # (anything set in the environment that the pipeline or the harness reads applies to every arm)
  env | grep -E '^(WIKIDATA|BENCH|POLARS|RAYON|UV_PYTHON)_?[A-Z0-9_]*=' | sort | sed 's/^/env /' || true
  echo "lscpu:"
  lscpu 2>/dev/null || echo "(no lscpu)"
} > "$DIR/meta.txt"

# --- The dump: bench-full's cache, at the run's scale (a no-op when it is cached)
step="fetch"
uv run --frozen python bench/full/run.py fetch --release "$release" $scale

# --- The runs: bz2threads.py --full, its balanced order, its checks and its summary
step="runs"
ok=1
uv run --frozen python bench/bz2threads.py --full "--release $release $scale" --threads "$threads" \
  --rounds "$rounds" --out "$DIR" || ok=0

[ -f "$DIR/summary.md" ] || die "bz2threads.py wrote no summary"
step="summary"
{
  echo
  echo "## The plan each arm gets (derived, not logged)"
  echo
  sed 's/^# //; s/^\([^ ]*\) \([0-9]*\) \(.*\)$/- `\1` (\2 threads): \3/' "$DIR/plans.txt"
} >> "$DIR/summary.md"
cat "$DIR/summary.md"

if [ -f "$DIR/runs.jsonl" ] && python3 -c '
import json, sys
recs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
sys.exit(0 if any(not (r.get("ok") and r.get("same_as_first")) for r in recs) else 1)' "$DIR/runs.jsonl"; then
  outcome="PARITY FAILED"
  exit 1
fi
[ $ok = 1 ] || { outcome="INCOMPLETE: runs"; exit 1; }
outcome="COMPLETE"
