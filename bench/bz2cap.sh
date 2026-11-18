#!/usr/bin/env bash
# The split's decoder cap after mibz2 0.5.6 (`just bench-bz2-cap [RELEASE [THREADS [ROUNDS]]]`;
# bench/README.md, "The native decoder's threads"): bench/bz2threads.py's release-shaped
# comparison (bench-full's run, overlapped, from the cached dump) once an arm a round, each
# arm's WIKIDATA_BZ2_THREADS paired with auto (the default cap, 4), its published files'
# digest checked. The workload is bench-bz2-teams's (docs/journal/2026-10-30-bz2-threads-teams.md):
# --release 20261019 --windows 24 --window-mb 25, 8 rounds. The arms by default: auto, 6, 8
# and 9. mibz2 0.5.6 plans one block a thread (Workers) from 6 threads to the physical cores,
# with lazy letters from 8 workers (LAZY_FROM_THREADS), and past the cores on a machine of 8
# to 10 of them one worker a core. On the measured i9-10900K alone (lmmx/mibz2#524, #537) it
# then runs fewer workers than that plan: on the whole machine (as the split runs), 7 lazy
# workers from 10 threads asked, the window 2 blocks a worker; pinned one CPU a core on
# exactly the threads asked, also 6 lazy at 8, 5 eager at 6 and 3 eager at 4. So on the i9,
# unpinned, auto (4), 6, 8 and 9 run as many workers as threads, as with 0.5.2, and 10 runs 7,
# which bz2threads.py's check refuses as not the threads asked: the plans step refuses any
# such arm before the runs. (0.5.5's wider lazy admission and 0.5.6's one-scan writer change
# the kernels, not the plan: the lazy arms take both, the eager ones the writer.)
#   bash bench/bz2cap.sh                       # 20261019, auto,6,8,9, 8 rounds
#   bash bench/bz2cap.sh 20261019 auto,8 4
#
# Refuses, before anything is made, when tracked files have changes, or when the mibz2
# reshape/Cargo.lock locks is not BZ2_CAP_MIBZ2 (default 0.5.6; `any` skips the check).
# Refuses at its plans step, before the dump is fetched or any arm run, an arm whose derived
# decoder runs other threads than asked.
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
threads=${2:-auto,6,8,9}
rounds=${3:-8}
scale="--windows 24 --window-mb 25"
want=${BZ2_CAP_MIBZ2:-0.5.6}
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

# --- The decoder each arm gets: derived, not logged (the split logs the decoder's threads
#     only). mibz2 0.5.6's StreamDecoder::new, as the split builds it (dump._in_process_lines:
#     Bz2Decoder(threads), 2 blocks a thread, Kernels::detect unless WIKIDATA_BZ2_SIMD is
#     off): Plan::for_threads (plan_with) on the physical cores and L3 as mibz2 reads them; its
#     service core (exactly 10 threads, never while the plan there is workers); and its
#     trimmed workers on the measured i9-10900K (trimmed: trim_cell, measured_i9,
#     affinity_admits), the window scaled to them. A derived decoder whose threads are not
#     the threads asked is refused here, before anything is fetched or run: bz2threads.py's
#     check would refuse it mid-run. (BZ2_CAP_ROOT, for the tests alone, reads /proc and /sys
#     under another root.)
step="plans"
uv run --frozen python - "$threads" "${locked:-?}" > "$DIR/plans.txt" <<'PY'
import math, os, platform, re, sys
from pathlib import Path
os.environ.pop("WIKIDATA_BZ2_THREADS", None)
from wikidata.dump import bz2_threads
arms, version = sys.argv[1].split(","), sys.argv[2]
ROOT = os.environ.get("BZ2_CAP_ROOT", "")
SYS = "/sys/devices/system/cpu"
I9 = (("GenuineIntel", 6, 165), "Intel(R) Core(TM) i9-10900K CPU @ 3.70GHz")
TEN_CORES, I9_LOGICAL, L3_20M = 10, 20, 20 << 20
LINUX_X86 = sys.platform.startswith("linux") and platform.machine() in ("x86_64", "AMD64")

def read(p):
    try:
        return Path(ROOT + p).read_text()
    except OSError:
        return None
def num(s):  # Rust's str::parse::<u64> on a trimmed string
    s = s.strip() if s is not None else ""
    return int(s) if re.fullmatch(r"\+?[0-9]+", s) and int(s) < 1 << 64 else None
def cpu_list(s):  # cpu_list: strictly, None for anything else
    if s is None:
        return None
    s, v = s.strip(), set()
    for part in s.split(",") if s else ():
        a, _, b = part.partition("-")
        a, b = num(a), num(b if _ else a)
        if a is None or b is None or a > b or b > 1 << 16:
            return None
        v.update(range(a, b + 1))
    return v
def allowed_list():
    st = read("/proc/self/status")
    return next((l[len("Cpus_allowed_list:"):] for l in (st or "").splitlines() if l.startswith("Cpus_allowed_list:")), None)
def physical_cores():  # physical_cores_of: the plan policy's cores, the affinity's
    s = allowed_list()
    if s is None:
        return None
    cores = set()
    for part in s.strip().split(","):
        a, _, b = part.partition("-")
        a, b = num(a), num(b if _ else a)
        if a is None or b is None:
            return None
        for c in range(a, b + 1):
            t = f"{SYS}/cpu{c}/topology"
            core = read(f"{t}/core_id")
            if core is None:
                continue  # (an allowed CPU that is offline)
            pkg = read(f"{t}/physical_package_id")
            if pkg is None:
                return None
            cores.add((pkg.strip(), core.strip()))
    return len(cores) or None
def cache_size(s):
    s = (s or "").strip()
    if not s:
        return None
    n = num(s[:-1]) if s[-1] in "KM" else num(s)
    return None if n is None else n * {"K": 1 << 10, "M": 1 << 20}.get(s[-1], 1)
def cpu_ids():  # cpu_ids: /proc/cpuinfo's records by processor, strictly
    info = read("/proc/cpuinfo")
    if info is None:
        return None
    ids = {}
    for rec in (r for r in info.split("\n\n") if r.strip()):
        def field(name):
            for line in rec.split("\n"):
                k, sep, v = line.partition(":")
                if sep and k.strip() == name:
                    return v.strip()
            return None
        f = {k: field(k) for k in ("processor", "vendor_id", "cpu family", "model", "model name")}
        if None in f.values() or None in (num(f["processor"]), num(f["cpu family"]), num(f["model"])):
            return None
        n = num(f["processor"])
        if n in ids:
            return None
        ids[n] = ((f["vendor_id"], num(f["cpu family"]), num(f["model"])), f["model name"])
    return ids
def topology(among):  # topology_of: (phys, logical, l3, cpu, name) of the present, online CPUs
    online, present = cpu_list(read(f"{SYS}/online")), cpu_list(read(f"{SYS}/present"))
    if online is None or present is None:
        return None
    cpus = sorted(c for c in present if c in online and (among is None or c in among))
    ids = cpu_ids() if cpus else None
    if ids is None:
        return None
    cores, l3, cpu = set(), None, None
    for c in cpus:
        t = f"{SYS}/cpu{c}"
        core, pkg = num(read(f"{t}/topology/core_id")), num(read(f"{t}/topology/physical_package_id"))
        if core is None or pkg is None or num(read(f"{t}/cache/index3/level")) != 3:
            return None
        cores.add((pkg, core))
        size = cache_size(read(f"{t}/cache/index3/size"))
        if size is None or l3 not in (None, size) or c not in ids or cpu not in (None, ids[c]):
            return None
        l3, cpu = size, ids[c]
    return (len(cores), len(cpus), l3, cpu[0], cpu[1])
def parse_cpu_max(s):  # "max" for no limit, a number of CPUs, or None (unknown)
    f = (s[:-1] if s.endswith("\n") else s).split(" ")
    pos = lambda t: int(t) if re.fullmatch(r"[0-9]+", t) and 0 < int(t) < 1 << 64 else None
    if len(f) != 2 or pos(f[1]) is None:
        return None
    return "max" if f[0] == "max" else (None if pos(f[0]) is None else pos(f[0]) / pos(f[1]))
def quota_unlimited():  # cpu_quota_of == Quota::Unlimited: cgroup v2, to the hierarchy's true root
    cg, info = read("/proc/self/cgroup"), read("/proc/self/mountinfo")
    v2 = [l[3:] for l in (cg or "").splitlines() if l.startswith("0::")]
    if cg is None or len(v2) != 1:
        return False
    path = v2[0].strip()
    if not path.startswith("/") or ".." in path or " " in path or info is None:
        return False
    mounts = [pre for pre, sep, post in (l.partition(" - ") for l in info.splitlines())
              if sep and post.split()[:1] == ["cgroup2"]]
    if len(mounts) != 1 or len(mounts[0].split()) < 5:
        return False
    root, point = mounts[0].split()[3:5]
    if "\\" in root or "\\" in point or not point.startswith("/"):
        return False
    if root == "/":
        rel = path
    elif path.startswith(root) and (path == root or path[len(root)] == "/"):
        rel = path[len(root):]
    else:
        return False
    at = point.rstrip("/")
    levels = [at] + [at := f"{at}/{p}" for p in rel.split("/") if p]
    q = "max"
    for d in levels:
        if not d or not Path(ROOT + d).is_dir():
            return False
        try:
            text = Path(ROOT + d + "/cpu.max").read_text()
        except FileNotFoundError:
            continue
        except OSError:
            return False
        b = parse_cpu_max(text)
        if b is None:
            return False
        q = b if q == "max" else (q if b == "max" else min(q, b))
    return q == "max" and not Path(ROOT + point.rstrip("/") + "/cgroup.type").exists()
def kernels():  # Kernels::detect, unless WIKIDATA_BZ2_SIMD is off (reshape's bz2_py.rs)
    if os.environ.get("WIKIDATA_BZ2_SIMD") in ("0", "off") or not LINUX_X86:
        return (False, False)
    flags = next((l.split(":", 1)[1].split() for l in (read("/proc/cpuinfo") or "").splitlines()
                  if l.startswith("flags")), [])
    return ({"avx2", "bmi1"} <= set(flags), {"pclmulqdq", "sse4_1", "ssse3"} <= set(flags))
phys, l3 = physical_cores(), cache_size(read(f"{SYS}/cpu0/cache/index3/size"))
allowed = cpu_list(allowed_list())
target = topology(allowed) if allowed is not None else None
machine = topology(None)
avx2, clmul = kernels()
def is_i9(t):  # i9_profile's (and service_target's) snapshot test
    return t is not None and (t[3], t[4]) == I9 and t[:3] == (TEN_CORES, I9_LOGICAL, L3_20M)
ids = cpu_ids() or {}
may_be_i9 = sum(1 for v in ids.values() if v == I9) >= I9_LOGICAL
measured_i9 = LINUX_X86 and avx2 and clmul and may_be_i9 and is_i9(machine) and quota_unlimited()
def plan_with(threads):  # mibz2 0.5.6, src/parallel.rs, plan_with (unchanged since 0.5.1)
    if phys is None:
        return ("workers", threads)
    cores = min(threads, phys)
    if cores < 6 or (threads < 11 and threads <= phys):
        return ("workers", threads)
    if threads > phys and 8 <= phys < 11:
        return ("workers", phys)
    if phys == TEN_CORES and l3 == L3_20M and threads >= 10:
        return ("teams", 3, 3)
    cap = max(1, math.floor(0.7 * l3 / 3.6e6 + 0.5)) if l3 is not None else 4
    return ("teams", max(1, min(cores // 2, cap)), 2)
def trimmed(threads, plan):  # trim_cell, then measured_i9 and affinity_admits
    cell = {"ten": (7, True), 8: (6, True), 6: (5, False), 4: (3, False)}
    key = "ten" if threads >= 10 and plan == ("workers", 10) else threads
    if key not in cell or (key != "ten" and plan != ("workers", threads)) or not measured_i9:
        return None
    pinned = target is not None and target[0] == threads and target[1] == threads
    whole = target is not None and target == machine
    return cell[key] if pinned or (whole and threads >= 10) else None
def decoder(threads):  # StreamDecoder::new(threads, 2 * threads, k): (decoder threads, what it is)
    slots, plan = 2 * threads, plan_with(threads)
    if threads == 10 and plan[0] != "workers" and avx2 and clmul and LINUX_X86 and may_be_i9 \
            and is_i9(target) and quota_unlimited():
        return 8, f"the service core (2 teams of 3, 2 service workers; window {max(slots * 2 // threads, 2)})"
    cell = trimmed(threads, plan)
    if cell:
        n, lazy = cell
        return n, (f"Workers({n}){' (lazy letters)' if lazy else ''}, trimmed on the measured i9 "
                   f"(window {max(slots * n // threads, n)})")
    if plan[0] == "workers":
        return plan[1], f"Workers({plan[1]}){' (lazy letters)' if plan[1] >= 8 else ''}"
    _, teams, k = plan
    piped = " (pipelined)" if (teams, k) == (3, 3) else ""
    return teams * k, f"Teams {{ teams: {teams}, k: {k} }}{piped}"
where = ("all of its CPUs" if target is not None and target == machine else
         f"{target[1]} CPUs on {target[0]} cores" if target else "an affinity not read")
print(f"# mibz2 {version}'s StreamDecoder::new, derived for {phys} physical cores, a "
      f"{l3 and l3 >> 20} MiB L3, {where}, {'the measured i9' if measured_i9 else 'not the measured i9'} "
      f"(not logged by the split)")
refused = []
for a in arms:
    n = bz2_threads() if a == "auto" else int(a)
    used, what = decoder(n)
    print(f"{a} {n} {what}" + (f": {used} decoder threads, not {n}" if used != n else ""))
    if used != n:
        refused.append(f"{a} ({n} threads asked, {used} run)")
if refused:
    print(f"bench-bz2-cap: mibz2 {version} runs fewer or other decoder threads than asked at "
          f"{', '.join(refused)}, which bz2threads.py's check refuses: leave those arms out",
          file=sys.stderr)
    sys.exit(1)
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
