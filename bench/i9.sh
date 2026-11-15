#!/usr/bin/env bash
# The workstation bench (`just bench-i9 PR [BASE]`; bench/README.md, "The workstation
# bench"): the Benchmark workflow's pull-request run (the PR tier, 3 rounds, each a forward
# and a reverse job) reproduced on this machine, the six round jobs one after another, then
# combined as the workflow's finish job combines them. Same harness, same statistics; this
# machine's CPUs, all of them (nothing pins or caps the runs).
#   bash bench/i9.sh 219            # before: git merge-base origin/master HEAD
#   bash bench/i9.sh 219 <sha>      # before: <sha>
#
# Everything goes into one new directory, target/bench-i9/<UTC stamp> (meta.txt, suite.json,
# each job's r<round>-<order>/, summary.md, console.log, outcome.txt: COMPLETE, PARITY FAILED
# or INCOMPLETE: <step>), packed by bench/bundle.sh on exit, whatever the outcome. `just
# submit bench-i9 PR` then archives that bundle and posts it on the pull request.
#
# Kept between attempts (target/bench-i9/): the arms' checkouts and environments (arms/<sha>),
# the bench data at the workflow's DATA_COMMIT (data/), and the dump slice
# (dump-slice.json.bz2, remade when the data or the harness's dumpslice.py changes).
set -euo pipefail

pr=${1:?usage: bench/i9.sh PR [BASE]}
base=${2:-}
die() { echo "bench-i9: $*" >&2; exit 1; }
[[ $pr =~ ^[1-9][0-9]*$ ]] || die "PR is a pull request's number, not '$pr'"

root=$(git rev-parse --show-toplevel)
cd "$root"
top="$root/target/bench-i9"
gh=${GH:-gh}
export UV_PYTHON=${UV_PYTHON:-3.13} ROUNDS=3   # (the workflow's env)

# --- refusals, before anything is made: nothing measured, nothing to submit
git diff --quiet HEAD -- || die "tracked files have uncommitted changes: commit or stash them (the after arm is HEAD as committed)"
git fetch -q origin || die "couldn't fetch origin"
after=$(git rev-parse HEAD)
[ -n "$(git for-each-ref --contains "$after" --format=x refs/remotes/origin)" ] \
  || die "HEAD ($after) is not on origin: push it first (the notice links its commits)"
if [ -n "$base" ]; then
  before=$(git rev-parse --verify -q "$base^{commit}") || die "no commit $base"
else
  before=$(git merge-base origin/master HEAD) || die "no merge base of origin/master and HEAD"
fi
[ "$before" != "$after" ] || die "before and after are the same commit ($after): nothing to compare"
data_commit=$(sed -n 's/^  DATA_COMMIT: \([0-9a-f]*\).*/\1/p' .github/workflows/bench.yml)
[[ $data_commit =~ ^[0-9a-f]{40}$ ]] || die "no DATA_COMMIT in .github/workflows/bench.yml"

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
  echo "bench-i9: $outcome"
  # (the console log closed before it is packed, so tar reads it whole)
  exec 1>&3 2>&4
  wait "$tee_pid" 2>/dev/null
  bash "$root/bench/bundle.sh" "$DIR"
  [ "$outcome" = COMPLETE ] && exit "$code"
  exit $(( code ? code : 1 ))
}
trap finish EXIT

echo "bench-i9: pull request #$pr, before $before, after $after, in $DIR"
{
  echo "pr $pr"
  echo "before $before"
  echo "after $after"
  echo "data $data_commit"
  echo "nproc $(nproc)"
  # (anything set in the environment that the pipeline or the harness reads applies to both arms)
  env | grep -E '^(WIKIDATA|BENCH|POLARS|RAYON|UV_PYTHON)_?[A-Z0-9_]*=' | sort | sed 's/^/env /' || true
  echo "lscpu:"
  lscpu 2>/dev/null || echo "(no lscpu)"
} > "$DIR/meta.txt"

# --- Decoders: the split's lbzip2 and igzip, and pigz for the gz slice (the workflow's apt step)
step="decoders"
for tool in lbzip2 igzip pigz; do
  command -v "$tool" >/dev/null || die "$tool is missing: sudo apt-get install -y lbzip2 isal pigz"
done

# --- Arms: a detached worktree per commit, reused across attempts
step="arms"
git worktree prune
arm() {
  local sha=$1 dir="$top/arms/$1"
  if [ -d "$dir" ]; then
    [ "$(git -C "$dir" rev-parse HEAD 2>/dev/null)" = "$sha" ] || die "$dir is not a checkout of $sha: remove it (git worktree remove --force $dir)"
    git -C "$dir" diff --quiet HEAD -- || die "$dir has changes to tracked files: remove it (git worktree remove --force $dir)"
  else
    mkdir -p "$top/arms"
    git worktree add -q --detach "$dir" "$sha"
  fi
  echo "$dir"
}
a=$(arm "$before")
b=$(arm "$after")
harness="$b/bench"  # (the commit after's harness; the workflow's is the pull request's merge ref)

# --- Prebuilt wheels: each arm's bench/wheels, downloaded into its wheels/ (the workflow's step)
step="prebuilt wheels"
for dir in "$a" "$b"; do
  [ -f "$dir/bench/wheels" ] || continue
  if ! command -v "$gh" >/dev/null || ! "$gh" auth status >/dev/null 2>&1; then
    echo "bench-i9: skipping $dir/bench/wheels: gh is missing or not signed in, so its wheels can't be downloaded (uv sync fails if the lockfile needs them)"
    continue
  fi
  if ls "$dir/wheels/"*.whl >/dev/null 2>&1; then  # (a reused checkout: downloaded by an earlier attempt)
    echo "bench-i9: $dir/wheels already holds $(cd "$dir/wheels" && ls *.whl | tr '\n' ' ')"
    continue
  fi
  grep -Ev '^[[:space:]]*(#|$)' "$dir/bench/wheels" | while read -r repo run artifact; do
    if "$gh" run download "$run" -R "$repo" -n "$artifact" -D "$dir/wheels"; then
      echo "bench-i9: $(basename "$dir"): $artifact of $repo run $run: $(cd "$dir/wheels" && ls *.whl | tr '\n' ' ')"
    else
      echo "bench-i9: skipping $artifact of $repo run $run: gh couldn't download it (uv sync fails if the lockfile needs it)"
    fi
  done
done

# --- Environments: each arm on its own lockfile
step="environments"
(cd "$a" && uv sync --frozen)
(cd "$b" && uv sync --frozen)
py="$b/.venv/bin/python"

# --- Data: the bench data at the workflow's DATA_COMMIT, checked
step="data"
data="$top/data"
git cat-file -e "$data_commit^{commit}" 2>/dev/null || git fetch -q origin bench-data
if [ -d "$data" ]; then
  [ "$(git -C "$data" rev-parse HEAD)" = "$data_commit" ] || git -C "$data" checkout -q --detach "$data_commit"
else
  git worktree add -q --detach "$data" "$data_commit"
fi
(cd "$data" && sha256sum --quiet -c SHA256SUMS)

# --- Dump slice: made by the first run that needs it (pipeline.py, dumpslice.py) and kept,
#     keyed as the workflow's cache is: the data, the workload tier, the harness's dumpslice.py
step="dump slice"
export BENCH_DUMP_SLICE="$top/dump-slice.json.bz2"
key="$data_commit quick $(sha256sum < "$harness/dumpslice.py" | cut -c1-64)"
if [ "$(cat "$top/dump-slice.key" 2>/dev/null)" != "$key" ]; then
  rm -f "$top"/dump-slice.json.bz2 "$top"/dump-slice.json.gz "$top"/dump-slice.orjson.json.bz2
  echo "$key" > "$top/dump-slice.key"
fi

# --- Checks: the PR tier's selection, with no labels and an empty body (the workflow's step)
step="checks"
: > "$DIR/pr-body.txt"
python3 "$harness/tiers.py" "$a" "$b" --suite pr --labels-json '[]' --body-file "$DIR/pr-body.txt" \
  --out "$DIR/suite.json" > /dev/null
deferred=$(python3 -c 'import json, sys; print(",".join(json.load(open(sys.argv[1]))["deferred"]))' "$DIR/suite.json")
echo "bench-i9: the PR tier${deferred:+, then the extras: $deferred}"

# --- The runs: the workflow's six round jobs, one after another. Like its matrix (fail-fast
#     off), a job that fails doesn't stop the others; a job's extras run only after its core
#     passed, as the workflow's later step does
parity_failed() {  # whether a job's results.json has a parity flag False
  python3 - "$1" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
keys = ("parity_ok", "gz_split_parity_ok", "misfit_parity_ok", "evolve_parity_ok", "bz2native_parity_ok")
sys.exit(0 if any(r.get(k) is False for k in keys) else 1)
PY
}
failed_jobs=() parity_jobs=() files=()
for round in 1 2 3; do
  for order in forward reverse; do
    job="r$round-$order"
    step="round $job"
    out="$DIR/$job"
    echo "bench-i9: $job"
    ok=1
    ROUND=$round ORDER=$order "$py" "$harness/ab.py" --data "$data" --out "$out" --suite "$DIR/suite.json" \
      --arm before="$a" --arm after="$b" --rounds 1 --seed "$round" --order "$order" --tier quick || ok=0
    if [ $ok = 1 ] && [ -n "$deferred" ]; then
      step="round $job extras"
      ROUND=$round ORDER=$order "$py" "$harness/ab.py" --data "$data" --out "$out" --resume || ok=0
    fi
    echo "$job $([ $ok = 1 ] && echo ok || echo failed)" >> "$DIR/jobs.txt"
    if [ -f "$out/results.json" ]; then
      files+=("$out/results.json")
      parity_failed "$out/results.json" && parity_jobs+=("$job")
    fi
    [ $ok = 1 ] || failed_jobs+=("$job")
  done
done

# --- Combining: every job's results, as the workflow's finish job does
step="combine"
[ ${#files[@]} -gt 0 ] || die "no round job wrote results.json"
python3 "$harness/combine.py" 3 "$DIR/summary.md" "${files[@]}"
cat "$DIR/summary.md"

if [ ${#parity_jobs[@]} -gt 0 ]; then
  outcome="PARITY FAILED"
  echo "bench-i9: parity failed in ${parity_jobs[*]}"
  exit 1
elif [ ${#failed_jobs[@]} -gt 0 ]; then
  outcome="INCOMPLETE: round ${failed_jobs[*]}"
  exit 1
fi
outcome="COMPLETE"
